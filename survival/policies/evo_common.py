"""evo_common (n3 S1 lane, Sat 19 Sep): bb_juke + switchable FOOD-COLLAPSE mechanisms.
All switches OFF reproduces bb_juke bit-for-bit (same code path, only bookkeeping added).
Each evo_<name>.py turns one mechanism on via make(**kw). Params are in EVO below; EVO_PARAMS='{"ripe_s": 14}' overrides.

Bookkeeping (per agent, exact from the simulator's cost code, environment.py:496-571, 623-647):
  predicted energy after our action = e_prev - act_cost - 0.1 (base drain, all biomes drain 1.0/s)
  residual r = e_now - predicted:  r > +0.5  -> a fruit was eaten this step (gain ~ r, capped at max_energy)
                                   r ~ -0.01*age -> the agent is past its hidden max_age (elder), age >= 55
Mechanisms:
  patience  (T-food): a hungry agent above ripe_min_frac*max_e, not sprint-locked, not a newborn, ignores a fruit it
            SAW APPEAR (new track within its hearing radius while it stood still >= 2 steps) less than ripe_s ago.
            Fruit energy grows 20 -> 60 over 20 s, so a proven-young fruit is worth up to +40 for waiting ~1/s.
            Fruits of unknown age (first seen while moving / in the far cone) are eaten as in bb_juke.
  elder     : detected elders (i) spawn as soon as energy > elder_spawn_energy (an elder drains ~0.01*age/step, so
            its energy is lost anyway; the newborn gets 75), allowed elder_over_cap over max_herd; (ii) after
            elder_yield_t0 an elder does not target fruit while a hungry (< yield_frac) non-elder sibling is
            within yield_d px.
  lock      : a birth may not leave the parent below max_e/5 + lock_margin (sprint lock), elders and the n<=2 emergency exempt.
  supply    : after supply_t0, max_herd = clamp(herd fruit income over supply_win s / supply_e_per_agent, supply_min_herd, 12).
  birth     : after birth_t0 an agent spawns only if it ATE within birth_recent_s (elders and n<=2 exempt).
  camp      : after camp_t0 no dispersal to the 2nd tree and no crowd-wander; leave a tree only when no fruit has been
            visible for camp_barren_s and another tree is visible.
Diag counters in self.diag (evo_diag.py prints them)."""
import math
import os
import sys
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bb_juke  # noqa: E402

TWO_PI = 2 * math.pi

EVO = dict(
    patience=0, ripe_s=18.0, ripe_min_frac=0.35, ripe_match=8.0, ripe_still=0.5, ripe_hear_margin=5.0, ripe_track_mem=30,
    ripe_mode=0,  # 0: new fruit = unseen track while still >= 2 steps and within hearing; 1: motion-aware, d + last move + 1 <= hearing
    # 2: mode 1 plus: an agent camped at its tree for >= ripe_camp_steps has swept the whole circle with its scan (0.08 rad/step,
    #    78 steps), so a new track within ripe_vis_frac*vision at that tree is <= ~8 s old (age estimate ripe_vis_age s)
    ripe_camp_steps=80, ripe_vis_frac=0.9, ripe_vis_age=4.0, ripe_tree_d=15.0,
    elder=0, elder_from_age=55.0, elder_tol=0.25, elder_steps=2, elder_spawn=1, elder_spawn_energy=101.0, elder_over_cap=2,
    elder_yield=1, elder_yield_t0=600.0, yield_d=40.0, yield_frac=0.5,
    lock=0, lock_margin=15.0,
    supply=0, supply_t0=600.0, supply_win=60.0, supply_e_per_agent=2.5, supply_min_herd=4,
    birth=0, birth_t0=600.0, birth_recent_s=20.0,
    camp=0, camp_t0=600.0, camp_barren_s=40.0,
)
if os.environ.get("EVO_PARAMS"):
    import json
    EVO.update(json.loads(os.environ["EVO_PARAMS"]))


def act_cost(a, act):
    """Exact energy the simulator charges for this action (before the 0.1 base drain and aging)."""
    speed, sprint = a["speed"], a["sprint_speed"]
    d = max(0.0, min(act["move_distance"], sprint))
    if a["energy"] < a["max_energy"] / 5 and d > speed:
        d = speed
    c = d * 0.05 if d <= speed else speed * 0.05 + (d - speed) * 0.5
    c += min(math.pi, abs(act["turn_angle"])) / TWO_PI
    return c


class Hivemind(bb_juke.Hivemind):
    def __init__(self, params=None):
        q = dict(EVO)
        if params:
            q.update(params)
        super().__init__(q)
        self._max_herd0 = self.p["max_herd"]
        self.eats = deque()  # (step, gain) herd-wide
        self._cap_step = -1
        self.diag.update({"eat": 0, "eat_gain": 0.0, "elder_flag": 0, "elder_spawn": 0, "elder_yield": 0,
                          "ripe_skip": 0, "ripe_new": 0, "ripe_new_vis": 0, "ripe_unknown": 0, "lock_block": 0, "birth_block": 0,
                          "supply_cap_sum": 0.0, "supply_steps": 0, "camp_leave": 0, "spawns": 0})

    # ------------------------------------------------------------------ bookkeeping
    def _book(self, a, m):
        p = self.p
        e, age = a["energy"], a["age"]
        m.setdefault("elder", False)
        m.setdefault("last_eat", -10 ** 9)
        m.setdefault("still", 0)
        if "e_prev" in m:
            pred = m["e_prev"] - m["cost_prev"] - 0.1
            r = e - pred
            if m.get("spawned_prev"):
                pass  # -100 step: no inference
            elif r > 0.5:
                m["last_eat"] = self.step
                self.eats.append((self.step, r))
                self.diag["eat"] += 1; self.diag["eat_gain"] += r
            elif (p["elder"] or p["lock"] or p["birth"]) and not m["elder"] and age >= p["elder_from_age"]:
                if abs(r + 0.01 * age) < p["elder_tol"]:
                    m["elder_n"] = m.get("elder_n", 0) + 1
                    if m["elder_n"] >= p["elder_steps"]:
                        m["elder"] = True; self.diag["elder_flag"] += 1
                else:
                    m["elder_n"] = 0
        if p["patience"]:
            self._fruit_tracks(a, m)

    def _fruit_tracks(self, a, m):
        p = self.p
        X, Y, H = m["X"], m["Y"], m["H"]
        tr = m.setdefault("ftracks", [])
        still = m["still"] >= 2
        hear = a["hearing_radius"] - p["ripe_hear_margin"]
        if p["ripe_mode"] >= 1:
            still = True
            hear = a["hearing_radius"] - 1.0 - m.get("mv1", 0.0)
        camped = False
        if p["ripe_mode"] == 2:
            tr_d = min((o["distance"] for o in a["observations"] if o["type"] == "Tree"), default=1e9)
            if tr_d <= p["tree_stay"] + p["ripe_tree_d"]:
                m.setdefault("tree_since", self.step)
            else:
                m.pop("tree_since", None)
            camped = "tree_since" in m and self.step - m["tree_since"] >= p["ripe_camp_steps"]
            vis = p["ripe_vis_frac"] * a["vision_range"]
        used = set()
        for o in a["observations"]:
            if o["type"] != "Fruit":
                continue
            d, th = o["distance"], o["angle"]
            wx, wy = X + d * math.cos(H + th), Y + d * math.sin(H + th)
            best, bd = None, p["ripe_match"]
            for i, t in enumerate(tr):
                if i in used:
                    continue
                dd = math.hypot(wx - t["wx"], wy - t["wy"])
                if dd < bd:
                    best, bd = i, dd
            if best is not None:
                used.add(best); tr[best]["seen"] = self.step
                o["_age"] = None if tr[best]["first"] is None else (self.step - tr[best]["first"]) * 0.1
            else:
                new = still and d <= hear
                first = self.step if new else None
                if not new and camped and d <= vis:
                    new = True; first = self.step - int(p["ripe_vis_age"] * 10)
                    self.diag["ripe_new_vis"] += 1
                tr.append({"wx": wx, "wy": wy, "first": first, "seen": self.step})
                used.add(len(tr) - 1)
                o["_age"] = None if first is None else (self.step - first) * 0.1
                self.diag["ripe_new" if new else "ripe_unknown"] += 1
        m["ftracks"] = [t for t in tr if self.step - t["seen"] <= p["ripe_track_mem"]]

    def _record(self, a, m, act):
        m["e_prev"] = a["energy"]
        m["cost_prev"] = act_cost(a, act)
        m["spawned_prev"] = bool(act["spawn_agent"]) and a["energy"] > 100
        m["still"] = m["still"] + 1 if act["move_distance"] <= self.p["ripe_still"] else 0
        m["mv1"] = max(0.0, min(act["move_distance"], a["sprint_speed"]))
        if act["spawn_agent"]:
            self.diag["spawns"] += 1

    def _supply_cap(self, n):
        p = self.p
        t = self.step * 0.1
        if t < p["supply_t0"]:
            p["max_herd"] = self._max_herd0
            return
        w = int(p["supply_win"] * 10)
        while self.eats and self.eats[0][0] < self.step - w:
            self.eats.popleft()
        income = sum(g for _s, g in self.eats) / p["supply_win"]
        cap = int(income / p["supply_e_per_agent"])
        cap = max(p["supply_min_herd"], min(self._max_herd0, cap))
        p["max_herd"] = cap
        self.diag["supply_cap_sum"] += cap; self.diag["supply_steps"] += 1

    # ------------------------------------------------------------------ hooks
    def _act_one(self, a, m, n, spawned, fits, med, stats, cur_by, fde):
        p = self.p
        if p["supply"] and self._cap_step != self.step:
            self._cap_step = self.step
            self._supply_cap(n)
        self._book(a, m)
        self._cur_stats = stats
        act, spawned = super()._act_one(a, m, n, spawned, fits, med, stats, cur_by, fde)
        e, max_e = a["energy"], a["max_energy"]
        t = self.step * 0.1
        if act["spawn_agent"] and n + spawned > 2 and not m["elder"]:
            if p["lock"] and e - 100 < max_e / 5 + p["lock_margin"]:
                act["spawn_agent"] = False; spawned -= 1; self.diag["lock_block"] += 1
            elif p["birth"] and t >= p["birth_t0"] and (self.step - m["last_eat"]) * 0.1 > p["birth_recent_s"]:
                act["spawn_agent"] = False; spawned -= 1; self.diag["birth_block"] += 1
        if p["elder"] and p["elder_spawn"] and m["elder"] and not act["spawn_agent"] and e > p["elder_spawn_energy"] \
                and n + spawned < p["max_herd"] + p["elder_over_cap"]:
            act["spawn_agent"] = True; spawned += 1; self.diag["elder_spawn"] += 1
        self._record(a, m, act)
        return act, spawned

    def _forage(self, a, m, fruits, trees, sibs, edges, stats):
        p = self.p
        t = self.step * 0.1
        e, max_e, age = a["energy"], a["max_energy"], a["age"]
        if p["patience"] and fruits and e > p["ripe_min_frac"] * max_e and e > max_e / 5 + 10 and age >= p["newborn_age"]:
            keep = [f for f in fruits if f.get("_age") is None or f["_age"] >= p["ripe_s"]]
            if len(keep) < len(fruits):
                self.diag["ripe_skip"] += len(fruits) - len(keep)
                fruits = keep
        if p["elder"] and p["elder_yield"] and m.get("elder") and fruits and t >= p["elder_yield_t0"]:
            for s in sibs:
                if s["distance"] < p["yield_d"]:
                    se, sme, _sa = stats.get(s["id"], (max_e, max_e, 0.0))
                    if se < p["yield_frac"] * sme and not self.mem.get(s["id"], {}).get("elder"):
                        fruits = []; self.diag["elder_yield"] += 1
                        break
        if p["camp"] and t >= p["camp_t0"]:
            if fruits:
                m["last_fruit"] = self.step
            dn, cn = p["disperse_n"], p["crowd_n"]
            p["disperse_n"], p["crowd_n"] = 10 ** 6, 10 ** 6
            try:
                md, mdir, turn = super()._forage(a, m, fruits, trees, sibs, edges, stats)
            finally:
                p["disperse_n"], p["crowd_n"] = dn, cn
            if md == 0.0 and turn != 0.0 and len(trees) >= 2 and (self.step - m.get("last_fruit", self.step)) * 0.1 > p["camp_barren_s"]:
                st = sorted(trees, key=lambda o: o["distance"])
                md, mdir, turn = min(a["speed"], st[1]["distance"]), st[1]["angle"], 0.0
                m["last_fruit"] = self.step  # reset the barren clock
                self.diag["camp_leave"] += 1
            return md, mdir, turn
        return super()._forage(a, m, fruits, trees, sibs, edges, stats)


def make(**kw):
    class H(Hivemind):
        def __init__(self, params=None):
            q = dict(kw)
            if params:
                q.update(params)
            super().__init__(q)
    return H
