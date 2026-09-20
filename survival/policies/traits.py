"""traits (island G, trait-breeding-oracle lane, Sat 19 Sep evening): bb_juke + a two-phase demography.

Facts (organisers' simulator, verified): score = seconds survived (fruit/kill terms are < 1%); the game ends when the
LAST agent dies. A predator sprints at 15, so speed >= 15.1 out-walks it (walking costs 0.05/px whatever the trait).
Traits mutate on spawn with p=0.1 each x U(0.5,1.5) (caps speed 20 / sprint 40 / max_e 1000 / hearing 100 / vision 400
/ cone pi/2). Trees decay ~108*sqrt(0.5^(t/300)) (54 at 600 s, 13 at 1800 s, 7 at 2400 s) and live ~58 s; a fruit is
worth 20 at birth and 60 after 20 s. ORACLE (this lane, bb_juke behaviour with every agent forced to speed 16 / vision
400 / cone pi/2 / hearing 100): games still end at 1050-1450 s, predator deaths -75% but the 12-herd starves
(median energy < 150 from 600 s). So: traits buy immunity, the late game needs a lean, patient, spread-out herd.

Phase A (t < t_b): bb_juke foraging/evasion; RATCHET breeding: only the top-fitness lineage breeds (fitness = speed
   first, +2 bonus at >= 15.1, small vision/cone/hearing/max_e terms), elites spawn from energy > spawn_e_a at any age,
   herd cap cap_a; lineages that regressed below the elite margin never breed.
Phase B (t >= t_b): herd cap cap_b (cap_c after t_c); only fast (>= 15.1, if any) top-fitness agents breed; births are
   staggered (>= gap_s since the last birth) except elder conversion (an agent whose energy drops by the hidden
   0.01*age/tick aging drain spawns at once while energy > 101, allowed elder_over above the cap) and the n <= 1 emergency;
   CAMP foraging: no dispersal/crowd wander, fruit whose birth was observed (appeared within hearing while standing
   still) is eaten only once ripe (>= ripe_s) unless energy < low_e, an idle agent sharing a tree with a needier sibling
   moves to another visible tree that has no sibling near it, a barren tree (no fruit for barren_s) is left for another.
Everything else is bb_juke (FDE + juke evasion, need-based claiming). Params: TRAITS_PARAMS='{"cap_b": 3}' or make(**kw)."""
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bb_juke  # noqa: E402

TWO_PI = 2 * math.pi
FAST = 15.1

T = dict(
    # phase A ratchet
    ratchet=1, elite_margin=0.25, elite_k=3, cap_a=14, spawn_e_a=180.0, min_speed=7.0,
    w_vis=0.6, w_cone=0.3, w_hear=0.3, w_maxe=0.1,
    # phase B
    t_b=600.0, t_b_max=900.0, fast_n=2, cap_taper_s=300.0, cap_b=4, t_c=1e9, cap_c=3, spawn_e_b=180.0, gap_s=20.0, elite_k_b=2,
    elder=1, elder_from_age=55.0, elder_tol=0.25, elder_steps=2, elder_over=1, elder_spawn_e=101.0,
    camp=1, ripe=1, ripe_s=18.0, low_e=60.0, ripe_match=8.0, ripe_hear_margin=5.0, ripe_track_mem=60, ripe_still=0.5,
    sep=1, sep_d=60.0, sep_far=100.0, barren_s=30.0, tgt_steps=120,
    # fast-agent predator economy (speed >= 15.1 out-walks a predator): face + back-pedal at back_walk px/step from
    # back_d_fast (the pivoting predator closes at 15*cos(pi/4) = 10.6 px/step, so 11 holds the range at 0.55/step
    # until it sleeps), never sprint, flee radially inside close_d (the juke's radial component 20*cos(1.25) < 15 would
    # let the chaser close); walk_cap caps phase-B foraging/wander moves (same cost per px, fewer px wandered)
    back_walk=11.0, back_d_fast=115.0, walk_cap=12.0,
    # births need local food: >= birth_fruit_n fruit visible or a tree within food_tree_d (newborns start with 75)
    birth_food=1, birth_fruit_n=1, food_tree_d=60.0, elder_gap_s=3.0,
    # sleeper escape for fast agents: leave the resting predator's perception sideways, with margins, committed esc_commit steps
    sleep_lateral=1, esc_cone_margin=0.3, esc_hear_margin=25.0, esc_commit=4,
)
if os.environ.get("TRAITS_PARAMS"):
    T.update(json.loads(os.environ["TRAITS_PARAMS"]))


def act_cost(a, act):
    """Exact energy the simulator charges for this action (before the 0.1 base drain and aging)."""
    speed, sprint = a["speed"], a["sprint_speed"]
    d = max(0.0, min(act["move_distance"], sprint))
    if a["energy"] < a["max_energy"] / 5 and d > speed:
        d = speed
    c = d * 0.05 if d <= speed else speed * 0.05 + (d - speed) * 0.5
    c += min(math.pi, abs(act["turn_angle"])) / TWO_PI
    return c


def t_fitness(a, p):
    sp = a["speed"]
    f = 3.0 * min(sp, 20.0) / 20.0 + (2.0 if sp >= FAST else 0.0)
    f += p["w_vis"] * min(a["vision_range"], 400.0) / 400.0 + p["w_cone"] * min(a["vision_angle"], math.pi / 2) / (math.pi / 2)
    f += p["w_hear"] * min(a["hearing_radius"], 100.0) / 100.0 + p["w_maxe"] * min(a["max_energy"], 1000.0) / 1000.0
    return f


class Hivemind(bb_juke.Hivemind):
    def __init__(self, params=None):
        q = dict(T)
        if params:
            q.update(params)
        super().__init__(q)
        self.p["small_herd_n"] = 0          # no free-for-all breeding in small herds (the n<=1 emergency stays)
        self.p["old_spare_energy"] = 1e9    # no 'old spare' births
        self.p["early_repro_age"] = 0.0     # thresholds are set per agent below
        self.last_birth = -10 ** 9
        self.phase = "A"
        self.t_phase_b = 0.0
        self._pesc = dict(self.p, cone_margin=self.p["esc_cone_margin"], hear_margin=self.p["esc_hear_margin"])
        self.diag["sleep_esc"] = 0
        self.diag.update({"spawns": 0, "elder_flag": 0, "elder_spawn": 0, "stagger_veto": 0, "ripe_skip": 0,
                          "ripe_new": 0, "sep_move": 0, "barren_leave": 0, "eat": 0, "eat_gain": 0.0})

    # ------------------------------------------------------------------ herd-level decisions
    def act(self, agents, n_total=None):
        p = self.p
        n = len(agents)
        self.step += 1
        t = self.step * 0.1
        fitv = {a["agent_id"]: t_fitness(a, p) for a in agents}
        speeds = {a["agent_id"]: a["speed"] for a in agents}
        n_fast = sum(1 for s in speeds.values() if s >= FAST)
        any_fast = n_fast > 0
        # phase B starts at t_b once >= fast_n fast agents exist (the ratchet keeps running otherwise), at t_b_max regardless
        if self.phase == "A" and (t >= p["t_b_max"] or (t >= p["t_b"] and n_fast >= p["fast_n"])):
            self.phase = "B"; self.t_phase_b = t
        # eligibility
        cand = [aid for aid in fitv if speeds[aid] >= p["min_speed"]]
        if self.phase == "B" and any_fast:
            cand = [aid for aid in cand if speeds[aid] >= FAST]
        if not cand:
            cand = list(fitv)
        ranked = sorted(cand, key=lambda i: -fitv[i])
        top = fitv[ranked[0]] if ranked else 0.0
        k = p["elite_k"] if self.phase == "A" else p["elite_k_b"]
        elig = set(ranked[:k]) | {i for i in ranked if fitv[i] >= top - p["elite_margin"]}
        if not p["ratchet"] and self.phase == "A":
            elig = set(fitv)
        fits = {aid: (1.0 if aid in elig else 0.0) for aid in fitv}
        med = 0.5
        # herd cap
        if self.phase == "A":
            p["max_herd"] = p["cap_a"]
        else:
            cap = p["cap_c"] if t >= p["t_c"] else p["cap_b"]
            if p["cap_taper_s"] > 0:  # taper the cap from cap_a down to cap over cap_taper_s so births never stop
                frac = min(1.0, (t - self.t_phase_b) / p["cap_taper_s"])
                cap = max(cap, int(round(p["cap_a"] - frac * (p["cap_a"] - cap))))
            p["max_herd"] = cap
        stats = {a["agent_id"]: (a["energy"], a["max_energy"], a["age"]) for a in agents}
        self._youngest = min((a["age"] for a in agents), default=0.0)
        alive = set()
        cur_by = {}
        fde = p["pred_mode"] == "fde"
        for a in agents:
            aid = a["agent_id"]
            alive.add(aid)
            m = self.mem.setdefault(aid, {"scan_dir": 1 if aid % 2 else -1, "wander": 0, "X": 0.0, "Y": 0.0, "H": 0.0,
                                          "tracks": []})
            cur_by[aid] = self._track(a, m) if fde else []
        self._last_cur = cur_by
        actions = []
        spawned = 0
        for a in agents:
            aid = a["agent_id"]
            m = self.mem[aid]
            act, spawned = self._act_one(a, m, n, spawned, fits, med, stats, cur_by, fde)
            if fde:
                self._odom(a, m, act["move_distance"], act["move_direction"], act["turn_angle"])
            actions.append(act)
        for k_ in list(self.mem):
            if k_ not in alive:
                del self.mem[k_]
        self.diag["agent_steps"] += n
        return actions

    # ------------------------------------------------------------------ per-agent bookkeeping
    def _book(self, a, m):
        p = self.p
        e, age = a["energy"], a["age"]
        m.setdefault("elder", False)
        m.setdefault("still", 0)
        if "e_prev" in m:
            pred = m["e_prev"] - m["cost_prev"] - 0.1
            r = e - pred
            if m.get("spawned_prev"):
                pass
            elif r > 0.5:
                self.diag["eat"] += 1; self.diag["eat_gain"] += r
                m["last_eat"] = self.step
            elif p["elder"] and not m["elder"] and age >= p["elder_from_age"]:
                if abs(r + 0.01 * age) < p["elder_tol"]:
                    m["elder_n"] = m.get("elder_n", 0) + 1
                    if m["elder_n"] >= p["elder_steps"]:
                        m["elder"] = True; self.diag["elder_flag"] += 1
                else:
                    m["elder_n"] = 0
        if p["ripe"] and self.phase == "B":
            self._fruit_tracks(a, m)

    def _fruit_tracks(self, a, m):
        """Give each visible fruit an '_age' (s) when its birth was observed: a fruit that appears within hearing while
        the agent has stood still >= 2 steps must have just spawned (hearing has no cone, fruit does not move)."""
        p = self.p
        X, Y, H = m["X"], m["Y"], m["H"]
        tr = m.setdefault("ftracks", [])
        still = m["still"] >= 2
        hear = a["hearing_radius"] - p["ripe_hear_margin"]
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
                tr.append({"wx": wx, "wy": wy, "first": first, "seen": self.step})
                used.add(len(tr) - 1)
                o["_age"] = 0.0 if new else None
                if new:
                    self.diag["ripe_new"] += 1
        m["ftracks"] = [t for t in tr if self.step - t["seen"] <= p["ripe_track_mem"]]

    def _record(self, a, m, act):
        m["e_prev"] = a["energy"]
        m["cost_prev"] = act_cost(a, act)
        m["spawned_prev"] = bool(act["spawn_agent"]) and a["energy"] > 100
        m["still"] = m["still"] + 1 if act["move_distance"] <= self.p["ripe_still"] else 0
        if act["spawn_agent"] and a["energy"] > 100:
            self.diag["spawns"] += 1
            self.last_birth = self.step

    # ------------------------------------------------------------------ per-agent action (bb_juke FDE, fuse off)
    def _act_one(self, a, m, n, spawned, fits, med, stats, cur_by, fde):
        p = self.p
        self._book(a, m)
        aid = a["agent_id"]
        obs = a["observations"]
        energy = a["energy"]; max_e = a["max_energy"]; age = a["age"]
        speed = a["speed"]; sprint = a["sprint_speed"]
        fast = speed >= FAST
        fruits = [o for o in obs if o["type"] == "Fruit"]
        trees = [o for o in obs if o["type"] == "Tree"]
        sibs = [o for o in obs if o["type"] == "Agent" and "id" in o]
        edges = [o for o in obs if o["type"] == "Edge"]
        move_d, move_dir, turn = 0.0, 0.0, 0.0

        # --- spawning ---
        spawn = False
        elig = fits.get(aid, 0.0) >= med
        thr = p["spawn_e_a"] if self.phase == "A" else p["spawn_e_b"]
        near_tree = bool(trees) and min(o["distance"] for o in trees) <= p["food_tree_d"]
        food_ok = self.phase == "A" or not p["birth_food"] or len(fruits) >= p["birth_fruit_n"] or near_tree
        gap_ok = (self.step - self.last_birth) * 0.1 >= p["gap_s"] and self._youngest >= p["gap_s"]
        if elig and energy > max(thr, 101) and n + spawned < p["max_herd"] and food_ok:
            if self.phase == "A" or n + spawned <= 2 or gap_ok:
                spawn = True
            else:
                self.diag["stagger_veto"] += 1
        elif p["elder"] and m["elder"] and elig and energy > p["elder_spawn_e"] and food_ok \
                and n + spawned < p["max_herd"] + p["elder_over"] and (self.step - self.last_birth) * 0.1 >= p["elder_gap_s"]:
            spawn = True; self.diag["elder_spawn"] += 1
        elif n + spawned <= 1 and energy > 101:
            spawn = True
        if spawn:
            spawned += 1

        # --- predator assessment (own tracks) ---
        danger = None
        dangers = []
        sleepers = []
        plist = []
        for c in cur_by.get(aid, []):
            x, y = bb_juke.to_xy(c["d"], c["th"])
            plist.append({"x": x, "y": y, "phi": c["th"] + math.pi - c["r"], "rest": c["rest"]})
        sib_xy = [bb_juke.to_xy(s["distance"], s["angle"]) for s in sibs]
        close_d = p["close_d"] if (fast or energy > max_e / 5 + 10) else max(p["close_d"], p["lock_close_d"])
        for P in plist:
            d = math.hypot(P["x"], P["y"])
            if d > p["threat_d"]:
                continue
            if P["rest"] >= (p["rest_steps_close"] if d < close_d else p["rest_steps"]):
                sleepers.append((P["x"], P["y"], d, P["phi"]))
                continue
            if d < close_d:
                dangers.append((P["x"], P["y"], d, P["phi"])); continue
            if not bb_juke.perceives(P["x"], P["y"], P["phi"], 0.0, 0.0, p):
                continue
            target = True
            for sx, sy in sib_xy:
                ds = math.hypot(sx - P["x"], sy - P["y"])
                if ds + p["target_slack"] < d and bb_juke.perceives(P["x"], P["y"], P["phi"], sx, sy, p):
                    target = False; break
            if target or d < p["other_close_d"]:
                dangers.append((P["x"], P["y"], d, P["phi"]))
        if not dangers:
            X, Y, H = m["X"], m["Y"], m["H"]
            for t in m["tracks"]:
                if t["seen"] < self.step and t.get("threat") and t["rest"] < p["rest_steps"]:
                    dx, dy = t["wx"] - X, t["wy"] - Y
                    ang = math.atan2(dy, dx) - H
                    dd = max(0.0, math.hypot(dx, dy) - 12.0 * (self.step - t["seen"]))
                    if dd < p["threat_d"]:
                        dangers.append((dd * math.cos(ang), dd * math.sin(ang), dd, t["psi"] - H))
        for t in m["tracks"]:
            if t["seen"] == self.step:
                X, Y, H = m["X"], m["Y"], m["H"]
                lx, ly = t["wx"] - X, t["wy"] - Y
                t["threat"] = any(math.hypot(lx * math.cos(-H) - ly * math.sin(-H) - dx_, lx * math.sin(-H) + ly * math.cos(-H) - dy_) < 1.0
                                  for dx_, dy_, _d, _f in dangers)
        if dangers:
            danger = min(dangers, key=lambda z: z[2])

        if danger is not None:
            dx, dy, d, phi = danger
            face = math.atan2(dy, dx)
            back_d = p["back_d_fast"] if fast else p["back_d"]
            vx = vy = 0.0
            for (x, y, dd, _f) in dangers:
                if dd < max(back_d, p["back_d"]):
                    w = 1.0 / max(dd, 1.0)
                    vx -= w * x / max(dd, 1e-6); vy -= w * y / max(dd, 1e-6)
            if d < close_d:
                self.diag["close"] += 1
                if spawn:
                    spawn = False; spawned -= 1
                if fast:
                    flee = self._avoid_edges(math.atan2(vy, vx), edges)
                    md = speed
                else:
                    can_sprint = energy > max_e / 5 + 10
                    r = bb_juke.wrap(face + math.pi - phi)
                    flee = self._juke_dir(math.atan2(vy, vx), r, edges, aid)
                    md = sprint if (can_sprint and d < p["juke_sprint_d"]) else speed
                    if md > speed:
                        self.diag["close_sprint"] += 1; self.diag["close_sprint_px"] += md
                act = self._pack(aid, md, flee, bb_juke.wrap(face), False)
                self._record(a, m, act)
                return act, spawned
            if d < back_d:
                if spawn:
                    spawn = False; spawned -= 1
                flee = self._avoid_edges(math.atan2(vy, vx), edges)
                md = min(speed, p["back_walk"]) if (fast and p["back_walk"] > 0) else speed
                act = self._pack(aid, md, flee, bb_juke.wrap(face), False)
                self._record(a, m, act)
                return act, spawned
            fm_d, fm_dir, _t = self._forage(a, m, fruits, trees, sibs, edges, stats)
            if fm_d > 0 and math.cos(bb_juke.wrap(fm_dir - face)) > 0.3:
                fm_d = 0.0
            act = self._pack(aid, fm_d, fm_dir, bb_juke.wrap(face), spawn)
            self._record(a, m, act)
            return act, spawned

        md, mdir, turn = self._forage(a, m, fruits, trees, sibs, edges, stats)
        if p["sleep_escape"] and sleepers:
            sx, sy, sd, sphi = min(sleepers, key=lambda z: z[2])
            if p["sleep_lateral"] and fast:
                # a sleeper wakes in <= 3.4 s with its heading unchanged: leave its perception (hearing 60 px, 250 px
                # cone +-30 deg) SIDEWAYS (bb_juke's radial retreat stays on its cone axis, so it re-engages on waking)
                pe = self._pesc
                ang_me = math.atan2(-sy, -sx)
                if m.get("esc_until", 0) > self.step or bb_juke.perceives(sx, sy, sphi, 0.0, 0.0, pe):
                    if m.get("esc_until", 0) <= self.step:
                        m["esc_until"] = self.step + p["esc_commit"]; self.diag["sleep_esc"] += 1
                    off = bb_juke.wrap(ang_me - sphi)
                    side = 1 if off >= 0 else -1
                    esc = ang_me if sd < 60 + pe["hear_margin"] else sphi + side * math.pi / 2
                    md, mdir, turn = speed, self._avoid_edges(bb_juke.wrap(esc), edges), 0.0
            elif sd < p["sleep_margin"]:
                away = math.atan2(-sy, -sx)
                if md == 0.0 or math.cos(bb_juke.wrap(mdir - math.atan2(sy, sx))) > 0.0:
                    md, mdir = min(speed, p["walk_cap"]) if self.phase == "B" else speed, self._avoid_edges(away, edges)
        act = self._pack(aid, md, mdir, turn, spawn)
        self._record(a, m, act)
        return act, spawned

    def _forage(self, a, m, fruits, trees, sibs, edges, stats):
        p = self.p
        if self.phase != "B" or not p["camp"]:
            return super()._forage(a, m, fruits, trees, sibs, edges, stats)
        e, max_e, aid = a["energy"], a["max_energy"], a["agent_id"]
        speed = a["speed"]
        if p["ripe"] and fruits and e >= p["low_e"]:
            keep = [f for f in fruits if f.get("_age") is None or f["_age"] >= p["ripe_s"]]
            if len(keep) < len(fruits):
                self.diag["ripe_skip"] += len(fruits) - len(keep)
                fruits = keep
        if fruits:
            m["last_fruit"] = self.step
        # committed move to another tree (separation / barren leave), in the odometry frame
        tgt = m.get("camp_tgt")
        if tgt is not None:
            X, Y, H = m["X"], m["Y"], m["H"]
            dx, dy = tgt[0] - X, tgt[1] - Y
            dd = math.hypot(dx, dy)
            if dd <= p["tree_stay"] + 10 or self.step - tgt[2] > p["tgt_steps"]:
                m["camp_tgt"] = None
            else:
                hungry_usable = bool(fruits) and e < p["eat_energy_frac"] * max_e
                if not hungry_usable:
                    ang = bb_juke.wrap(math.atan2(dy, dx) - H)
                    return min(speed, dd - p["tree_stay"]), self._avoid_edges(ang, edges), 0.0
        dn, cn = p["disperse_n"], p["crowd_n"]
        p["disperse_n"], p["crowd_n"] = 10 ** 6, 10 ** 6
        a_c = a
        if p["walk_cap"] > 0 and speed > p["walk_cap"]:
            a_c = dict(a); a_c["speed"] = p["walk_cap"]
        try:
            md, mdir, turn = super()._forage(a_c, m, fruits, trees, sibs, edges, stats)
        finally:
            p["disperse_n"], p["crowd_n"] = dn, cn
        idle_at_tree = md == 0.0 and turn != 0.0 and trees and min(o["distance"] for o in trees) <= p["tree_stay"] + 10
        if not idle_at_tree or len(trees) < 2:
            return md, mdir, turn
        sib_xy = [(s, bb_juke.to_xy(s["distance"], s["angle"])) for s in sibs]
        # separation: an idle agent sharing its tree with a needier (or lower-id) sibling moves to a free tree
        leave = False
        if p["sep"]:
            my = e / max(max_e, 1.0)
            for s, _xy in sib_xy:
                if s["distance"] < p["sep_d"]:
                    se, sme, _sa = stats.get(s["id"], (max_e, max_e, 0.0))
                    sr = se / max(sme, 1.0)
                    if sr < my or (sr == my and s["id"] < aid):
                        leave = True; break
        if not leave and (self.step - m.get("last_fruit", self.step)) * 0.1 > p["barren_s"]:
            leave = True; m["last_fruit"] = self.step
            self.diag["barren_leave"] += 1
        elif leave:
            self.diag["sep_move"] += 1
        if leave:
            st = sorted(trees, key=lambda o: o["distance"])
            for tr in st[1:]:
                tx, ty = bb_juke.to_xy(tr["distance"], tr["angle"])
                if all(math.hypot(tx - sx, ty - sy) >= p["sep_far"] for _s, (sx, sy) in sib_xy):
                    X, Y, H = m["X"], m["Y"], m["H"]
                    m["camp_tgt"] = (X + tr["distance"] * math.cos(H + tr["angle"]), Y + tr["distance"] * math.sin(H + tr["angle"]), self.step)
                    return min(speed, tr["distance"] - p["tree_stay"]), self._avoid_edges(tr["angle"], edges), 0.0
        return md, mdir, turn


def make(**kw):
    class H(Hivemind):
        def __init__(self, params=None):
            q = dict(kw)
            if params:
                q.update(params)
            super().__init__(q)
    return H
