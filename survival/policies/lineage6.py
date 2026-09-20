"""lineage6 = lineage v4 with herd_div 6 (paired vs bb_juke on seeds 5000-5031: +190 se 55, z 3.45, 21/32 wins). lineage (Sat 19 Sep, LINEAGE ECONOMY lane): bb_juke's predator model + a demographic redesign.

Thesis: the game is a 3000 s demography problem. bb_juke (12-herd, everyone breeds) burns ~33% of its income on
walking, ~16% on elders past max_age (0.1*age/s), ~16% carried into predators, and dies at ~1100 s when the
fruit supply (halving every 300 s) no longer feeds 12 walkers.

Mechanisms (all params in LIN, override with LIN_PARAMS='{"k": v}' or Hivemind(params)):
  1. Elite lineage breeding: only carriers of the best traits breed (speed first - speed >= 15.1 out-walks a
     sprinting predator at no extra cost - then vision/hearing/max_energy). Phase 1 (t < lottery_t) runs a big
     herd to buy mutation lottery tickets.
  2. Herd target by time: target(t) = clamp(trees_est(t) / herd_div, herd_min, herd_max) with
     trees_est = 108 * 0.5^(t/600) (the simulator's tree equilibrium), so the herd shrinks with the food supply.
  3. Age diversity: at most one scheduled birth per birth_gap s (unless n <= herd_emerg), from the best-fed elite;
     a birth is also due whenever the youngest agent is older than young_gap s.
  4. Senescence conversion: an agent whose energy drops by 0.01*age/step beyond its action cost is past its hidden
     max_age (exact detection); it spawns at once while energy > elder_bar (over the cap by elder_over), ranks last
     for fruit, never walks far, and otherwise acts as a facing decoy.
  5. Energy reserve: a normal birth must leave the parent >= reserve(t) = res0 + res_slope * t and above the
     sprint lock (max_e/5 + lock_margin) unless the agent is fast (speed >= fast_speed).
  6. Minimal walking: walk to a fruit only within reach(energy); at a tree stand and scan; disperse once to a
     second visible tree instead of crowd-wandering; wander for a tree only with energy > wander_frac * max_e
     (otherwise stand, scan and wait for a tree to grow into view); foraging moves capped at walk_cap px/step.
Predator handling (FDE + juke) is bb_juke's, with the back-pedal capped at back_cap px/step."""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bb_juke  # noqa: E402
from bb_juke import wrap, to_xy, perceives, TWO_PI  # noqa: E402

LIN = dict(
    # breeding / lineage
    lottery_t=600.0, herd_max=12, herd_min=3, herd_div=6.0, herd_emerg=2, elite_margin=0.05, elite_min=2,
    tree0=100.0, tree_half=560.0,
    fast_speed=15.1, w_speed=3.0, w_speed2=0.3, w_vis=1.0, w_hear=0.6, w_maxe=0.0, w_cone=0.2, w_sprint=0.2,
    birth_gap=6.0, young_gap=40.0, spawn_bar=170.0, spawn_bar_late=230.0, fast_bar=150.0, res0=60.0, res_slope=0.02, lock_margin=10.0,
    pre_age=55.0, pre_bar=130.0,
    low_herd_any=3,   # if n < target - low_herd_any, the top-3 by fitness may breed even if not elite
    # senescence
    elder_from_age=55.0, elder_tol=0.25, elder_steps=2, elder_bar=101.0, elder_over=2, elder_reach=0.0, elder_yield_d=150.0,
    # economy
    eat_gap=25.0, reach_hi=1e9, reach_hi_d=90.0, reach_mid=1e9, reach_mid_d=160.0, walk_cap=10.0, back_cap=12.0,
    wander_frac=0.45, wander_min=140.0, disperse_sibs=2, disperse_d=80.0, disperse_min_e=150.0, tree_stay=35.0,
    scan_rate=0.08, nudge_d=12.0, nudge_px=2.0,
)
if os.environ.get("LIN_PARAMS"):
    import json
    LIN.update(json.loads(os.environ["LIN_PARAMS"]))


def act_cost(a, act):
    """Exact energy the simulator charges for this action (before the 0.1 base drain and aging)."""
    speed, sprint = a["speed"], a["sprint_speed"]
    d = max(0.0, min(act["move_distance"], sprint))
    if a["energy"] < a["max_energy"] / 5 and d > speed:
        d = speed
    c = d * 0.05 if d <= speed else speed * 0.05 + (d - speed) * 0.5
    c += min(math.pi, abs(act["turn_angle"])) / TWO_PI
    return c


def lin_fitness(a, p):
    sp = a["speed"]
    f = p["w_speed"] * min(sp, p["fast_speed"] + 0.4) / (p["fast_speed"] + 0.4) + p["w_speed2"] * sp / 20.0
    f += p["w_vis"] * min(a["vision_range"], 400.0) / 400.0 + p["w_hear"] * min(a["hearing_radius"], 100.0) / 100.0
    f += p["w_maxe"] * min(a["max_energy"], 1000.0) / 1000.0 + p["w_cone"] * min(a["vision_angle"], math.pi / 2) / (math.pi / 2)
    f += p["w_sprint"] * min(a["sprint_speed"], 25.0) / 25.0
    return f


class Hivemind(bb_juke.Hivemind):
    def __init__(self, params=None):
        q = dict(LIN)
        if params:
            q.update(params)
        super().__init__(q)
        self.last_birth_step = -10 ** 9
        self.diag.update({"births": 0, "elder_births": 0, "elder_flag": 0, "wander_steps": 0, "wait_steps": 0,
                          "reach_skip": 0, "res_block": 0, "disperse": 0, "sched_births": 0, "cap_births": 0,
                          "w_fruit": 0.0, "w_tree": 0.0, "w_disp": 0.0, "w_wander": 0.0, "w_nudge": 0.0, "w_back": 0.0,
                          "w_close": 0.0, "w_sleep": 0.0, "w_far": 0.0})

    # ------------------------------------------------------------------ herd level
    def target(self, t):
        p = self.p
        if t < p["lottery_t"]:
            return p["herd_max"]
        trees = p["tree0"] * 0.5 ** (t / p["tree_half"])
        return int(max(p["herd_min"], min(p["herd_max"], trees / p["herd_div"])))

    def _book(self, a, m):
        """Per-agent bookkeeping: elder detection from the exact energy residual."""
        p = self.p
        e, age = a["energy"], a["age"]
        m.setdefault("elder", False)
        m.setdefault("last_eat", -10 ** 9)
        if "e_prev" in m and not m.get("spawned_prev"):
            r = e - (m["e_prev"] - m["cost_prev"] - 0.1)
            if r > 0.5:
                m["last_eat"] = self.step
            elif not m["elder"] and age >= p["elder_from_age"]:
                if abs(r + 0.01 * age) < p["elder_tol"]:
                    m["elder_n"] = m.get("elder_n", 0) + 1
                    if m["elder_n"] >= p["elder_steps"]:
                        m["elder"] = True; self.diag["elder_flag"] += 1
                else:
                    m["elder_n"] = 0

    def act(self, agents, n_total=None):
        p = self.p
        n = len(agents)
        self.step += 1
        t = self.step * 0.1
        stats = {a["agent_id"]: (a["energy"], a["max_energy"], a["age"]) for a in agents}
        alive = set()
        cur_by = {}
        fde = p["pred_mode"] == "fde"
        for a in agents:
            aid = a["agent_id"]
            alive.add(aid)
            m = self.mem.setdefault(aid, {"scan_dir": 1 if aid % 2 else -1, "wander": 0, "X": 0.0, "Y": 0.0, "H": 0.0,
                                          "tracks": []})
            self._book(a, m)
            cur_by[aid] = self._track(a, m) if fde else []
        self._last_cur = cur_by
        # ---- breeding plan (herd level) ----
        fits = {a["agent_id"]: lin_fitness(a, p) for a in agents}
        tgt = self.target(t)
        top = max(fits.values()) if fits else 0.0
        ranked = sorted(agents, key=lambda a: -fits[a["agent_id"]])
        elite = {a["agent_id"] for a in agents if fits[a["agent_id"]] >= top - p["elite_margin"]}
        for a in ranked[:p["elite_min"]]:
            elite.add(a["agent_id"])
        if n < tgt - p["low_herd_any"]:
            for a in ranked[:3]:
                elite.add(a["agent_id"])
        bar = p["spawn_bar"] if t < p["lottery_t"] else p["spawn_bar_late"]
        reserve = p["res0"] + p["res_slope"] * t
        youngest = min((a["age"] for a in agents), default=0.0)
        due = (n < tgt) or (self.step - self.last_birth_step) * 0.1 >= p["young_gap"] or youngest >= p["young_gap"]
        gap_ok = (self.step - self.last_birth_step) * 0.1 >= p["birth_gap"]
        slots = 0
        if n <= p["herd_emerg"]:
            slots = max(1, tgt - n)
        elif due and gap_ok and n < tgt:
            slots = 1
        elif due and gap_ok and n < tgt + 1:
            slots = 1  # age-diversity birth may go one over the target
        permits = {}
        cands = []
        n_now = n
        elders = sorted((a for a in agents if self.mem[a["agent_id"]].get("elder")),
                        key=lambda a: (-(a["agent_id"] in elite), -a["energy"]))
        for a in elders:
            aid = a["agent_id"]
            if a["energy"] <= p["elder_bar"]:
                continue
            lim = tgt + p["elder_over"] if aid in elite else tgt
            if n_now < lim:
                permits[aid] = "elder"; n_now += 1
        # pre-senescence conversion: from pre_age on (max_age >= 60) an agent may die any time and its stock is lost,
        # so it converts energy above pre_bar into a child (elite: over the cap by elder_over; others into a free slot)
        pre = sorted((a for a in agents if not self.mem[a["agent_id"]].get("elder") and a["age"] >= p["pre_age"]
                      and a["energy"] > p["pre_bar"]), key=lambda a: (-(a["agent_id"] in elite), -a["energy"]))
        for a in pre:
            aid = a["agent_id"]
            lim = tgt + p["elder_over"] if aid in elite else tgt
            if n_now < lim and aid not in permits:
                permits[aid] = "elder"; n_now += 1
        for a in agents:
            aid = a["agent_id"]; m = self.mem[aid]
            e, max_e = a["energy"], a["max_energy"]
            if m.get("elder") or aid in permits:
                continue
            mybar = bar
            if a["speed"] >= p["fast_speed"]:
                mybar = min(bar, p["fast_bar"])
            if aid not in elite or e <= max(mybar, 101.0):
                continue
            left = e - 100.0
            if left < reserve:
                self.diag["res_block"] += 1; continue
            if a["speed"] < p["fast_speed"] and left < max_e / 5 + p["lock_margin"]:
                self.diag["res_block"] += 1; continue
            cands.append((fits[aid], e, aid))
        if n <= p["herd_emerg"] and not cands:
            for a in agents:
                if a["energy"] > 101.0 and a["agent_id"] not in permits:
                    cands.append((fits[a["agent_id"]], a["energy"], a["agent_id"]))
        cands.sort(reverse=True)
        n_el = sum(1 for v in permits.values() if v == "elder")
        for f, e, aid in cands[:max(0, slots - min(n_el, slots))]:
            permits[aid] = "sched"
        actions = []
        for a in agents:
            aid = a["agent_id"]
            m = self.mem[aid]
            act = self._act_one(a, m, n, permits.get(aid), stats, cur_by, fde)
            if act["spawn_agent"]:
                self.last_birth_step = self.step
                self.diag["births"] += 1
                self.diag["elder_births" if permits.get(aid) == "elder" else "sched_births"] += 1
            m["e_prev"] = a["energy"]; m["cost_prev"] = act_cost(a, act)
            m["spawned_prev"] = bool(act["spawn_agent"]) and a["energy"] > 100
            if fde:
                self._odom(a, m, act["move_distance"], act["move_direction"], act["turn_angle"])
            actions.append(act)
        for k in list(self.mem):
            if k not in alive:
                del self.mem[k]
        self.diag["agent_steps"] += n
        if self._diag_on and self.step % 500 == 0:
            print(f"LINDIAG step={self.step} n={n} tgt={tgt} " + " ".join(f"{k}={v:.0f}" for k, v in self.diag.items()),
                  file=sys.stderr, flush=True)
        return actions

    # ------------------------------------------------------------------ per agent
    def _act_one(self, a, m, n, permit, stats, cur_by, fde):
        p = self.p
        aid = a["agent_id"]
        obs = a["observations"]
        energy = a["energy"]; max_e = a["max_energy"]
        speed = a["speed"]; sprint = a["sprint_speed"]
        fruits = [o for o in obs if o["type"] == "Fruit"]
        trees = [o for o in obs if o["type"] == "Tree"]
        sibs = [o for o in obs if o["type"] == "Agent" and "id" in o]
        edges = [o for o in obs if o["type"] == "Edge"]
        spawn = permit is not None
        wspeed = min(speed, p["walk_cap"])
        bspeed = min(speed, p["back_cap"])

        # --- predator assessment (bb_juke FDE, verbatim) ---
        danger = None
        dangers = []
        sleepers = []
        if fde:
            plist = []
            for c in cur_by.get(aid, []):
                x, y = to_xy(c["d"], c["th"])
                plist.append({"x": x, "y": y, "phi": c["th"] + math.pi - c["r"], "rest": c["rest"], "own": True})
            sib_xy = [to_xy(s["distance"], s["angle"]) for s in sibs]
            close_d = p["close_d"] if energy > max_e / 5 + 10 else max(p["close_d"], p["lock_close_d"])
            for P in plist:
                d = math.hypot(P["x"], P["y"])
                if d > p["threat_d"]:
                    continue
                if P["rest"] >= (p["rest_steps_close"] if d < close_d else p["rest_steps"]):
                    sleepers.append((P["x"], P["y"], d))
                    continue
                if d < close_d:
                    dangers.append((P["x"], P["y"], d, P["phi"])); continue
                if not perceives(P["x"], P["y"], P["phi"], 0.0, 0.0, p):
                    continue
                target = True
                for sx, sy in sib_xy:
                    ds = math.hypot(sx - P["x"], sy - P["y"])
                    if ds + p["target_slack"] < d and perceives(P["x"], P["y"], P["phi"], sx, sy, p):
                        target = False; break
                if target or d < p["other_close_d"]:
                    dangers.append((P["x"], P["y"], d, P["phi"]))
            if not dangers:
                X, Y, H = m["X"], m["Y"], m["H"]
                for tr in m["tracks"]:
                    if tr["seen"] < self.step and tr.get("threat") and tr["rest"] < p["rest_steps"]:
                        dx, dy = tr["wx"] - X, tr["wy"] - Y
                        ang = math.atan2(dy, dx) - H
                        dd = max(0.0, math.hypot(dx, dy) - 12.0 * (self.step - tr["seen"]))
                        if dd < p["threat_d"]:
                            dangers.append((dd * math.cos(ang), dd * math.sin(ang), dd, tr["psi"] - H))
            for tr in m["tracks"]:
                if tr["seen"] == self.step:
                    X, Y, H = m["X"], m["Y"], m["H"]
                    lx, ly = tr["wx"] - X, tr["wy"] - Y
                    tr["threat"] = any(math.hypot(lx * math.cos(-H) - ly * math.sin(-H) - dx_, lx * math.sin(-H) + ly * math.cos(-H) - dy_) < 1.0
                                       for dx_, dy_, _d, _f in dangers)
            if dangers:
                danger = min(dangers, key=lambda z: z[2])

        if danger is not None:
            dx, dy, d, phi = danger
            face = math.atan2(dy, dx)
            vx = vy = 0.0
            for (x, y, dd, _f) in dangers:
                if dd < p["back_d"]:
                    w = 1.0 / max(dd, 1.0)
                    vx -= w * x / max(dd, 1e-6); vy -= w * y / max(dd, 1e-6)
            if d < close_d:
                can_sprint = energy > max_e / 5 + 10
                self.diag["close"] += 1
                r = wrap(face + math.pi - phi)
                flee = self._juke_dir(math.atan2(vy, vx), r, edges, aid)
                md = sprint if (can_sprint and d < p["juke_sprint_d"] and speed < p["fast_speed"]) else speed
                if md > speed:
                    self.diag["close_sprint"] += 1; self.diag["close_sprint_px"] += md
                self.diag["w_close"] += md
                # an elder converting its energy may still spawn while fleeing (the child is born behind it)
                return self._pack(aid, md, flee, wrap(face), spawn and permit == "elder")
            if d < p["back_d"]:
                flee = self._avoid_edges(math.atan2(vy, vx), edges)
                self.diag["w_back"] += bspeed
                return self._pack(aid, bspeed, flee, wrap(face), spawn and permit == "elder")
            fm_d, fm_dir, _t = self._forage(a, m, fruits, trees, sibs, edges, stats)
            if fm_d > 0 and math.cos(wrap(fm_dir - face)) > 0.3:
                fm_d = 0.0
            return self._pack(aid, fm_d, fm_dir, wrap(face), spawn)

        md, mdir, turn = self._forage(a, m, fruits, trees, sibs, edges, stats)
        if fde and p["sleep_escape"] and sleepers:
            sx, sy, sd = min(sleepers, key=lambda z: z[2])
            if sd < p["sleep_margin"]:
                away = math.atan2(-sy, -sx)
                if md == 0.0 or math.cos(wrap(mdir - math.atan2(sy, sx))) > 0.0:
                    md, mdir = wspeed, self._avoid_edges(away, edges)
                    self.diag["w_sleep"] += md
        return self._pack(aid, md, mdir, turn, spawn)

    # ------------------------------------------------------------------ economy
    def _forage(self, a, m, fruits, trees, sibs, edges, stats):
        """Minimal-walking forager: need-based claiming (bb_juke) restricted to fruit within reach(energy);
        stand and scan at a tree; disperse once to a second tree when crowded; wander only when rich."""
        p = self.p
        aid = a["agent_id"]; energy = a["energy"]; max_e = a["max_energy"]; age = a["age"]; speed = a["speed"]
        wspeed = min(speed, p["walk_cap"])
        elder = m.get("elder", False)
        move_d, move_dir, turn = 0.0, 0.0, 0.0
        hungry = energy < max_e - p["eat_gap"]
        # reach: how far this agent will walk for a fruit
        hear = a["hearing_radius"]
        if elder:
            reach = max(hear, p["elder_reach"])
            # an elder eats only when no younger sibling is near enough to use the fruit instead
            if any(s["distance"] < p["elder_yield_d"] and not self.mem.get(s["id"], {}).get("elder") for s in sibs):
                reach = -1.0
        elif energy > p["reach_hi"]:
            reach = max(hear + 10.0, p["reach_hi_d"])
        elif energy > p["reach_mid"]:
            reach = p["reach_mid_d"]
        else:
            reach = 1e9
        target = None
        deferred_all = False
        if fruits and hungry:
            own_rank = (2 if elder else 0, energy / max(max_e, 1.0))
            if not elder and (age < p["newborn_age"] or energy < p["newborn_energy"]):
                own_rank = (0, energy / max(max_e, 1.0))
            elif not elder:
                own_rank = (1, energy / max(max_e, 1.0))
            sib_xy = [(s, to_xy(s["distance"], s["angle"])) for s in sibs]
            usable = []
            for f in fruits:
                if f["distance"] > reach:
                    self.diag["reach_skip"] += 1
                    continue
                fx, fy = to_xy(f["distance"], f["angle"])
                radius = f["distance"] + p["claim_radius"]
                beaten = False
                for s, (sx, sy) in sib_xy:
                    if math.hypot(fx - sx, fy - sy) > radius:
                        continue
                    e, me, ag = stats.get(s["id"], (max_e, max_e, 999.0))
                    sm = self.mem.get(s["id"], {})
                    if sm.get("elder"):
                        s_rank = (2, e / max(me, 1.0))
                    elif ag < p["newborn_age"] or e < p["newborn_energy"]:
                        s_rank = (0, e / max(me, 1.0))
                    else:
                        s_rank = (1, e / max(me, 1.0))
                    if s_rank < own_rank or (s_rank == own_rank and s["id"] < aid):
                        beaten = True; break
                if not beaten:
                    usable.append(f)
            if usable:
                target = min(usable, key=lambda o: o["distance"])
                move_dir = target["angle"]; move_d = min(speed, target["distance"])
                self.diag["w_fruit"] += move_d
            elif any(f["distance"] <= reach for f in fruits):
                deferred_all = True
        if target is not None:
            m["wander"] = 0
            m["t"] = m.get("t", 0) + 1
            return move_d, move_dir, turn
        # no fruit to take: hold a tree
        st = sorted(trees, key=lambda o: o["distance"]) if trees else []
        crowd = sum(1 for s in sibs if s["distance"] < p["disperse_d"])
        if st:
            tree = st[0]
            if len(st) >= 2 and crowd >= p["disperse_sibs"] and not elder and energy > p["disperse_min_e"] \
                    and tree["distance"] < p["tree_stay"] + 15 and m.get("disp_cool", 0) <= 0:
                # crowded at this tree: walk once to the second tree
                m["disp_cool"] = 300
                m["disp_target"] = self.step
                self.diag["disperse"] += 1
                move_dir = st[1]["angle"]; move_d = min(wspeed, st[1]["distance"] - p["tree_stay"])
                self.diag["w_disp"] += move_d
            elif tree["distance"] > p["tree_stay"] + 10 and not elder:
                move_dir = tree["angle"]; move_d = min(wspeed, tree["distance"] - p["tree_stay"])
                self.diag["w_tree"] += move_d
            else:
                move_d = 0.0; turn = p["scan_rate"] * m["scan_dir"]
            m["wander"] = 0
        elif fruits and deferred_all:
            move_d = 0.0; turn = p["scan_rate"] * m["scan_dir"]
        else:
            # nothing in view: wander only when rich (or very young and unfed), else stand, scan and wait
            rich = energy > max(p["wander_frac"] * max_e, p["wander_min"])
            if rich and not elder:
                if m["wander"] <= 0:
                    m["wander"] = 60
                m["wander"] -= 1
                self.diag["wander_steps"] += 1
                if (m.get("t", 0) // 10) % 3 == 0:
                    turn = p["scan_turn"] * m["scan_dir"]; move_d = 0.0
                else:
                    move_d = wspeed; move_dir = self._avoid_edges(0.0, edges)
                    self.diag["w_wander"] += move_d
            else:
                self.diag["wait_steps"] += 1
                move_d = 0.0; turn = p["scan_rate"] * m["scan_dir"]
        m["disp_cool"] = m.get("disp_cool", 0) - 1
        close = [s for s in sibs if s["distance"] < p["nudge_d"]]
        if close and move_d == 0.0:
            s = min(close, key=lambda o: o["distance"])
            move_dir = wrap(s["angle"] + math.pi); move_d = min(wspeed, p["nudge_px"])
            self.diag["w_nudge"] += move_d
        m["t"] = m.get("t", 0) + 1
        return move_d, move_dir, turn


def make(**kw):
    class H(Hivemind):
        def __init__(self, params=None):
            q = dict(kw)
            if params:
                q.update(params)
            super().__init__(q)
    return H
