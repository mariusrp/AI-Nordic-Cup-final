"""scratch (survival, 19 Sep): FROM-SCRATCH MINIMAL HERD on top of bb_juke's predator machinery.

Thesis: the species lives only through continuous births; bb_juke keeps 12 mouths, they age in cohorts and the
newborns starve while 20-60 fruit still stand on the map. This policy keeps a SMALL, age-staggered herd that camps
fruiting trees, eats fruit ripe, walks only when there is something worth walking to, and times births to food.

Central demography (per tick, hivemind-wide):
  * N_max(t): a schedule of the number of non-elder agents allowed (rich early map -> 8, sparse late map -> 3).
  * N_min and a BRIDGE rule: at least `young_n` agents younger than `young_age` s at all times, so no cohort death
    (hidden max_age 60-120 s, then 0.1*age/s drain) can take the whole species.
  * Elders (aging drain detected from the exact energy residual) convert their stock into children at once
    (energy > 101), instead of burning it at 6-12/s.
  * Discretionary births only from a rich parent (energy >= rich_frac*max_e) standing at food (fruit visible),
    never while an awake predator is in contact, never below the sprint floor (max_e/5 + margin).
  * Parent choice among candidates: elders first, then trait score (speed, vision) + energy.
Foraging (per agent, own odometry frame):
  * fruit tracks with an age estimate (a fruit that appears inside the hearing circle of a still agent is new):
    camp at a tree, eat ripe (>= ripe_e) or unknown-age fruit, eat anything when energy < hungry_frac.
  * tree memory (own frame) + import of visible siblings' trees through the exact relative pose; leave a barren
    or dead tree for the best known tree, otherwise explore (walk 300 px, full scan, repeat) with edge avoidance.
  * stand still (1 energy/s) and scan (0.08 rad/tick) whenever nothing is worth walking to.
Predators: bb_juke's FDE + walk-juke, unchanged (tracking, resting detection, chaser prediction, memory).
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bb_juke  # noqa: E402
from bb_juke import wrap, to_xy, perceives  # noqa: E402

TWO_PI = 2 * math.pi

DEFAULT = dict(bb_juke.DEFAULT)
DEFAULT.update(
    # demography
    n_sched="0:8,500:8,800:6,1100:5,1500:4,2000:3", n_min=3, young_n=2, young_age=40.0,
    rich_frac=0.75, floor_margin=30.0, spawn_pred_d=200.0, elder_over=1, elder_spawn_e=101.0, elder_age=50.0,
    duo_spawn_e=180.0, at_food_d=150.0,
    # foraging
    hungry_frac=0.85, bank_frac=0.5, starve_frac=0.25, ripe_e=54.0, reach_hungry=260.0, reach_full=140.0, full_frac=0.95,
    camp_d=22.0, barren_s=35.0, tree_mem_s=150.0, fruit_mem_s=60.0, mature_s=20.0, explore_leg=30, scan_ticks=20,
    share_trees=1, disperse_d=55.0, disperse_e=220.0, w_vis=1.0, w_spd=1.0,
    elder_eat_sib_d=120.0,
)


def _sched(s):
    out = []
    for part in s.split(","):
        t, n = part.split(":")
        out.append((float(t), int(n)))
    return out


class Hivemind(bb_juke.Hivemind):
    def __init__(self, params=None):
        q = dict(DEFAULT)
        if params:
            q.update(params)
        super().__init__(q)
        self.sched = _sched(self.p["n_sched"])
        self.diag.update({"births": 0, "b_bridge": 0, "b_elder": 0, "b_rich": 0, "b_min": 0, "explore": 0, "camp": 0, "walk_fruit": 0, "wait": 0,
                          "camp_px": 0, "fruit_px": 0, "switch": 0, "phantom": 0, "phantom_imp": 0, "leave_barren": 0, "leave_crowd": 0})

    # ------------------------------------------------------------------ helpers
    def _n_max(self, t):
        n = self.sched[0][1]
        for t0, k in self.sched:
            if t >= t0:
                n = k
        return n

    def _to_world(self, m, d, ang):
        return m["X"] + d * math.cos(m["H"] + ang), m["Y"] + d * math.sin(m["H"] + ang)

    def _to_local(self, m, wx, wy):
        dx, dy = wx - m["X"], wy - m["Y"]
        return math.hypot(dx, dy), wrap(math.atan2(dy, dx) - m["H"])

    def _in_view(self, a, m, wx, wy, margin=0.9):
        d, ang = self._to_local(m, wx, wy)
        if d <= a["hearing_radius"] * margin:
            return True
        return d <= a["vision_range"] * margin and abs(ang) <= a["vision_angle"] / 2 - 0.06

    def _elder_check(self, a, m):
        """Exact residual: e_now - (e_prev - act_cost - 0.1). Aging drain shows as r ~ -0.01*age."""
        if m.get("elder"):
            return
        if "e_prev" not in m:
            return
        pred = m["e_prev"] - m["cost"] - 0.1
        r = a["energy"] - pred
        if a["age"] >= self.p["elder_age"] and -0.02 * a["age"] < r < -0.005 * a["age"]:
            m["elder_hits"] = m.get("elder_hits", 0) + 1
            if m["elder_hits"] >= 2:
                m["elder"] = True
        else:
            m["elder_hits"] = 0

    def _act_cost(self, a, act):
        speed, sprint = a["speed"], a["sprint_speed"]
        d = max(0.0, min(act["move_distance"], sprint))
        if a["energy"] < a["max_energy"] / 5 and d > speed:
            d = speed
        c = d * 0.05 if d <= speed else speed * 0.05 + (d - speed) * 0.5
        c += min(math.pi, abs(act["turn_angle"])) / TWO_PI
        return c

    def _update_memory(self, a, m, fruits, trees):
        """Tree + fruit memory in the agent's own odometry frame."""
        p = self.p
        st = self.step
        still = m.get("last_move", 1.0) < 0.5 and m.get("last_move2", 1.0) < 0.5
        hear = a["hearing_radius"]
        # trees
        seen = []
        for o in trees:
            wx, wy = self._to_world(m, o["distance"], o["angle"])
            best = None
            for t in m["trees"]:
                if math.hypot(t["x"] - wx, t["y"] - wy) < 25:
                    best = t; break
            if best is None:
                best = {"x": wx, "y": wy, "first": st, "fruit": -10**9, "nf": 0}
                m["trees"].append(best)
            else:
                best["x"] = 0.7 * best["x"] + 0.3 * wx; best["y"] = 0.7 * best["y"] + 0.3 * wy
            best["last"] = st
            seen.append(best)
        keep = []
        for t in m["trees"]:
            if t["last"] == st:
                keep.append(t); continue
            if st - t["last"] > p["tree_mem_s"] * 10:
                continue
            if self._in_view(a, m, t["x"], t["y"]):
                self.diag["phantom"] += 1
                if t.get("imp"):
                    self.diag["phantom_imp"] += 1
                continue  # in view but not observed: dead
            keep.append(t)
        m["trees"] = keep
        # fruits
        for o in fruits:
            wx, wy = self._to_world(m, o["distance"], o["angle"])
            best = None
            for f in m["fruits"]:
                if math.hypot(f["x"] - wx, f["y"] - wy) < 4.0:
                    best = f; break
            if best is None:
                new = still and o["distance"] < hear - 3.0 and m.get("here", 0) >= 2
                best = {"x": wx, "y": wy, "first": st, "new": new}
                m["fruits"].append(best)
            best["last"] = st; best["d"] = o["distance"]; best["ang"] = o["angle"]
            for t in m["trees"]:
                if math.hypot(t["x"] - wx, t["y"] - wy) < 75:
                    t["fruit"] = st; t["nf"] = t.get("nf", 0) + (1 if best["first"] == st else 0)
        keep = []
        for f in m["fruits"]:
            if f["last"] == st:
                keep.append(f); continue
            if st - f["last"] > p["fruit_mem_s"] * 10 or self._in_view(a, m, f["x"], f["y"]):
                continue
            keep.append(f)
        m["fruits"] = keep

    def _fruit_e(self, f):
        age = (self.step - f["first"]) * 0.1
        if f["new"]:
            return min(60.0, 20.0 + 2.0 * age)
        return min(60.0, 40.0 + 2.0 * age)  # unknown age: assume >= 10 s old when first seen

    def _share(self, a, m, sibs):
        """Import visible siblings' tree memories through the exact relative pose (this step)."""
        for s in sibs:
            sm = self.mem.get(s["id"])
            if sm is None or "H" not in sm:
                continue
            bx, by = self._to_world(m, s["distance"], s["angle"])
            hb = m["H"] + s["angle"] + math.pi - s.get("rel_dir", 0.0)  # sibling heading in my frame
            dh = hb - sm["H"]
            c, sn = math.cos(dh), math.sin(dh)
            for t in sm["trees"]:
                if self.step - t["last"] > 300:
                    continue
                rx, ry = t["x"] - sm["X"], t["y"] - sm["Y"]
                wx, wy = bx + c * rx - sn * ry, by + sn * rx + c * ry
                if any(math.hypot(u["x"] - wx, u["y"] - wy) < 30 for u in m["trees"]):
                    continue
                m["trees"].append({"x": wx, "y": wy, "first": t["first"], "last": t["last"], "fruit": t["fruit"], "nf": t.get("nf", 0), "imp": True})

    # ------------------------------------------------------------------ main
    def act(self, agents, n_total=None):
        p = self.p
        n = len(agents)
        self.step += 1
        t = self.step * 0.1
        alive = set()
        cur_by = {}
        for a in agents:
            aid = a["agent_id"]
            alive.add(aid)
            m = self.mem.get(aid)
            if m is None:
                m = self.mem[aid] = {"scan_dir": 1 if aid % 2 else -1, "X": 0.0, "Y": 0.0, "H": 0.0, "tracks": [],
                                     "trees": [], "fruits": [], "born": self.step, "here": 0, "ex": None, "ex_left": 0,
                                     "scan_left": 0, "camp": None, "camp_since": 0, "cost": 0.0}
            self._elder_check(a, m)
            cur_by[aid] = self._track(a, m)
        self._last_cur = cur_by
        # --- central demography ---
        elders = [a for a in agents if self.mem[a["agent_id"]].get("elder")]
        nonelder = n - len(elders)
        young = sum(1 for a in agents if a["age"] < p["young_age"])
        n_max = self._n_max(t)
        young_n = p["young_n"] if n_max >= 5 else 1
        need = max(0, p["n_min"] - nonelder, young_n - young)
        budget = n_max - nonelder
        cands = []
        for a in agents:
            aid = a["agent_id"]; m = self.mem[aid]
            e, me = a["energy"], a["max_energy"]
            if e <= max(p["elder_spawn_e"], 101.0):
                continue
            awake = min((c["d"] for c in cur_by[aid] if c["rest"] < p["rest_steps"]), default=1e9)
            if awake < p["spawn_pred_d"]:
                continue
            elder = bool(m.get("elder"))
            floor = me / 5 + p["floor_margin"]
            fruit_near = sum(1 for o in a["observations"] if o["type"] == "Fruit" and o["distance"] < p["at_food_d"])
            tree_near = any(o["type"] == "Tree" and o["distance"] < 80 for o in a["observations"])
            at_food = fruit_near > 0 or tree_near
            score = (p["w_spd"] * a["speed"] / 10.0 + p["w_vis"] * a["vision_range"] / 200.0) + e / me
            cands.append((0 if elder else 1, -score, aid, a, elder, e - 100 >= floor, at_food, fruit_near))
        cands.sort(key=lambda z: (z[0], z[1], z[2]))
        spawn_ids = set()
        for _o, _s, aid, a, elder, affordable, at_food, fruit_near in cands:
            e, me = a["energy"], a["max_energy"]
            if elder:
                if budget + p["elder_over"] > 0:
                    spawn_ids.add(aid); budget -= 1; need = max(0, need - 1); self.diag["b_elder"] += 1
                continue
            if not affordable and n > 2:
                continue
            if n <= 1 or (n == 2 and e > p["duo_spawn_e"]):
                spawn_ids.add(aid); budget -= 1; need = max(0, need - 1); self.diag["b_min"] += 1
                continue
            if need > 0 and (at_food or e >= p["rich_frac"] * me):
                spawn_ids.add(aid); budget -= 1; need -= 1; self.diag["b_bridge"] += 1
                continue
            if budget > 0 and e >= p["rich_frac"] * me and fruit_near > 0:
                spawn_ids.add(aid); budget -= 1; self.diag["b_rich"] += 1
        self.diag["births"] += len(spawn_ids)
        # --- per-agent actions ---
        actions = []
        for a in agents:
            aid = a["agent_id"]
            m = self.mem[aid]
            act = self._act_one(a, m, n, cur_by, aid in spawn_ids)
            m["cost"] = self._act_cost(a, act) + (0.0 if not act["spawn_agent"] or a["energy"] <= 100 else 100.0)
            m["e_prev"] = a["energy"]
            md = act["move_distance"]
            m["last_move2"] = m.get("last_move", 0.0); m["last_move"] = md
            m["here"] = m["here"] + 1 if md < 0.5 else 0
            self._odom(a, m, md, act["move_direction"], act["turn_angle"])
            actions.append(act)
        for k in list(self.mem):
            if k not in alive:
                del self.mem[k]
        self.diag["agent_steps"] += n
        if self._diag_on and self.step % 1000 == 0:
            print(f"SCRATCHDIAG step={self.step} n={n} nonelder={nonelder} young={young} nmax={n_max} "
                  + " ".join(f"{k}={v:.0f}" for k, v in self.diag.items()), file=sys.stderr, flush=True)
        return actions

    def _act_one(self, a, m, n, cur_by, spawn):
        """bb_juke's predator assessment (verbatim, fde/no-fuse) around the new forager."""
        p = self.p
        aid = a["agent_id"]
        obs = a["observations"]
        energy = a["energy"]; max_e = a["max_energy"]
        speed = a["speed"]; sprint = a["sprint_speed"]
        fruits = [o for o in obs if o["type"] == "Fruit"]
        trees = [o for o in obs if o["type"] == "Tree"]
        sibs = [o for o in obs if o["type"] == "Agent" and "id" in o]
        edges = [o for o in obs if o["type"] == "Edge"]
        self._update_memory(a, m, fruits, trees)
        if p["share_trees"] and sibs:
            self._share(a, m, sibs)

        danger = None
        dangers = []
        sleepers = []
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
                md = sprint if (can_sprint and d < p["juke_sprint_d"]) else speed
                if md > speed:
                    self.diag["close_sprint"] += 1; self.diag["close_sprint_px"] += md
                return self._pack(aid, md, flee, wrap(face), False)
            if d < p["back_d"]:
                flee = self._avoid_edges(math.atan2(vy, vx), edges)
                return self._pack(aid, speed, flee, wrap(face), False)
            fm_d, fm_dir, _t = self._forage(a, m, fruits, trees, sibs, edges, n)
            if fm_d > 0 and math.cos(wrap(fm_dir - face)) > 0.3:
                fm_d = 0.0
            return self._pack(aid, fm_d, fm_dir, wrap(face), spawn)

        md, mdir, turn = self._forage(a, m, fruits, trees, sibs, edges, n)
        if p["sleep_escape"] and sleepers:
            sx, sy, sd = min(sleepers, key=lambda z: z[2])
            if sd < p["sleep_margin"]:
                away = math.atan2(-sy, -sx)
                if md == 0.0 or math.cos(wrap(mdir - math.atan2(sy, sx))) > 0.0:
                    md, mdir = speed, self._avoid_edges(away, edges)
        return self._pack(aid, md, mdir, turn, spawn)

    # ------------------------------------------------------------------ forager
    def _forage(self, a, m, fruits, trees, sibs, edges, n):
        p = self.p
        aid = a["agent_id"]; energy = a["energy"]; max_e = a["max_energy"]; speed = a["speed"]
        st = self.step
        elder = bool(m.get("elder"))
        frac = energy / max_e
        hungry = frac < p["hungry_frac"]
        starving = frac < p["starve_frac"]
        scan = 0.08 * m["scan_dir"]
        # ---- 1. fruit: choose among tracked fruits (visible ones have fresh d/ang)
        best, best_v = None, -1e9
        reach = p["reach_hungry"] if hungry else p["reach_full"]
        if frac < p["full_frac"] and not (elder and any(s["distance"] < p["elder_eat_sib_d"] and not self.mem.get(s["id"], {}).get("elder") for s in sibs)):
            for f in m["fruits"]:
                if f["last"] != st:
                    continue
                d = f["d"]
                if d > reach:
                    continue
                fe = self._fruit_e(f)
                if f["new"] and fe < p["ripe_e"] and frac >= p["bank_frac"]:
                    continue  # a known-young fruit is a growing bank: leave it until ripe or needed
                if energy + fe > max_e + 15:
                    continue
                # sibling claim: a hungrier visible sibling closer to it takes it
                fx, fy = to_xy(d, f["ang"])
                beaten = False
                for s in sibs:
                    sx, sy = to_xy(s["distance"], s["angle"])
                    ds = math.hypot(fx - sx, fy - sy)
                    if ds < d - 3.0:
                        sm = self.mem.get(s["id"], {})
                        se = sm.get("e_prev", energy)
                        if se < energy or (abs(se - energy) < 1e-9 and s["id"] < aid):
                            beaten = True; break
                if beaten:
                    continue
                v = fe - 0.15 * d
                if v > best_v:
                    best, best_v = f, v
        if best is not None:
            self.diag["walk_fruit"] += 1; self.diag["fruit_px"] += min(speed, best["d"])
            return min(speed, best["d"]), best["ang"], 0.0
        # ---- 2. camp at a tree
        camp = None
        if m["trees"]:
            def tree_val(t):
                d, _ang = self._to_local(m, t["x"], t["y"])
                mature = (st - t["first"]) * 0.1 >= p["mature_s"] or t["fruit"] > -10**8
                recent_fruit = (st - t["fruit"]) * 0.1 < p["barren_s"]
                v = -0.05 * d + (30 if recent_fruit else 0) + (10 if mature else -5) + (5 if t.get("imp") else 0)
                if t is m.get("camp"):
                    v += 8
                return v, d
            cur = m.get("camp")
            if cur is not None and cur not in m["trees"]:
                m["camp"] = cur = None
            if cur is not None:
                camped_s = (st - m["camp_since"]) * 0.1
                bs = p["barren_s"] * (0.3 if starving else 1.0)
                barren = (st - cur["fruit"]) * 0.1 > bs and camped_s > bs
                crowd = sum(1 for s in sibs if s["distance"] < p["disperse_d"] and not self.mem.get(s["id"], {}).get("elder"))
                leave = barren or (crowd >= 1 and energy > p["disperse_e"] and not elder and
                                   all(energy >= self.mem.get(s["id"], {}).get("e_prev", 0) for s in sibs if s["distance"] < p["disperse_d"]))
                if leave:
                    self.diag["leave_barren" if barren else "leave_crowd"] += 1
                    alt = [t for t in m["trees"] if t is not cur and self._to_local(m, t["x"], t["y"])[0] > 60]
                    if alt:
                        cur = max(alt, key=lambda t: tree_val(t)[0])
                        m["camp"] = cur; m["camp_since"] = st; self.diag["switch"] += 1
                    elif barren and not crowd:
                        cur = None; m["camp"] = None
            if cur is None:
                cur = max(m["trees"], key=lambda t: tree_val(t)[0])
                m["camp"] = cur; m["camp_since"] = st
            camp = cur
        if camp is not None:
            d, ang = self._to_local(m, camp["x"], camp["y"])
            if d > p["camp_d"] + 8:
                self.diag["camp"] += 1; self.diag["camp_px"] += min(speed, d - p["camp_d"])
                return min(speed, d - p["camp_d"]), self._avoid_edges(ang, edges), 0.0
            self.diag["wait"] += 1
            m["ex"] = None
            return 0.0, 0.0, scan
        # ---- 3. explore: leg of explore_leg ticks walking, then a scan
        self.diag["explore"] += 1
        if m["ex"] is None or (m["ex_left"] <= 0 and m["scan_left"] <= 0):
            m["ex"] = m["H"] + (((aid * 7919 + st * 31) % 360) / 360.0 - 0.5) * 2.5 if m["ex"] is not None else m["H"]
            m["ex_left"] = p["explore_leg"]; m["scan_left"] = p["scan_ticks"]
        if m["ex_left"] > 0:
            m["ex_left"] -= 1
            rel = self._avoid_edges(wrap(m["ex"] - m["H"]), edges)
            if abs(wrap(rel - wrap(m["ex"] - m["H"]))) > 0.3:
                m["ex"] = m["H"] + rel
            return speed, rel, 0.0
        m["scan_left"] -= 1
        return 0.0, 0.0, TWO_PI / p["scan_ticks"] * m["scan_dir"]


def make(**kw):
    class H(Hivemind):
        def __init__(self, params=None):
            q = dict(kw)
            if params:
                q.update(params)
            super().__init__(q)
    return H
