"""lin_tail (surv3, 19 Sep 2026 night) = lineage6 + UPPER-TAIL mechanisms (all switchable via LIN_PARAMS):
  F1 ff_on: fast agents (speed >= ff_speed) flee a close predator by a straight back-pedal at full speed (facing it)
     instead of bb_juke's 1.25 rad juke (built for speed-10 agents; at speed 19 the juke gives only ~6 px/step of
     radial gain against a 15 px/step chase); ff_back caps the back-zone back-pedal of fast agents.
  F2 nsd_on: no birth while a predator is inside back_d (lineage6 lets elders spawn while fleeing -> the child
     is born 10-30 px from the predator; 51 newborn predator deaths per 48 games after t=900).
  F3 wm_on: WALL MEMORY. Edges are observed only inside the vision cone, so an agent that faces a predator and
     back-pedals is blind to the wall behind it - the kill autopsy shows fast agents pinned against the map edge or an
     obstacle, not out-run. Every observed edge is stored in the odometry frame (rounded key, kept wm_keep steps) and the
     flee/back-pedal direction becomes the direction closest to "away" with >= wm_room px of free path (remembered +
     visible edges); the juke of slow agents gets the same edge list.
  L1 l1_on: speed lottery - before lottery_t only elite agents breed on ANY path (elder / pre-senescence
     conversion included; lineage6 lets non-elite converters fill free slots with slow children), with a lower
     elite bar l1_bar and birth gap l1_gap. pure_on: after pure_t, once >= pure_nfast agents are fast, slow agents
     never breed (keeps the late herd predator-proof).
  C  srch_on: from srch_t, a non-elder with energy > srch_min_e never waits blind: with nothing in view it walks a
     straight search leg (persistent odometry heading, facing forward, srch_speed px/step); at a tree with no fruit
     within fz_d for barren_s s it walks to another visible tree or a leg away from it.
DEFAULTS = the best-measured arm (ff_on 1, nsd_on 1, srch_on 1, wm_on 0, l1_on 0, pure_on 0). NOTHING here is a
significant win; LIN_PARAMS='{"ff_on":0,"nsd_on":0,"srch_on":0}' reproduces lineage6 exactly (every mechanism's code
path is untouched when its switch is 0). Screens paired on seeds 7000-7063 against a lineage6 run of the same command
(pod = tdiag.py, one fresh process per seed; Mac = survival/evaluate.py):
  F1+F2            -93.6 se 40.7 (Mac, 64) and +8.6 se 34.2 (pod, 64) -> pooled ~-43 +- 27: a blind straight back-pedal
                   is worse than the juke, which at least slides along the wall.
  F3 wall memory   -31.5 se 39.8 (pod, 64), tail 2/64 vs 1/64 games >= 1800; acts on ~3% of close-predator steps.
  F1+F2+C search   +29.2 se 36.7 (pod, 64), tail 2/64 >= 1800, max 1888 (base 1885).
  F1+F2+L1 lottery +59.9 se 40.4 (pod, 64), 42/64 wins, p75 1472 vs 1426, but tail 1/64 >= 1800, max 1809  <- defaults.
DEFAULTS = ff_on 1, nsd_on 1, l1_on 1, pure_on 1 (the best-measured arm, z 1.5 = NOT significant). The tail (the metric
that matters for a top-3 validation) is unchanged by every arm; keep lineage6 live unless a 128-seed rerun of the
lottery arm confirms the mean.
Diagnosis (48 lineage6 games, fresh worker per seed): games >= 1690 s had a fast lineage early (first speed>=15.1
at ~218 s, 9/11 agents fast at 600 s) vs 2-4 fast at 600 s in shorter games; P(long | >= 8 fast at 600 s) = 4/13
vs 2/35 otherwise. Late (t >= 900) deaths: old 612, pred 382 (205 of fast agents), starve 377."""
import math
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lineage6  # noqa: E402
from lineage6 import LIN, act_cost, lin_fitness  # noqa: E402
from bb_juke import wrap, to_xy, perceives, TWO_PI  # noqa: E402

TAIL = dict(
    ff_on=1, ff_speed=15.5, ff_back=12.0,
    wm_on=0, wm_keep=600, wm_room=60.0,
    nsd_on=1,
    l1_on=1, l1_bar=150.0, l1_gap=3.0, pure_on=1, pure_t=600.0, pure_nfast=2,
    srch_on=0, srch_t=700.0, srch_min_e=25.0, srch_speed=20.0, fz_d=150.0, barren_s=20.0, leg_len=40, leg_min=8,
    leg_turn=0.6, leg_edge=60.0, leg_face=0.5,
)
if os.environ.get("LIN_PARAMS"):
    TAIL.update({k: v for k, v in json.loads(os.environ["LIN_PARAMS"]).items() if k in TAIL})


class Hivemind(lineage6.Hivemind):
    def __init__(self, params=None):
        q = dict(TAIL)
        if params:
            q.update(params)
        super().__init__(q)
        self.diag.update({"barred": 0, "ff": 0, "srch": 0, "barren": 0, "w_leg": 0.0, "wm_turn": 0})

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
            if p["wm_on"] and fde:
                self._remember_walls(a, m)
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
        n_fast = sum(1 for a in agents if a["speed"] >= p["fast_speed"])
        lot = p["l1_on"] and t < p["lottery_t"] and n > p["herd_emerg"]
        pure = p["pure_on"] and t >= p["pure_t"] and n_fast >= p["pure_nfast"] and n > p["herd_emerg"]
        def barred(a):
            if lot and a["agent_id"] not in elite:
                return True
            return pure and a["speed"] < p["fast_speed"]
        if p["l1_on"] and t < p["lottery_t"]:
            bar = p["l1_bar"]
        reserve = p["res0"] + p["res_slope"] * t
        youngest = min((a["age"] for a in agents), default=0.0)
        due = (n < tgt) or (self.step - self.last_birth_step) * 0.1 >= p["young_gap"] or youngest >= p["young_gap"]
        gap_ok = (self.step - self.last_birth_step) * 0.1 >= (p["l1_gap"] if (p["l1_on"] and t < p["lottery_t"]) else p["birth_gap"])
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
            if barred(a):
                self.diag["barred"] += 1; continue
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
            if barred(a):
                continue
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
            if aid not in elite or e <= max(mybar, 101.0) or (pure and a["speed"] < p["fast_speed"]):
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
            print(f"TAILDIAG step={self.step} n={n} tgt={tgt} " + " ".join(f"{k}={v:.0f}" for k, v in self.diag.items()),
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
            if p["wm_on"] and fde:
                edges = edges + self._wall_edges(m)
            vx = vy = 0.0
            for (x, y, dd, _f) in dangers:
                if dd < p["back_d"]:
                    w = 1.0 / max(dd, 1.0)
                    vx -= w * x / max(dd, 1e-6); vy -= w * y / max(dd, 1e-6)
            if d < close_d:
                can_sprint = energy > max_e / 5 + 10
                self.diag["close"] += 1
                r = wrap(face + math.pi - phi)
                if p["ff_on"] and speed >= p["ff_speed"]:
                    # fast lineage: straight back-pedal (facing the predator) out-walks its 15 px sprint
                    flee = self._avoid_edges(self._open_dir(math.atan2(vy, vx), edges, r), edges)
                    md = speed
                    self.diag["ff"] += 1
                else:
                    flee = self._juke_dir(math.atan2(vy, vx), r, edges, aid)
                    md = sprint if (can_sprint and d < p["juke_sprint_d"] and speed < p["fast_speed"]) else speed
                if md > speed:
                    self.diag["close_sprint"] += 1; self.diag["close_sprint_px"] += md
                self.diag["w_close"] += md
                # an elder converting its energy may still spawn while fleeing (the child is born behind it)
                return self._pack(aid, md, flee, wrap(face), spawn and permit == "elder" and not p["nsd_on"])
            if d < p["back_d"]:
                away = math.atan2(vy, vx)
                if p["wm_on"] and fde:
                    away = self._open_dir(away, edges, wrap(face + math.pi - phi))
                flee = self._avoid_edges(away, edges)
                if p["ff_on"] and speed >= p["ff_speed"]:
                    bspeed = min(speed, p["ff_back"])
                self.diag["w_back"] += bspeed
                return self._pack(aid, bspeed, flee, wrap(face), spawn and permit == "elder" and not p["nsd_on"])
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
        t_now = self.step * 0.1
        if any(f["distance"] < p["fz_d"] for f in fruits):
            m["fz"] = self.step
        m.setdefault("fz", self.step)
        if target is not None:
            m["wander"] = 0
            m["t"] = m.get("t", 0) + 1
            return move_d, move_dir, turn
        srch = p["srch_on"] and t_now >= p["srch_t"] and not elder and energy > p["srch_min_e"]
        if srch and m.get("leg", 0) > 0 and (m.get("leg_age", 0) < p["leg_min"] or not trees):
            return self._leg_step(a, m, edges)
        m["leg"] = 0
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
            elif srch and (self.step - m.get("fz", self.step)) * 0.1 > p["barren_s"] \
                    and (self.step - m.get("at_tree_since", self.step)) * 0.1 > p["barren_s"]:
                # barren tree: no fruit near me for barren_s s -> another visible tree, else a search leg away from it
                m["fz"] = self.step; m["at_tree_since"] = self.step
                others = [o for o in st[1:] if o["distance"] > p["tree_stay"] + 40]
                self.diag["barren"] += 1
                if others:
                    o = others[0]
                    m["leg"] = max(3, int((o["distance"] - p["tree_stay"]) / max(min(speed, p["srch_speed"]), 1.0)))
                    m["leg_age"] = 0; m["lh"] = wrap(m["H"] + o["angle"])
                else:
                    m["leg"] = p["leg_len"]; m["leg_age"] = 0; m["lh"] = wrap(m["H"] + tree["angle"] + math.pi)
                return self._leg_step(a, m, edges)
            else:
                move_d = 0.0; turn = p["scan_rate"] * m["scan_dir"]
                if "at_tree_since" not in m or m.get("at_tree_t", -9) < self.step - 1:
                    m["at_tree_since"] = self.step
                m["at_tree_t"] = self.step
            m["wander"] = 0
        elif fruits and deferred_all:
            move_d = 0.0; turn = p["scan_rate"] * m["scan_dir"]
        else:
            # nothing in view: wander only when rich (or very young and unfed), else stand, scan and wait
            rich = energy > max(p["wander_frac"] * max_e, p["wander_min"])
            if srch:
                # nothing in view in the lean late game: waiting is a death sentence -> straight search leg
                self.diag["srch"] += 1
                if m.get("leg", 0) <= 0:
                    prev = m.get("lh")
                    m["lh"] = wrap((prev if prev is not None else m["H"]) + p["leg_turn"] * (1 if (self.step // 97 + aid) % 2 else -1))
                    m["leg"] = p["leg_len"]; m["leg_age"] = 0
                return self._leg_step(a, m, edges)
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


    def _leg_step(self, a, m, edges):
        """One step of a straight search leg along world heading m['lh'] (odometry frame), facing along it."""
        p = self.p
        rel = wrap(m["lh"] - m["H"])
        if self._edge_room(rel, edges) < p["leg_edge"]:
            m["lh"] = wrap(m["lh"] + (math.pi / 2) * (1 if a["agent_id"] % 2 else -1))
            rel = wrap(m["lh"] - m["H"])
        md = min(a["speed"], p["srch_speed"])
        mdir = self._avoid_edges(rel, edges)
        turn = max(-p["leg_face"], min(p["leg_face"], rel))
        m["leg"] = m.get("leg", 0) - 1; m["leg_age"] = m.get("leg_age", 0) + 1
        self.diag["w_leg"] += md
        m["t"] = m.get("t", 0) + 1
        return md, mdir, turn


    def _remember_walls(self, a, m):
        """Wall memory in the odometry frame: edges are observed only inside the vision cone, so an agent that faces a
        predator and back-pedals is blind to the wall behind it (autopsy: fast agents die pinned at the map edge)."""
        walls = m.setdefault("walls", {})
        X, Y, H = m["X"], m["Y"], m["H"]
        c, s_ = math.cos(H), math.sin(H)
        for o in a["observations"]:
            if o["type"] != "Edge":
                continue
            (x1, y1), (x2, y2) = o["coords"]
            w = (X + x1 * c - y1 * s_, Y + x1 * s_ + y1 * c, X + x2 * c - y2 * s_, Y + x2 * s_ + y2 * c)
            walls[(round(w[0] / 10), round(w[1] / 10), round(w[2] / 10), round(w[3] / 10))] = w + (self.step,)
        if self.step % 50 == 0 and walls:
            old = self.step - self.p["wm_keep"]
            for k in [k for k, v in walls.items() if v[4] < old]:
                del walls[k]

    def _wall_edges(self, m):
        X, Y, H = m["X"], m["Y"], m["H"]
        c, s_ = math.cos(-H), math.sin(-H)
        out = []
        for (x1, y1, x2, y2, _st) in m.get("walls", {}).values():
            ax, ay, bx, by = x1 - X, y1 - Y, x2 - X, y2 - Y
            if min(abs(ax), abs(bx)) > 300 and min(abs(ay), abs(by)) > 300:
                continue
            out.append({"coords": ((ax * c - ay * s_, ax * s_ + ay * c), (bx * c - by * s_, bx * s_ + by * c))})
        return out

    def _open_dir(self, away, edges, rd):
        """The direction closest to `away` with >= wm_room px of free path (remembered + visible walls)."""
        room0 = self._edge_room(away, edges)
        if room0 >= self.p["wm_room"]:
            return away
        self.diag["wm_turn"] += 1
        sg = 1 if rd >= 0 else -1
        best, bestr = away, room0
        for off in (0.4, 0.8, 1.2, 1.6, 2.0):
            for s_ in (sg, -sg):
                dr = wrap(away + s_ * off)
                r = self._edge_room(dr, edges)
                if r >= self.p["wm_room"]:
                    return dr
                if r > bestr:
                    best, bestr = dr, r
        return best
