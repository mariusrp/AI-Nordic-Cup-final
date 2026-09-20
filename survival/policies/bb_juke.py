"""bb_juke (island A, evolve g4, APredatore-g4-1): bb_mpc with its radial sprint inside close_d replaced by
the apred_g3_1 walk-juke: move juke_off rad off the away vector toward the side the chaser's heading already
errs to (sign of its rel_dir w.r.t. me), walking; sprint only inside juke_sprint_d. Everything >= close_d (FDE
face / back-pedal / drain) is unchanged. juke=0 reproduces bb_mpc exactly.

BB2 (survival big bet): model-based predator handling on the champion forager.

Base forager = food_r1_1 spawning (fitness-gated, early reproduction) + food_r2_2 egocentric
need-based fruit claiming / dispersal. Predator handling is switchable (params["pred_mode"]):

  base : v1/food_r1_1 flee (face + back off inside 140 px, sprint inside 70 px).
  fde  : "face-drain-escape", built on the EXACT predator rule (upstream predator.py):
         a predator chases its CLOSEST perceived agent if that agent faces away or is < 90 px,
         otherwise it pivots at 45 deg at sprint (2.55 energy/step) and sleeps at 0 energy.
         Resting predators neither move nor kill. Each agent
           * tracks predators in its own odometry frame (own heading is exact: turns are applied
             verbatim by the sim; own displacement uses the reported biome move penalty) and
             flags a predator as RESTING when its heading and position are unchanged for 2 steps
             (an awake predator always turns except in a dead-straight charge);
           * predicts whether an awake predator will chase IT: the predator perceives it
             (hearing 60 or 250 px / +-30 deg cone, from rel_dir) and no visible sibling is a
             closer perceived agent, or it is inside 90 px;
           * only a predicted chaser triggers action: inside close_d flee at sprint while facing it,
             inside back_d face it and back-pedal at walking pace (pivot closes ~10.6 px/step, so
             the predator drains ~2.55/step until it sleeps), beyond that keep foraging but face it;
           * a resting predator is harmless until it wakes (+3/step, wakes above 100): agents keep
             sleep_margin px from it and otherwise forage;
           * awake threats that left view are remembered for mem_steps and faced (facing = no chase).
         fuse=1 adds instantaneous sibling-relative-pose fusion: when A sees sibling B, B's pose in
         A's frame is exact this step (distance, angle, rel_dir), so every predator B sees is
         re-expressed in A's frame with no dead reckoning.
Observation angles and move_direction are RELATIVE to heading."""
import math
import os
import statistics
import sys

TWO_PI = 2 * math.pi
PI6 = math.pi / 6

DEFAULT = dict(
    flee_dist=140.0, sprint_dist=70.0, face_pred=1.0, eat_energy_frac=0.85, tree_stay=35.0,
    spawn_energy=260.0, spawn_age=55.0, spawn_energy_old=140.0, max_herd=12, scan_turn=0.35,
    crowd_dist=40.0, crowd_n=3,
    # food_r1_1 breeding
    early_repro_age=50.0, small_herd_n=4, old_spare_age=55.0, old_spare_energy=180.0,
    w_energy=0.6, w_speed=0.25, w_hearing=0.15,
    # food_r2_2 claiming
    newborn_age=15.0, newborn_energy=120.0, claim_radius=25.0, disperse_dist=80.0, disperse_n=3,
    disperse_energy_frac=0.6,
    # BB2 predator model
    pred_mode="fde", fuse=0, sleep_escape=1,
    close_d=95.0, back_d=150.0, threat_d=260.0, cone_margin=0.10, hear_margin=8.0,
    sleep_margin=110.0, claim=1, lock_close_d=95.0, rest_steps=3, rest_steps_close=6, mem_steps=8, target_slack=0.0, other_close_d=120.0,
    # iteration 2 switches (0 = off = confirmed configuration)
    spawn_guard_d=0.0, spawn_guard_margin=10.0, face_all_d=0.0,
    # APredatore-g4-1 walk-juke inside close_d
    juke=1, juke_off=1.25, juke_sprint_d=40.0, juke_tie=0.05, juke_edge=45.0,
)

BIOME_PEN = {"swamp": 0.5, "desert": 0.8, "river": 0.3}


def wrap(a):
    return (a + math.pi) % TWO_PI - math.pi


def to_xy(d, a):
    return d * math.cos(a), d * math.sin(a)


def fitness(a, p):
    vr = a["vision_range"] / 200.0
    va = a["vision_angle"] / (math.pi / 3)
    return (vr * va) + p["w_energy"] * a["max_energy"] / 500.0 + p["w_speed"] * (a["speed"] + a["sprint_speed"]) / 30.0 \
        + p["w_hearing"] * a["hearing_radius"] / 50.0


def need_rank(energy, max_e, age, p):
    newborn = age < p["newborn_age"] or energy < p["newborn_energy"]
    return (0 if newborn else 1, energy / max(max_e, 1.0))


def perceives(px, py, phi, tx, ty, p):
    """Would a predator at (px,py) with heading phi (all in one frame) perceive a point (tx,ty)?"""
    dd = math.hypot(tx - px, ty - py)
    if dd <= 60 + p["hear_margin"]:
        return True
    if dd > 250 + p["hear_margin"]:
        return False
    return abs(wrap(math.atan2(ty - py, tx - px) - phi)) <= PI6 + p["cone_margin"]


class Hivemind:
    def __init__(self, params=None):
        self.p = dict(DEFAULT)
        if params:
            self.p.update(params)
        self.mem = {}
        self.step = 0
        self.diag = {"agent_steps": 0, "close": 0, "close_sprint": 0, "close_sprint_px": 0.0, "flip": 0, "blocked": 0, "tie": 0}
        self._diag_on = bool(os.environ.get("JUKE_DIAG"))

    # ------------------------------------------------------------------ tracking
    def _track(self, a, m):
        """Update this agent's predator tracks from its own observation. Returns list of
        current predator dicts (own frame) with a 'rest' count."""
        X, Y, H = m["X"], m["Y"], m["H"]
        cur = []
        for o in a["observations"]:
            if o["type"] != "Predator":
                continue
            d, th, r = o["distance"], o["angle"], o.get("rel_dir", 0.0)
            wx, wy = X + d * math.cos(H + th), Y + d * math.sin(H + th)
            psi = H + th + math.pi - r
            cur.append({"d": d, "th": th, "r": r, "wx": wx, "wy": wy, "psi": psi, "rest": 0, "own": True})
        old = m["tracks"]
        used = set()
        for c in cur:
            best, bd = None, 30.0
            for i, t in enumerate(old):
                if i in used:
                    continue
                dd = math.hypot(c["wx"] - t["wx"], c["wy"] - t["wy"])
                if dd < bd:
                    best, bd = i, dd
            if best is not None:
                used.add(best)
                t = old[best]
                if t["seen"] == self.step - 1 and abs(wrap(c["psi"] - t["psi"])) < 1e-6 and bd < 3.0:
                    c["rest"] = t["rest"] + 1
                c["threat"] = t.get("threat", False)
        tracks = []
        for c in cur:
            tracks.append({"wx": c["wx"], "wy": c["wy"], "psi": c["psi"], "rest": c["rest"], "seen": self.step,
                           "threat": c.get("threat", False)})
        for i, t in enumerate(old):
            if i not in used and self.step - t["seen"] <= self.p["mem_steps"]:
                tracks.append(t)
        m["tracks"] = tracks
        return cur

    def _odom(self, a, m, move_d, move_dir, turn):
        speed, sprint = a["speed"], a["sprint_speed"]
        d = max(0.0, min(move_d, sprint))
        if a["energy"] < a["max_energy"] / 5 and d > speed:
            d = speed
        d *= BIOME_PEN.get(a.get("biome"), 1.0)
        m["X"] += d * math.cos(m["H"] + move_dir)
        m["Y"] += d * math.sin(m["H"] + move_dir)
        m["H"] += turn

    # ------------------------------------------------------------------ main
    def act(self, agents, n_total=None):
        p = self.p
        n = len(agents)
        self.step += 1
        fits = {a["agent_id"]: fitness(a, p) for a in agents}
        med = statistics.median(fits.values()) if fits else 0.0
        stats = {a["agent_id"]: (a["energy"], a["max_energy"], a["age"]) for a in agents}
        alive = set()
        cur_by = {}
        fde = p["pred_mode"] == "fde"
        # pass 1: tracking
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
        for k in list(self.mem):
            if k not in alive:
                del self.mem[k]
        self.diag["agent_steps"] += n
        if self._diag_on and self.step % 500 == 0:
            print(f"JUKEDIAG {id(self)} step={self.step} n={n} " + " ".join(f"{k}={v:.0f}" for k, v in self.diag.items()),
                  file=sys.stderr, flush=True)
        return actions

    def _fused(self, a, cur_by):
        """Predators seen by visible siblings, expressed exactly in this agent's frame."""
        out = []
        own = [(c["d"] * math.cos(c["th"]), c["d"] * math.sin(c["th"])) for c in cur_by.get(a["agent_id"], [])]
        for s in a["observations"]:
            if s["type"] != "Agent" or "id" not in s:
                continue
            bx, by = to_xy(s["distance"], s["angle"])
            hb = s["angle"] + math.pi - s.get("rel_dir", 0.0)  # sibling heading in my frame
            for c in cur_by.get(s["id"], []):
                if not c.get("own"):
                    continue
                px = bx + c["d"] * math.cos(hb + c["th"])
                py = by + c["d"] * math.sin(hb + c["th"])
                if any(math.hypot(px - ox, py - oy) < 20 for ox, oy in own):
                    continue
                if any(math.hypot(px - f["x"], py - f["y"]) < 20 for f in out):
                    continue
                phi = hb + c["th"] + math.pi - c["r"]
                out.append({"x": px, "y": py, "phi": phi, "rest": c["rest"], "own": False})
        return out

    def _act_one(self, a, m, n, spawned, fits, med, stats, cur_by, fde):
        p = self.p
        aid = a["agent_id"]
        obs = a["observations"]
        energy = a["energy"]; max_e = a["max_energy"]; age = a["age"]
        speed = a["speed"]; sprint = a["sprint_speed"]
        preds = [o for o in obs if o["type"] == "Predator"]
        fruits = [o for o in obs if o["type"] == "Fruit"]
        trees = [o for o in obs if o["type"] == "Tree"]
        sibs = [o for o in obs if o["type"] == "Agent" and "id" in o]
        edges = [o for o in obs if o["type"] == "Edge"]
        move_d, move_dir, turn, spawn = 0.0, 0.0, 0.0, False

        # --- food_r1_1 spawning ---
        thr = p["spawn_energy"] if age < p["spawn_age"] else p["spawn_energy_old"]
        if age >= p["early_repro_age"]:
            thr = min(thr, p["spawn_energy_old"])
        can_afford = energy > max(thr, 101)
        may = fits.get(aid, 0.0) >= med or (n + spawned) <= p["small_herd_n"] or \
            (age > p["old_spare_age"] and energy > p["old_spare_energy"])
        if can_afford and may and n + spawned < p["max_herd"]:
            spawn = True; spawned += 1
        elif n + spawned <= 1 and energy > 101:
            spawn = True; spawned += 1

        # --- predator assessment ---
        danger = None       # (x, y, d) of the most urgent predicted chaser, own frame
        dangers = []
        sleepers = []
        watchers = []
        if fde:
            plist = []
            for c in cur_by.get(aid, []):
                x, y = to_xy(c["d"], c["th"])
                plist.append({"x": x, "y": y, "phi": c["th"] + math.pi - c["r"], "rest": c["rest"], "own": True})
            if p["fuse"]:
                plist += self._fused(a, cur_by)
            sib_xy = [to_xy(s["distance"], s["angle"]) for s in sibs]
            close_d = p["close_d"] if energy > max_e / 5 + 10 else max(p["close_d"], p["lock_close_d"])
            awake_near = 1e9
            for P in plist:
                d = math.hypot(P["x"], P["y"])
                if P["rest"] < p["rest_steps"]:
                    awake_near = min(awake_near, d)
                if d > p["threat_d"]:
                    continue
                if not P["own"] and p["fuse"] == 2:
                    # restricted fusion: only an unseen predator that perceives me and targets me
                    if P["rest"] >= p["rest_steps"] or not perceives(P["x"], P["y"], P["phi"], 0.0, 0.0, p):
                        continue
                    if any(math.hypot(sx - P["x"], sy - P["y"]) < d and perceives(P["x"], P["y"], P["phi"], sx, sy, p)
                           for sx, sy in sib_xy):
                        continue
                    dangers.append((P["x"], P["y"], d, P["phi"])); continue
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
                elif d < p["face_all_d"]:
                    watchers.append((P["x"], P["y"], d))
            # remembered threats not currently visible (own tracks only)
            if not dangers:
                X, Y, H = m["X"], m["Y"], m["H"]
                for t in m["tracks"]:
                    if t["seen"] < self.step and t.get("threat") and t["rest"] < p["rest_steps"]:
                        dx, dy = t["wx"] - X, t["wy"] - Y
                        ang = math.atan2(dy, dx) - H
                        dd = max(0.0, math.hypot(dx, dy) - 12.0 * (self.step - t["seen"]))
                        if dd < p["threat_d"]:
                            dangers.append((dd * math.cos(ang), dd * math.sin(ang), dd, t["psi"] - H))
            # mark threat flag on own tracks (for memory)
            for t in m["tracks"]:
                if t["seen"] == self.step:
                    X, Y, H = m["X"], m["Y"], m["H"]
                    lx, ly = t["wx"] - X, t["wy"] - Y
                    t["threat"] = any(math.hypot(lx * math.cos(-H) - ly * math.sin(-H) - dx_, lx * math.sin(-H) + ly * math.cos(-H) - dy_) < 1.0
                                      for dx_, dy_, _d, _f in dangers)
            if dangers:
                danger = min(dangers, key=lambda z: z[2])
            if spawn and p["spawn_guard_d"] > 0 and awake_near < p["spawn_guard_d"] \
                    and energy - 100 < max_e / 5 + p["spawn_guard_margin"]:
                spawn = False; spawned -= 1

        if not fde:
            near_pred = [o for o in preds if o["distance"] < p["flee_dist"]]
            if near_pred:
                vx = vy = 0.0
                for o in near_pred:
                    w = 1.0 / max(o["distance"], 1.0)
                    vx -= w * math.cos(o["angle"]); vy -= w * math.sin(o["angle"])
                flee = self._avoid_edges(math.atan2(vy, vx), edges)
                closest = min(near_pred, key=lambda o: o["distance"])
                move_d = sprint if (closest["distance"] < p["sprint_dist"] and energy > max_e / 5 + 10) else speed
                return self._pack(aid, move_d, flee, wrap(closest["angle"]), False), spawned
        elif danger is not None:
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
                if p["juke"]:
                    # walk-juke: the chaser turns <= ~0.3 rad/step, so moving juke_off off the away vector toward
                    # the side its heading already errs to makes it overshoot; sprint only when really close.
                    r = wrap(face + math.pi - phi)  # rel_dir of me w.r.t. the chaser's heading (+ = CCW of it)
                    flee = self._juke_dir(math.atan2(vy, vx), r, edges, aid)
                    md = sprint if (can_sprint and d < p["juke_sprint_d"]) else speed
                else:
                    flee = self._avoid_edges(math.atan2(vy, vx), edges)
                    md = sprint if can_sprint else speed
                if md > speed:
                    self.diag["close_sprint"] += 1; self.diag["close_sprint_px"] += md
                return self._pack(aid, md, flee, wrap(face), False), spawned
            if d < p["back_d"]:
                flee = self._avoid_edges(math.atan2(vy, vx), edges)
                return self._pack(aid, speed, flee, wrap(face), False), spawned
            # far chaser: keep foraging but face it; never step toward it
            fm_d, fm_dir, _t = self._forage(a, m, fruits, trees, sibs, edges, stats)
            if fm_d > 0 and math.cos(wrap(fm_dir - face)) > 0.3:
                fm_d = 0.0
            return self._pack(aid, fm_d, fm_dir, wrap(face), spawn), spawned

        md, mdir, turn = self._forage(a, m, fruits, trees, sibs, edges, stats)
        if fde and watchers:
            wx, wy, _wd = min(watchers, key=lambda z: z[2])
            face = math.atan2(wy, wx)
            if md > 0 and math.cos(wrap(mdir - face)) > 0.3:
                md = 0.0
            return self._pack(aid, md, mdir, wrap(face), spawn), spawned
        if fde and p["sleep_escape"] and sleepers:
            sx, sy, sd = min(sleepers, key=lambda z: z[2])
            if sd < p["sleep_margin"]:
                away = math.atan2(-sy, -sx)
                if md == 0.0 or math.cos(wrap(mdir - math.atan2(sy, sx))) > 0.0:
                    md, mdir = speed, self._avoid_edges(away, edges)
        return self._pack(aid, md, mdir, turn, spawn), spawned

    @staticmethod
    def _pack(aid, d, dr, t, s):
        return {"agent_id": aid, "move_distance": float(d), "move_direction": float(dr), "turn_angle": float(t),
                "spawn_agent": bool(s)}

    def _forage(self, a, m, fruits, trees, sibs, edges, stats):
        """food_r2_2 foraging (need-based claiming + dispersal), returns (move_d, move_dir, turn)."""
        p = self.p
        aid = a["agent_id"]; energy = a["energy"]; max_e = a["max_energy"]; age = a["age"]; speed = a["speed"]
        move_d, move_dir, turn = 0.0, 0.0, 0.0
        cluster = sum(1 for s in sibs if s["distance"] < p["crowd_dist"])
        hungry = energy < p["eat_energy_frac"] * max_e
        target = None
        deferred_all = False
        if fruits and hungry and not p["claim"]:
            target = min(fruits, key=lambda o: o["distance"])
            move_dir = target["angle"]; move_d = min(speed, target["distance"])
        elif fruits and hungry:
            own_rank = need_rank(energy, max_e, age, p)
            sib_xy = [(s, to_xy(s["distance"], s["angle"])) for s in sibs]
            usable = []
            for f in fruits:
                fx, fy = to_xy(f["distance"], f["angle"])
                radius = f["distance"] + p["claim_radius"]
                beaten = False
                for s, (sx, sy) in sib_xy:
                    if math.hypot(fx - sx, fy - sy) > radius:
                        continue
                    e, me, ag = stats.get(s["id"], (max_e, max_e, 999.0))
                    s_rank = need_rank(e, me, ag, p)
                    if s_rank < own_rank or (s_rank == own_rank and s["id"] < aid):
                        beaten = True; break
                if not beaten:
                    usable.append(f)
            if usable:
                target = min(usable, key=lambda o: o["distance"])
                move_dir = target["angle"]; move_d = min(speed, target["distance"])
            else:
                deferred_all = True
        if target is None and not deferred_all and trees:
            tree = min(trees, key=lambda o: o["distance"])
            if tree["distance"] > p["tree_stay"] + 10:
                move_dir = tree["angle"]; move_d = min(speed, tree["distance"] - p["tree_stay"])
            else:
                move_d = 0.0; turn = 0.08 * m["scan_dir"]
            if cluster > p["crowd_n"] and fruits == [] and m["wander"] <= 0:
                m["wander"] = 60
        elif target is None and deferred_all:
            disperse = (energy > p["disperse_energy_frac"] * max_e
                        and sum(1 for s in sibs if s["distance"] < p["disperse_dist"]) > p["disperse_n"])
            st = sorted(trees, key=lambda o: o["distance"]) if trees else []
            if disperse and len(st) >= 2:
                move_dir = st[1]["angle"]; move_d = min(speed, st[1]["distance"])
            elif st:
                tree = st[0]
                if tree["distance"] > p["tree_stay"] + 10:
                    move_dir = tree["angle"]; move_d = min(speed, tree["distance"] - p["tree_stay"])
                else:
                    move_d = 0.0; turn = 0.08 * m["scan_dir"]
            else:
                m["wander"] = max(m["wander"], 1)
        elif target is None and fruits:
            move_d = 0.0; turn = 0.08 * m["scan_dir"]
        elif target is None:
            m["wander"] = max(m["wander"], 1)
        if m["wander"] > 0 and target is None:
            m["wander"] -= 1
            if not trees and not fruits:
                if (m.get("t", 0) // 10) % 3 == 0:
                    turn = p["scan_turn"] * m["scan_dir"]; move_d = 0.0
                else:
                    move_d = speed; move_dir = self._avoid_edges(0.0, edges)
        close = [s for s in sibs if s["distance"] < p["crowd_dist"]]
        if close and move_d == 0.0:
            s = min(close, key=lambda o: o["distance"])
            move_dir = wrap(s["angle"] + math.pi); move_d = min(speed, 3.0)
        m["t"] = m.get("t", 0) + 1
        return move_d, move_dir, turn

    def _juke_dir(self, away, rd, edges, aid):
        """apred_g3_1: direction juke_off off `away` toward sign(rd); near-zero rd -> side with more edge room;
        flip if the chosen side runs into an edge, radial (edge-avoided) if both do."""
        off = self.p["juke_off"]
        room = {s: self._edge_room(wrap(away + s * off), edges) for s in (1, -1)}
        if abs(rd) > self.p["juke_tie"]:
            side = 1 if rd > 0 else -1
        else:
            self.diag["tie"] += 1
            side = max((1, -1), key=lambda s: (room[s], s * (1 if aid % 2 else -1)))
        if room[side] < self.p["juke_edge"]:
            if room[-side] >= self.p["juke_edge"]:
                side = -side
                self.diag["flip"] += 1
            else:
                self.diag["blocked"] += 1
                return self._avoid_edges(away, edges)
        return wrap(away + side * off)

    @staticmethod
    def _edge_room(direction, edges):
        """Distance along `direction` to the nearest edge segment (inf if none is hit)."""
        c, s = math.cos(direction), math.sin(direction)
        best = float("inf")
        for e in edges:
            (x1, y1), (x2, y2) = e["coords"]
            vx, vy = x2 - x1, y2 - y1
            det = -c * vy + s * vx
            if abs(det) < 1e-9:
                continue
            t = (-vy * x1 + vx * y1) / det
            u = (-s * x1 + c * y1) / det
            if t >= 0 and 0 <= u <= 1:
                best = min(best, t)
        return best

    @staticmethod
    def _avoid_edges(direction, edges):
        best = direction
        for e in edges:
            (x1, y1), (x2, y2) = e["coords"]
            dx, dy = x2 - x1, y2 - y1
            L = dx * dx + dy * dy
            if L == 0:
                continue
            t = max(0.0, min(1.0, -(x1 * dx + y1 * dy) / L))
            px, py = x1 + t * dx, y1 + t * dy
            if math.hypot(px, py) < 40:
                ang = math.atan2(py, px)
                if abs(wrap(best - ang)) < math.pi / 2:
                    best = wrap(ang + math.pi / 2 * (1 if wrap(best - ang) >= 0 else -1))
        return best


def make(**kw):
    class H(Hivemind):
        def __init__(self, params=None):
            q = dict(kw)
            if params:
                q.update(params)
            super().__init__(q)
    return H
