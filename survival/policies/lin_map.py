"""lin_map = lineage6 + a SHARED DEAD-RECKONED WORLD MAP (survival round 3, 19-20 Sep).

Everything of lineage6 is kept; the map only adds a late-game routing branch to the forager.

Pose tracking (exact, per agent):
  * heading is exact (the simulator applies turn_angle verbatim);
  * displacement is predicted as min(d, sprint) (speed if sprint-locked) * biome penalty along H + move_direction
    (bb_juke._odom), then VERIFIED next step against static landmarks seen in both steps (trees, fruit, obstacle
    edge endpoints): for a static landmark T, (T - P_k) - (T - P_k+1) = actual displacement, to float precision.
    If the prediction does not fit, the simulator's collision rule (same distance, direction rotated in 10 deg
    steps alternating -/+, or no move) is searched pairwise over landmark pairs and the supported candidate wins.
  * frames: every founder/child starts in its own frame; when agent A sees agent B (distance, angle, rel_dir) B's
    pose in A's frame is exact, so B's whole frame (agents, predator tracks, map entries) is re-expressed in A's
    (children are merged on their first step, the parent hears them). Same-frame sightings correct the agent with
    the larger accumulated uncertainty (unc grows only on unverifiable moving steps).
Map: fruit (and trees) with first/last sighting; deleted when an agent that should perceive the spot (hearing
disc, or unoccluded vision cone) does not, when an agent eats next to it, or after fr_ttl s.
Routing: from route_t on, a hungry non-elder agent with no fruit in view walks (facing its path) to the best
remembered unclaimed fruit cluster (value - walk cost), one agent per cluster; predator handling overrides it."""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lineage6  # noqa: E402
from lineage6 import LIN  # noqa: E402
from bb_juke import wrap, BIOME_PEN  # noqa: E402

MAPP = dict(
    # defaults = screened arm B (strict, late): a route must be short, recent and worth >= 2 fruits
    map_on=1, route_on=1, route_t=800.0, route_e=0.5, route_min_e=25.0, route_max_d=400.0, route_wait=2.0,
    route_val=80.0, route_gain=70.0, route_clu=50.0, route_turn=0.4, route_timeout=1.5, route_tree=0,
    fr_seen=25.0, route_cool=20.0,
    disp_map=0, disp_t=600.0, disp_sibs=2, disp_min_e=120.0, disp_min_d=150.0, disp_hold=60.0,
    tree_wait=8.0, tree_amax=40.0, tree_seen=30.0, tree_min_d=80.0,
    fr_ttl=80.0, lm_tol=0.25, unc_rate=0.1, cell=100.0, del_every=2, route_rich=0,
)
if os.environ.get("MAP_PARAMS"):
    import json
    MAPP.update(json.loads(os.environ["MAP_PARAMS"]))


def _seg_hit(ax, ay, bx, by, cx, cy, dx, dy):
    """Do segments AB and CD intersect?"""
    def orient(px, py, qx, qy, rx, ry):
        return (qx - px) * (ry - py) - (qy - py) * (rx - px)
    o1 = orient(ax, ay, bx, by, cx, cy); o2 = orient(ax, ay, bx, by, dx, dy)
    o3 = orient(cx, cy, dx, dy, ax, ay); o4 = orient(cx, cy, dx, dy, bx, by)
    return (o1 > 0) != (o2 > 0) and (o3 > 0) != (o4 > 0)


class Hivemind(lineage6.Hivemind):
    def __init__(self, params=None):
        q = dict(MAPP)
        if params:
            q.update(params)
        super().__init__(q)
        self.next_frame = 0
        self.ents = {}      # id -> entry {k: 'F'|'T', F, x, y, t0, t1, unc}
        self.grid = {}      # (F, cx, cy) -> set(ids)
        self.next_ent = 0
        self.claims = {}    # entry id -> agent id
        self.diag.update({"mv_steps": 0, "ver_pred": 0, "ver_coll": 0, "unver": 0, "merges": 0, "sib_fix": 0,
                          "routes": 0, "route_arrive": 0, "route_abort": 0, "route_steps": 0, "route_px": 0.0,
                          "del_abs": 0, "del_eat": 0, "del_ttl": 0, "n_fruit": 0})

    # ------------------------------------------------------------------ map storage
    def _cell(self, F, x, y):
        c = self.p["cell"]
        return (F, int(x // c), int(y // c))

    def _add(self, k, F, x, y, t, unc):
        i = self.next_ent; self.next_ent += 1
        e = {"k": k, "F": F, "x": x, "y": y, "t0": t, "t1": t, "unc": unc}
        self.ents[i] = e
        self.grid.setdefault(self._cell(F, x, y), set()).add(i)
        return i

    def _remove(self, i):
        e = self.ents.pop(i, None)
        if e is None:
            return
        s = self.grid.get(self._cell(e["F"], e["x"], e["y"]))
        if s:
            s.discard(i)
        self.claims.pop(i, None)

    def _move_ent(self, i, x, y):
        e = self.ents[i]
        c0 = self._cell(e["F"], e["x"], e["y"]); c1 = self._cell(e["F"], x, y)
        e["x"], e["y"] = x, y
        if c0 != c1:
            self.grid.get(c0, set()).discard(i)
            self.grid.setdefault(c1, set()).add(i)

    def _near(self, F, x, y, r):
        c = self.p["cell"]
        out = []
        for cx in range(int((x - r) // c), int((x + r) // c) + 1):
            for cy in range(int((y - r) // c), int((y + r) // c) + 1):
                s = self.grid.get((F, cx, cy))
                if s:
                    out.extend(s)
        return out

    # ------------------------------------------------------------------ frames
    def _transform(self, Fsrc, Fdst, th, tx, ty):
        """Re-express every agent / entry of frame Fsrc in frame Fdst: p' = R(th) p + t, H' = H + th."""
        c, s = math.cos(th), math.sin(th)
        for aid, m in self.mem.items():
            if m.get("F") != Fsrc:
                continue
            X, Y = m["X"], m["Y"]
            m["X"], m["Y"] = c * X - s * Y + tx, s * X + c * Y + ty
            m["H"] += th
            for tr in m["tracks"]:
                wx, wy = tr["wx"], tr["wy"]
                tr["wx"], tr["wy"] = c * wx - s * wy + tx, s * wx + c * wy + ty
                tr["psi"] += th
            if m.get("lm_prev"):
                m["lm_prev"] = [(c * vx - s * vy, s * vx + c * vy) for vx, vy in m["lm_prev"]]
            if m.get("pred"):
                px, py, dd = m["pred"]
                m["pred"] = (c * px - s * py, s * px + c * py, dd)
            r = m.get("route")
            if r:
                r["x"], r["y"] = c * r["x"] - s * r["y"] + tx, s * r["x"] + c * r["y"] + ty
            m["F"] = Fdst
        for i, e in self.ents.items():
            if e["F"] != Fsrc:
                continue
            self.grid.get(self._cell(Fsrc, e["x"], e["y"]), set()).discard(i)
            X, Y = e["x"], e["y"]
            e["x"], e["y"] = c * X - s * Y + tx, s * X + c * Y + ty
            e["F"] = Fdst
            self.grid.setdefault(self._cell(Fdst, e["x"], e["y"]), set()).add(i)

    # ------------------------------------------------------------------ per-step map update
    @staticmethod
    def _landmarks(obs, H):
        c, s = math.cos(H), math.sin(H)
        out = []
        for o in obs:
            ty = o["type"]
            if ty == "Tree" or ty == "Fruit":
                a = H + o["angle"]
                out.append((o["distance"] * math.cos(a), o["distance"] * math.sin(a)))
            elif ty == "Edge":
                for (rx, ry) in o["coords"]:
                    out.append((c * rx - s * ry, s * rx + c * ry))
        return out

    def _verify(self, m, cur):
        """Return the actual displacement correction (ddx, ddy) relative to the prediction already applied, and a
        status: 'pred' (prediction confirmed), 'coll' (collision candidate found), None (unverifiable)."""
        prev = m.get("lm_prev")
        pred = m.get("pred")
        if not pred or not prev or not cur:
            return 0.0, 0.0, None
        px, py, dd = pred
        tol = self.p["lm_tol"]
        # prev points hashed on a 2 px grid; a candidate displacement D is supported by w if w + D is a prev point
        hsh = {}
        for v in prev:
            hsh.setdefault((int(v[0] // 2), int(v[1] // 2)), []).append(v)

        def support(dx, dy, first=False):
            n = 0
            for w in cur:
                qx, qy = w[0] + dx, w[1] + dy
                kx, ky = int(qx // 2), int(qy // 2)
                hit = False
                for ix in (kx - 1, kx, kx + 1):
                    for iy in (ky - 1, ky, ky + 1):
                        for v in hsh.get((ix, iy), ()):
                            if abs(v[0] - qx) < tol and abs(v[1] - qy) < tol:
                                hit = True; break
                        if hit:
                            break
                    if hit:
                        break
                if hit:
                    n += 1
                    if first:
                        return 1
            return n
        # 1. the prediction (no collision)
        if support(px, py, True):
            return 0.0, 0.0, "pred"
        # 2. the simulator's collision rule: same distance, direction rotated -10, +10, -20, ... deg; else no move
        phi = math.atan2(py, px)
        best, bestn = None, 0
        step = math.pi / 18
        for i in range(1, 36):
            a = phi + step * ((i + 1) // 2) * (-1) ** i
            dx, dy = dd * math.cos(a), dd * math.sin(a)
            n = support(dx, dy)
            if n > bestn:
                best, bestn = (dx, dy), n
        n0 = support(0.0, 0.0)
        if n0 > bestn:
            best, bestn = (0.0, 0.0), n0
        if best is None:
            return 0.0, 0.0, None
        return best[0] - px, best[1] - py, "coll"

    def _map_update(self, agents):
        p = self.p
        t = (self.step + 1) * 0.1
        byid = {}
        # 1. new agents + pose verification
        for a in agents:
            aid = a["agent_id"]
            byid[aid] = a
            m = self.mem.get(aid)
            if m is None:
                m = {"scan_dir": 1 if aid % 2 else -1, "wander": 0, "X": 0.0, "Y": 0.0, "H": 0.0, "tracks": [],
                     "F": self.next_frame, "unc": 0.0}
                self.next_frame += 1
                self.mem[aid] = m
            cur = self._landmarks(a["observations"], m["H"])
            pred = m.get("pred")
            if pred and pred[2] > 0:
                self.diag["mv_steps"] += 1
                ddx, ddy, st = self._verify(m, cur)
                if st == "pred":
                    self.diag["ver_pred"] += 1
                elif st == "coll":
                    self.diag["ver_coll"] += 1
                    m["X"] += ddx; m["Y"] += ddy
                else:
                    self.diag["unver"] += 1
                    m["unc"] = m.get("unc", 0.0) + p["unc_rate"] * pred[2]
            m["lm_prev"] = cur
            m["pred"] = None
        # 2. sightings: frame merges and same-frame corrections
        for a in agents:
            A = self.mem[a["agent_id"]]
            for o in a["observations"]:
                if o["type"] != "Agent" or "id" not in o:
                    continue
                B = self.mem.get(o["id"])
                if B is None:
                    continue
                ang = A["H"] + o["angle"]
                bx = A["X"] + o["distance"] * math.cos(ang)
                by = A["Y"] + o["distance"] * math.sin(ang)
                if o["distance"] > 1e-9:
                    bh = ang + math.pi - o.get("rel_dir", 0.0)
                else:  # spawned on top of the parent: atan2(0, 0) = 0 in both angles, no pi
                    bh = ang - o.get("rel_dir", 0.0)
                if B["F"] != A["F"]:
                    th = bh - B["H"]
                    c, s = math.cos(th), math.sin(th)
                    tx = bx - (c * B["X"] - s * B["Y"])
                    ty = by - (s * B["X"] + c * B["Y"])
                    if B["F"] > A["F"]:
                        uncB = max(B.get("unc", 0.0), A.get("unc", 0.0))
                        Fs = B["F"]
                        self._transform(Fs, A["F"], th, tx, ty)
                        B["unc"] = uncB
                    else:
                        # inverse: express A's frame in B's frame
                        ci, si = math.cos(-th), math.sin(-th)
                        itx = -(ci * tx - si * ty); ity = -(si * tx + ci * ty)
                        self._transform(A["F"], B["F"], -th, itx, ity)
                        A["unc"] = max(A.get("unc", 0.0), B.get("unc", 0.0))
                    self.diag["merges"] += 1
                else:
                    if abs(wrap(bh - B["H"])) > 1e-3:
                        self.diag["hd_bad"] = self.diag.get("hd_bad", 0) + 1
                        continue
                    ex, ey = bx - B["X"], by - B["Y"]
                    if ex * ex + ey * ey > 4.0:
                        self.diag["sib_fix"] += 1
                        uA, uB = A.get("unc", 0.0), B.get("unc", 0.0)
                        if uB >= uA:
                            B["X"] += ex; B["Y"] += ey; B["unc"] = uA + 1.0
                            for tr in B["tracks"]:
                                tr["wx"] += ex; tr["wy"] += ey
                        else:
                            A["X"] -= ex; A["Y"] -= ey; A["unc"] = uB + 1.0
                            for tr in A["tracks"]:
                                tr["wx"] -= ex; tr["wy"] -= ey
        # 3. map entries
        de = p["del_every"]
        for a in agents:
            aid = a["agent_id"]; m = self.mem[aid]
            F, X, Y, H, unc = m["F"], m["X"], m["Y"], m["H"], m.get("unc", 0.0)
            seen = set()
            for o in a["observations"]:
                ty = o["type"]
                if ty != "Fruit" and ty != "Tree":
                    continue
                k = "F" if ty == "Fruit" else "T"
                ang = H + o["angle"]
                gx, gy = X + o["distance"] * math.cos(ang), Y + o["distance"] * math.sin(ang)
                r = (3.0 if k == "F" else 6.0) + min(unc, 10.0)
                best, bd = None, r * r
                for i in self._near(F, gx, gy, r):
                    e = self.ents[i]
                    if e["k"] != k:
                        continue
                    d2 = (e["x"] - gx) ** 2 + (e["y"] - gy) ** 2
                    if d2 <= bd:
                        best, bd = i, d2
                if best is None:
                    best = self._add(k, F, gx, gy, t, unc)
                else:
                    e = self.ents[best]
                    e["t1"] = t
                    if unc < e["unc"]:
                        self._move_ent(best, gx, gy); e["unc"] = unc
                seen.add(best)
            # ate this step -> the fruit next to me is gone
            if "e_prev" in m and not m.get("spawned_prev"):
                if a["energy"] - (m["e_prev"] - m["cost_prev"] - 0.1) > 0.5:
                    for i in self._near(F, X, Y, 20.0):
                        e = self.ents[i]
                        if e["k"] == "F" and i not in seen and (e["x"] - X) ** 2 + (e["y"] - Y) ** 2 < 400.0:
                            self._remove(i); self.diag["del_eat"] += 1
            # absence check: entries this agent must perceive but does not
            if (self.step + aid) % de == 0:
                hear = a["hearing_radius"]; vr = a["vision_range"]; half = a["vision_angle"] / 2.0
                edges = [o["coords"] for o in a["observations"] if o["type"] == "Edge"]
                c, s = math.cos(-H), math.sin(-H)
                for i in self._near(F, X, Y, vr):
                    if i in seen:
                        continue
                    e = self.ents.get(i)
                    if e is None:
                        continue
                    marg = 3.0 + min(e["unc"], 10.0) + min(unc, 10.0)
                    dx, dy = e["x"] - X, e["y"] - Y
                    d = math.hypot(dx, dy)
                    if d < hear - marg:
                        self._remove(i); self.diag["del_abs"] += 1
                        continue
                    if d <= hear + marg or d > vr - marg:
                        continue
                    lx, ly = c * dx - s * dy, s * dx + c * dy
                    if abs(math.atan2(ly, lx)) > half - marg / max(d, 1.0):
                        continue
                    if any(_seg_hit(0.0, 0.0, lx, ly, e0[0], e0[1], e1[0], e1[1]) for e0, e1 in edges):
                        continue
                    self._remove(i); self.diag["del_abs"] += 1
        # 4. expiry
        if self.step % 10 == 0:
            ttl = p["fr_ttl"]
            for i in [i for i, e in self.ents.items() if e["k"] == "F" and t - e["t0"] > ttl]:
                self._remove(i); self.diag["del_ttl"] += 1
            for i in [i for i, e in self.ents.items() if e["k"] == "T" and t - e["t1"] > 120.0]:
                self._remove(i)
            self.diag["n_fruit"] = sum(1 for e in self.ents.values() if e["k"] == "F")
        # claims of dead agents / expired tree holds
        holds = {m2.get("hold_id") for m2 in self.mem.values() if m2.get("hold_until", 0.0) > t}
        for i, aid in list(self.claims.items()):
            if aid not in byid:
                self.claims.pop(i, None)
            elif i not in self.ents:
                self.claims.pop(i, None)
        for m2 in self.mem.values():
            if m2.get("hold_until", 0.0) <= t and m2.get("hold_id") is not None:
                if self.claims.get(m2["hold_id"]) is not None and m2["hold_id"] not in holds:
                    self.claims.pop(m2["hold_id"], None)
                m2["hold_id"] = None

    # ------------------------------------------------------------------ hooks
    def act(self, agents, n_total=None):
        if self.p["map_on"]:
            self._map_update(agents)
        return super().act(agents, n_total)

    def _odom(self, a, m, move_d, move_dir, turn):
        speed, sprint = a["speed"], a["sprint_speed"]
        d = max(0.0, min(move_d, sprint))
        if a["energy"] < a["max_energy"] / 5 and d > speed:
            d = speed
        d *= BIOME_PEN.get(a.get("biome"), 1.0)
        ang = m["H"] + move_dir
        m["pred"] = (d * math.cos(ang), d * math.sin(ang), d)
        m["X"] += m["pred"][0]
        m["Y"] += m["pred"][1]
        m["H"] += turn

    def _release(self, aid, m):
        r = m.get("route")
        if r:
            for i in r.get("ids", ()):
                if self.claims.get(i) == aid:
                    self.claims.pop(i, None)
        m["route"] = None

    def _pick_route(self, a, m):
        p = self.p
        aid = a["agent_id"]
        F, X, Y = m["F"], m["X"], m["Y"]
        t = self.step * 0.1
        cands = [i for i in self._near(F, X, Y, p["route_max_d"]) if self.ents[i]["k"] == "F"
                 and self.claims.get(i, aid) == aid and t - self.ents[i]["t1"] <= p["fr_seen"]]
        if not cands:
            return None
        best, bsc = None, -1e9
        for i in cands:
            e = self.ents[i]
            D = math.hypot(e["x"] - X, e["y"] - Y)
            if D > p["route_max_d"] or D < 1.0:
                continue
            clu = [j for j in cands if (self.ents[j]["x"] - e["x"]) ** 2 + (self.ents[j]["y"] - e["y"]) ** 2
                   < p["route_clu"] ** 2]
            val = 0.0
            for j in clu:
                ej = self.ents[j]
                age = t - ej["t0"] + 10.0
                # value discounted by the chance the fruit is already gone (rotted or eaten since last seen)
                val += min(60.0, 20.0 + 2.0 * age) * max(0.2, 1.0 - (t - ej["t1"]) / p["fr_ttl"])
            gain = val - 0.05 * D - 0.01 * D / max(1.0, min(a["speed"], p["walk_cap"]))
            sc = gain / (D + 100.0)
            if gain >= p["route_gain"] and val >= p["route_val"] and sc > bsc:
                best, bsc = (i, clu, D, gain), sc
        if best is None:
            return None
        i, clu, D, gain = best
        e = self.ents[i]
        for j in clu:
            self.claims[j] = aid
        self.diag["routes"] += 1
        return {"x": e["x"], "y": e["y"], "ids": clu, "t0": t, "D0": D, "arr": 12.0}

    def _pick_tree(self, a, m, min_d=None):
        """Fallback when no fruit is remembered: the nearest remembered tree that is probably alive and fruiting
        (first seen <= tree_amax s ago, seen alive <= tree_seen s ago), not claimed, not the one I stand at."""
        p = self.p
        aid = a["agent_id"]
        F, X, Y = m["F"], m["X"], m["Y"]
        t = self.step * 0.1
        min_d = p["tree_min_d"] if min_d is None else min_d
        best, bd = None, p["route_max_d"]
        for i in self._near(F, X, Y, p["route_max_d"]):
            e = self.ents[i]
            if e["k"] != "T" or self.claims.get(i, aid) != aid:
                continue
            if t - e["t0"] > p["tree_amax"] or t - e["t1"] > p["tree_seen"]:
                continue
            D = math.hypot(e["x"] - X, e["y"] - Y)
            if D < min_d:
                continue
            if D < bd:
                best, bd = i, D
        if best is None:
            return None
        e = self.ents[best]
        self.claims[best] = aid
        self.diag["troutes"] = self.diag.get("troutes", 0) + 1
        return {"x": e["x"], "y": e["y"], "ids": [best], "t0": t, "D0": bd, "arr": p["tree_stay"], "kind": "T"}

    def _forage(self, a, m, fruits, trees, sibs, edges, stats):
        p = self.p
        aid = a["agent_id"]
        t = self.step * 0.1
        if not (p["map_on"] and p["route_on"]) or t < p["route_t"] or m.get("elder"):
            if m.get("route"):
                self._release(aid, m)
            return super()._forage(a, m, fruits, trees, sibs, edges, stats)
        energy, max_e = a["energy"], a["max_energy"]
        if fruits:
            m["nofruit"] = 0
            if m.get("route"):
                self._release(aid, m); self.diag["route_arrive"] += 1
            return super()._forage(a, m, fruits, trees, sibs, edges, stats)
        m["nofruit"] = m.get("nofruit", 0) + 1
        r = m.get("route")
        hungry = energy < p["route_e"] * max_e and energy > p["route_min_e"]
        if r is None and hungry and m["nofruit"] * 0.1 >= p["route_wait"] and t >= m.get("route_cool", 0.0):
            r = m["route"] = self._pick_route(a, m)
            if r is None and p["route_tree"] and m["nofruit"] * 0.1 >= p["tree_wait"]:
                r = m["route"] = self._pick_tree(a, m)
        if r is None and p["disp_map"] and t >= p["disp_t"] and energy > p["disp_min_e"] \
                and t >= m.get("route_cool", 0.0) and not m.get("hold_until", 0.0) > t:
            if sum(1 for s in sibs if s["distance"] < p["disperse_d"]) >= p["disp_sibs"]:
                r = m["route"] = self._pick_tree(a, m, p["disp_min_d"])
                if r is not None:
                    self.diag["disp_routes"] = self.diag.get("disp_routes", 0) + 1
        if r is not None:
            dx, dy = r["x"] - m["X"], r["y"] - m["Y"]
            D = math.hypot(dx, dy)
            if D < r["arr"] and r.get("kind") == "T":
                # arrived at a remembered tree: hold the claim for a while so no sibling routes to it
                m["hold_until"] = t + p["disp_hold"]
                for i in r["ids"]:
                    m["hold_id"] = i
                self.diag["route_arrive"] += 1
                m["route"] = None
                return super()._forage(a, m, fruits, trees, sibs, edges, stats)
            if D < r["arr"] or t - r["t0"] > p["route_timeout"] * r["D0"] / max(1.0, min(a["speed"], p["walk_cap"])) * 0.1 + 5.0 \
                    or energy <= p["route_min_e"]:
                self.diag["route_abort"] += 1
                m["route_cool"] = t + p["route_cool"]
                for i in r["ids"]:
                    if i in self.ents and self.ents[i]["k"] == "F" and \
                            (self.ents[i]["x"] - m["X"]) ** 2 + (self.ents[i]["y"] - m["Y"]) ** 2 < 900:
                        self._remove(i)
                self._release(aid, m)
                return super()._forage(a, m, fruits, trees, sibs, edges, stats)
            rel = wrap(math.atan2(dy, dx) - m["H"])
            wspeed = min(a["speed"], p["walk_cap"])
            md = min(wspeed, D)
            turn = max(-p["route_turn"], min(p["route_turn"], rel))
            mdir = self._avoid_edges(rel, edges)
            self.diag["route_steps"] += 1; self.diag["route_px"] += md
            m["wander"] = 0
            m["t"] = m.get("t", 0) + 1
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
