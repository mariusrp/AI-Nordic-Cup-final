"""kiting (19 Sep, survival policy-designer lane): bb_juke + exact world localisation + hivemind map.

Predator handling, spawning and the juke are bb_juke's (unchanged code, imported). What is new:

LOCALISATION (exact, not approximate). Own heading is exact (turns are applied verbatim), own displacement is the
requested distance x biome penalty (bb_juke._odom), but the START pose is unknown. Edge observations carry the FULL
segment in local coordinates, and the four map walls (thickness 30) expose inner edges of length 1600 (y = 30 and
y = 1170, stored start->end = +x) and 1200 (x = 30 and x = 1570, stored +y). Seeing one of them gives the agent's
absolute (x, y, heading) exactly. A fixed agent that sees a sibling (distance, angle, rel_dir, id) fixes the
sibling's absolute pose exactly too, so a newborn is fixed by its parent on its first step and the whole herd shares
ONE world frame. Predator tracks are re-expressed when a frame jump happens.

MAP (hivemind, fed only by fixed agents): trees (last seen, misses), fruits (first/last seen -> ripeness), obstacle
edges (world segments), a 40 px biome grid (from each agent's reported biome), and a 'last seen' grid for
exploration.

USE OF THE MAP: (1) known obstacle edges within edge_inject_d are injected as pseudo Edge observations, so bb_juke's
edge-avoidance also works for edges behind the agent (it walks backwards a lot); (2) flee/back-pedal/juke directions
avoid stepping into river/swamp from dry ground (water_flee); (3) foraging: visible fruit (bb_juke claiming) ->
nearest unclaimed remembered fruit -> tree assignment (visible + remembered trees, at most tree_cap agents per tree,
barren trees abandoned after barren_s) -> exploration toward the least-recently-seen cells (river/desert cells
penalised) instead of bb_juke's blind wander. Unfixed agents behave exactly like bb_juke.
Params: KP below, override with KIT_PARAMS='{"tree_cap": 2}'."""
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bb_juke  # noqa: E402
from bb_juke import wrap, to_xy, need_rank  # noqa: E402

TWO_PI = 2 * math.pi
MAP_W, MAP_H, WALL = 1600.0, 1200.0, 30.0

KP = dict(
    map=1,
    cell=40.0,
    edge_inject_d=130.0,
    water_flee=1,
    fruit_map_d=320.0, fruit_miss=3, fruit_max_age=950,
    tree_cap=3, tree_map_d=700.0, tree_stale=400, tree_miss=3, barren_s=25.0, tree_hyst=120.0,
    explore_d=550.0, explore_age=400, explore_reeval=25, explore_reach=45.0,
    biome_cost={"river": 4.0, "swamp": 1.7, "desert": 1.2, "forest": 1.0, "grassland": 1.0},
    tree_prior={"forest": 1.0, "grassland": 1.15, "swamp": 1.1, "desert": 1.6, "river": 3.0},
    stuck_steps=40, stuck_gain=15.0,
    diag=0,
)
if os.environ.get("KIT_PARAMS"):
    KP.update(json.loads(os.environ["KIT_PARAMS"]))


class Hivemind(bb_juke.Hivemind):
    def __init__(self, params=None):
        q = dict(KP)
        if params:
            q.update(params)
        super().__init__(q)
        nx, ny = int(MAP_W // self.p["cell"]), int(MAP_H // self.p["cell"])
        self.nx, self.ny = nx, ny
        self.biome = [None] * (nx * ny)
        self.seen = [-10 ** 9] * (nx * ny)
        self.trees = []     # {"x","y","first","last","miss","barren","id"}
        self.fruits = []    # {"x","y","first","last","miss","id"}
        self.edges = {}     # key -> ((x1,y1),(x2,y2)) world
        # obstacle rectangles (xmin, xmax, ymin, ymax): the four walls are known a priori; interior obstacles are
        # reconstructed from seen edges as 30 px deep slabs on the far side (the obstacle side), deepened when a
        # perpendicular edge sharing a corner is known. Used to replicate the simulator's collision deflection.
        self.rects = [(0.0, MAP_W, 0.0, WALL), (0.0, MAP_W, MAP_H - WALL, MAP_H), (0.0, WALL, 0.0, MAP_H),
                      (MAP_W - WALL, MAP_W, 0.0, MAP_H)]
        self._rect_of = {}  # edge key -> index in rects
        self._nid = 0
        self._claims = set()
        self._load = {}
        self._cur = None
        self.diag.update({"fix_wall": 0, "fix_sib": 0, "fixed_steps": 0, "map_fruit": 0, "map_tree": 0, "explore": 0,
                          "vis_tree": 0, "water_deflect": 0, "stuck": 0, "inject": 0, "stale": 0, "river_route": 0, "route_dodge": 0})

    # ------------------------------------------------------------------ geometry helpers
    def _cell(self, x, y):
        c = self.p["cell"]
        i = int(x // c); j = int(y // c)
        if 0 <= i < self.nx and 0 <= j < self.ny:
            return i * self.ny + j
        return -1

    def _biome_at(self, x, y):
        k = self._cell(x, y)
        return self.biome[k] if k >= 0 else None

    @staticmethod
    def _reframe(m, X1, Y1, H1):
        """Move this agent's frame to the new pose; re-express its predator tracks."""
        X0, Y0, H0 = m["X"], m["Y"], m["H"]
        dH = H1 - H0
        c, s = math.cos(dH), math.sin(dH)
        for t in m["tracks"]:
            dx, dy = t["wx"] - X0, t["wy"] - Y0
            t["wx"] = X1 + dx * c - dy * s
            t["wy"] = Y1 + dx * s + dy * c
            t["psi"] = t["psi"] + dH
        m["X"], m["Y"], m["H"] = X1, Y1, H1

    def _wall_fix(self, a, m):
        for o in a["observations"]:
            if o["type"] != "Edge":
                continue
            (lx1, ly1), (lx2, ly2) = o["coords"]
            vx, vy = lx2 - lx1, ly2 - ly1
            L = math.hypot(vx, vy)
            if L < 1100.0:
                continue
            horiz = L > 1400.0
            Hn = wrap((0.0 if horiz else math.pi / 2) - math.atan2(vy, vx))
            c, s = math.cos(Hn), math.sin(Hn)
            rx1 = lx1 * c - ly1 * s
            ry1 = lx1 * s + ly1 * c
            if horiz:
                ax = -rx1
                ay = (WALL if ry1 < 0 else MAP_H - WALL) - ry1
            else:
                ay = -ry1
                ax = (WALL if rx1 < 0 else MAP_W - WALL) - rx1
            if not (0 < ax < MAP_W and 0 < ay < MAP_H):
                continue
            # keep heading continuous with the odometry frame (H is used modulo 2pi only)
            self._reframe(m, ax, ay, Hn)
            m["fix"] = True; m["fixq"] = 0; m["fix_src"] = ("wall", self.step, round(L))
            self.diag["fix_wall"] += 1
            return True
        return False

    # ------------------------------------------------------------------ map update
    def _mark_seen(self, a, m):
        X, Y, H = m["X"], m["Y"], m["H"]
        vr = 0.85 * a["vision_range"]; hc = 0.85 * a["vision_angle"] / 2; hr = a["hearing_radius"]
        c = self.p["cell"]
        r = int(vr // c) + 1
        i0, j0 = int(X // c), int(Y // c)
        for i in range(max(0, i0 - r), min(self.nx, i0 + r + 1)):
            cx = (i + 0.5) * c
            for j in range(max(0, j0 - r), min(self.ny, j0 + r + 1)):
                cy = (j + 0.5) * c
                dx, dy = cx - X, cy - Y
                d = math.hypot(dx, dy)
                if d < hr or (d < vr and abs(wrap(math.atan2(dy, dx) - H)) < hc):
                    self.seen[i * self.ny + j] = self.step

    def _covers(self, a, m, x, y, margin=0.8):
        dx, dy = x - m["X"], y - m["Y"]
        d = math.hypot(dx, dy)
        if d < a["hearing_radius"] - 4:
            return True
        return d < margin * a["vision_range"] and abs(wrap(math.atan2(dy, dx) - m["H"])) < a["vision_angle"] / 2 - 0.08

    def _map_update(self, a, m):
        p = self.p
        X, Y, H = m["X"], m["Y"], m["H"]
        k = self._cell(X, Y)
        if k >= 0 and a.get("biome"):
            self.biome[k] = a["biome"]
        self._mark_seen(a, m)
        seen_t, seen_f = [], []
        for o in a["observations"]:
            t = o["type"]
            if t == "Edge":
                (lx1, ly1), (lx2, ly2) = o["coords"]
                if math.hypot(lx2 - lx1, ly2 - ly1) > 1100:
                    continue
                c, s = math.cos(H), math.sin(H)
                w1 = (X + lx1 * c - ly1 * s, Y + lx1 * s + ly1 * c)
                w2 = (X + lx2 * c - ly2 * s, Y + lx2 * s + ly2 * c)
                key = (round(w1[0] / 3), round(w1[1] / 3), round(w2[0] / 3), round(w2[1] / 3))
                if key not in self.edges:
                    self.edges[key] = (w1, w2)
                    self._add_rect(key, w1, w2, X, Y)
            elif t == "Tree" or t == "Fruit":
                d, th = o["distance"], o["angle"]
                wx, wy = X + d * math.cos(H + th), Y + d * math.sin(H + th)
                (seen_t if t == "Tree" else seen_f).append((wx, wy))
        for lst, seen, rad in ((self.trees, seen_t, 12.0), (self.fruits, seen_f, 5.0)):
            matched = set()
            for wx, wy in seen:
                best, bd = None, rad
                for i, e in enumerate(lst):
                    dd = math.hypot(e["x"] - wx, e["y"] - wy)
                    if dd < bd:
                        best, bd = i, dd
                if best is None:
                    self._nid += 1
                    lst.append({"x": wx, "y": wy, "first": self.step, "last": self.step, "miss": 0, "barren": -1,
                                "id": self._nid})
                    matched.add(len(lst) - 1)
                else:
                    e = lst[best]; e["last"] = self.step; e["miss"] = 0
                    # refine position (average) only slightly
                    e["x"] = 0.8 * e["x"] + 0.2 * wx; e["y"] = 0.8 * e["y"] + 0.2 * wy
                    matched.add(best)
            for i, e in enumerate(lst):
                if i in matched:
                    continue
                if self._covers(a, m, e["x"], e["y"]):
                    e["miss"] += 1

    def _add_rect(self, key, w1, w2, X, Y):
        (x1, y1), (x2, y2) = w1, w2
        depth = 30.0
        if abs(y2 - y1) < abs(x2 - x1):   # horizontal edge
            ye = 0.5 * (y1 + y2)
            xa, xb = min(x1, x2), max(x1, x2)
            for (v1, v2) in self.edges.values():
                if abs(v2[0] - v1[0]) < abs(v2[1] - v1[1]) and (abs(v1[0] - xa) < 1.5 or abs(v1[0] - xb) < 1.5) \
                        and (abs(v1[1] - ye) < 1.5 or abs(v2[1] - ye) < 1.5):
                    depth = max(depth, abs(v2[1] - v1[1]))
            if ye > Y:
                r = (xa, xb, ye, ye + depth)
            else:
                r = (xa, xb, ye - depth, ye)
        else:
            xe = 0.5 * (x1 + x2)
            ya, yb = min(y1, y2), max(y1, y2)
            for (v1, v2) in self.edges.values():
                if abs(v2[1] - v1[1]) < abs(v2[0] - v1[0]) and (abs(v1[1] - ya) < 1.5 or abs(v1[1] - yb) < 1.5) \
                        and (abs(v1[0] - xe) < 1.5 or abs(v2[0] - xe) < 1.5):
                    depth = max(depth, abs(v2[0] - v1[0]))
            if xe > X:
                r = (xe, xe + depth, ya, yb)
            else:
                r = (xe - depth, xe, ya, yb)
        self._rect_of[key] = len(self.rects)
        self.rects.append(r)

    def _blocked(self, x, y, r=5.0):
        for (xa, xb, ya, yb) in self.rects:
            if xa - r < x < xb + r and ya - r < y < yb + r:
                return True
        return False

    def _sim_move(self, m, d, direction):
        """Replicate environment.update_entity_position's collision handling on our rectangle model.
        d = effective distance, direction = absolute. Returns (x, y) after the move."""
        X, Y = m["X"], m["Y"]
        nx_, ny_ = X + d * math.cos(direction), Y + d * math.sin(direction)
        if not self._blocked(nx_, ny_):
            return nx_, ny_
        step = math.pi / 18
        for i in range(36):
            ta = direction + step * ((i + 1) // 2) * (-1) ** i
            tx, ty = X + d * math.cos(ta), Y + d * math.sin(ta)
            if not self._blocked(tx, ty):
                return tx, ty
        return X, Y

    def _odom(self, a, m, move_d, move_dir, turn):
        if not m.get("fix"):
            return super()._odom(a, m, move_d, move_dir, turn)
        speed, sprint = a["speed"], a["sprint_speed"]
        d = max(0.0, min(move_d, sprint))
        if a["energy"] < a["max_energy"] / 5 and d > speed:
            d = speed
        d *= bb_juke.BIOME_PEN.get(a.get("biome"), 1.0)
        x, y = self._sim_move(m, d, m["H"] + move_dir)
        m["X"] = max(5.0, min(MAP_W - 5.0, x)); m["Y"] = max(5.0, min(MAP_H - 5.0, y))
        m["H"] += turn

    def _map_prune(self):
        p = self.p
        st = self.step
        self.trees = [e for e in self.trees if e["miss"] < p["tree_miss"] and st - e["last"] < p["tree_stale"]]
        self.fruits = [e for e in self.fruits if e["miss"] < p["fruit_miss"] and st - e["first"] < p["fruit_max_age"]]

    # ------------------------------------------------------------------ main
    def act(self, agents, n_total=None):
        p = self.p
        if not p["map"]:
            return super().act(agents, n_total)
        self.step += 1
        # pass 0: memories + own predator tracking (as bb_juke pass 1)
        fde = p["pred_mode"] == "fde"
        cur_by = {}
        alive = set()
        for a in agents:
            aid = a["agent_id"]
            alive.add(aid)
            m = self.mem.setdefault(aid, {"scan_dir": 1 if aid % 2 else -1, "wander": 0, "X": 0.0, "Y": 0.0, "H": 0.0,
                                          "tracks": [], "fix": False, "fixq": 10 ** 9})
            cur_by[aid] = self._track(a, m) if fde else []
            # the simulator skips the agent that follows one dying in its update loop (list mutation while
            # iterating): that agent's observation is one tick stale. Never localise or map from a stale list.
            m["stale"] = a["observations"] == m.get("prev_obs")
            m["prev_obs"] = list(a["observations"])
            if m["stale"]:
                self.diag["stale"] += 1
        self._last_cur = cur_by
        # pass 1: localisation. wall fixes first, then sibling transfers ordered by fix quality
        for a in agents:
            m = self.mem[a["agent_id"]]
            if m["stale"] or not self._wall_fix(a, m):
                if m["fix"]:
                    m["fixq"] += 1
        by_id = {a["agent_id"]: a for a in agents}
        changed = True; rounds = 0
        while changed and rounds < 4:
            changed = False; rounds += 1
            for a in sorted(agents, key=lambda z: self.mem[z["agent_id"]]["fixq"]):
                m = self.mem[a["agent_id"]]
                if not m["fix"]:
                    break
                if m["stale"]:
                    continue
                X, Y, H = m["X"], m["Y"], m["H"]
                for s in a["observations"]:
                    if s["type"] != "Agent" or "id" not in s or s["id"] not in by_id:
                        continue
                    ms = self.mem[s["id"]]
                    if ms["fixq"] <= m["fixq"] + 1:
                        continue
                    d, th = s["distance"], s["angle"]
                    bx, by = X + d * math.cos(H + th), Y + d * math.sin(H + th)
                    if d == 0.0:
                        bh = H + th - s.get("rel_dir", 0.0)
                    else:
                        bh = H + th + math.pi - s.get("rel_dir", 0.0)
                    # keep the sibling's H numerically close to its odometry H (mod 2pi)
                    bh = ms["H"] + wrap(bh - ms["H"])
                    self._reframe(ms, bx, by, bh)
                    ms["fix"] = True; ms["fixq"] = m["fixq"] + 1; ms["fix_src"] = ("sib", self.step, a["agent_id"], round(d, 1))
                    self.diag["fix_sib"] += 1
                    changed = True
        # pass 2: map update from fixed agents
        for a in agents:
            m = self.mem[a["agent_id"]]
            if m["fix"] and not m["stale"]:
                self.diag["fixed_steps"] += 1
                self._map_update(a, m)
        self._map_prune()
        self._claims = set()
        self._load = {}
        # pass 3: actions (bb_juke's per-agent logic with our hooks)
        import statistics
        fits = {a["agent_id"]: bb_juke.fitness(a, p) for a in agents}
        med = statistics.median(fits.values()) if fits else 0.0
        stats = {a["agent_id"]: (a["energy"], a["max_energy"], a["age"]) for a in agents}
        n = len(agents)
        actions = []
        spawned = 0
        for a in agents:
            aid = a["agent_id"]
            m = self.mem[aid]
            self._cur = (a, m)
            if m["fix"]:
                self._inject_edges(a, m)
            act, spawned = self._act_one(a, m, n, spawned, fits, med, stats, cur_by, fde)
            if fde:
                self._odom(a, m, act["move_distance"], act["move_direction"], act["turn_angle"])
            actions.append(act)
        self._cur = None
        for k in list(self.mem):
            if k not in alive:
                del self.mem[k]
        self.diag["agent_steps"] += n
        if p["diag"] and self.step % 1000 == 0:
            print(f"KITDIAG step={self.step} n={n} trees={len(self.trees)} fruits={len(self.fruits)} edges={len(self.edges)} "
                  + " ".join(f"{k}={v:.0f}" for k, v in self.diag.items()), file=sys.stderr, flush=True)
        return actions

    def _inject_edges(self, a, m):
        """Append known obstacle edges near the agent (in local coords) that are not already observed, deduped."""
        X, Y, H = m["X"], m["Y"], m["H"]
        obs = a["observations"]
        vis = []
        out = []
        for o in obs:
            if o["type"] == "Edge":
                (x1, y1), (x2, y2) = o["coords"]
                k = (round(x1), round(y1), round(x2), round(y2))
                if k not in vis:
                    vis.append(k); out.append(o)
            else:
                out.append(o)
        R = self.p["edge_inject_d"]
        c, s = math.cos(-H), math.sin(-H)
        for (w1, w2) in self.edges.values():
            # quick reject by bounding box
            if min(w1[0], w2[0]) - R > X or max(w1[0], w2[0]) + R < X or min(w1[1], w2[1]) - R > Y or max(w1[1], w2[1]) + R < Y:
                continue
            dx1, dy1, dx2, dy2 = w1[0] - X, w1[1] - Y, w2[0] - X, w2[1] - Y
            lx1, ly1 = dx1 * c - dy1 * s, dx1 * s + dy1 * c
            lx2, ly2 = dx2 * c - dy2 * s, dx2 * s + dy2 * c
            k = (round(lx1), round(ly1), round(lx2), round(ly2))
            if any(abs(k[0] - v[0]) <= 3 and abs(k[1] - v[1]) <= 3 and abs(k[2] - v[2]) <= 3 and abs(k[3] - v[3]) <= 3 for v in vis):
                continue
            out.append({"type": "Edge", "coords": ((lx1, ly1), (lx2, ly2)), "_map": True})
            self.diag["inject"] += 1
        a["observations"] = out

    # ------------------------------------------------------------------ terrain-aware direction choice
    def _dry(self, direction, dist=45.0, swamp=True):
        """True unless the point `dist` along `direction` (local) lies in river, or (swamp=True) in swamp while we
        are on dry land."""
        if not self._cur:
            return True
        a, m = self._cur
        if not m["fix"]:
            return True
        X, Y, H = m["X"], m["Y"], m["H"]
        b = self._biome_at(X + dist * math.cos(H + direction), Y + dist * math.sin(H + direction))
        if b is None:
            return True
        if b == "river" and a.get("biome") != "river":
            return False
        if swamp and b == "swamp" and a.get("biome") not in ("swamp", "river"):
            return False
        return True

    def _avoid_edges_route(self, direction, edges):
        """Routing variant: obstacle avoidance + never step into the river (swamp allowed, it has trees)."""
        base = bb_juke.Hivemind._avoid_edges(direction, edges)
        if self._dry(base, swamp=False):
            return base
        for off in (0.7, -0.7, 1.4, -1.4, 2.1, -2.1):
            cand = bb_juke.Hivemind._avoid_edges(wrap(base + off), edges)
            if self._dry(cand, swamp=False) and self._edge_room(cand, edges) > 30.0:
                self.diag["river_route"] += 1
                return cand
        return base

    def _avoid_edges(self, direction, edges):
        base = bb_juke.Hivemind._avoid_edges(direction, edges)
        if not self.p["water_flee"] or self._dry(base):
            return base
        for off in (0.6, -0.6, 1.2, -1.2, 1.8, -1.8, 2.4, -2.4, math.pi):
            cand = bb_juke.Hivemind._avoid_edges(wrap(base + off), edges)
            if self._dry(cand) and self._edge_room(cand, edges) > 30.0:
                self.diag["water_deflect"] += 1
                return cand
        return base

    def _juke_dir(self, away, rd, edges, aid):
        d = bb_juke.Hivemind._juke_dir(self, away, rd, edges, aid)
        if not self.p["water_flee"] or self._dry(d):
            return d
        off = self.p["juke_off"]
        for cand in (wrap(away - (d - away)), away, wrap(away + 0.5 * off), wrap(away - 0.5 * off)):
            if self._dry(cand) and self._edge_room(cand, edges) >= self.p["juke_edge"]:
                self.diag["water_deflect"] += 1
                return cand
        return d

    # ------------------------------------------------------------------ foraging with the map
    def _goto(self, m, x, y, speed, edges):
        dx, dy = x - m["X"], y - m["Y"]
        d = math.hypot(dx, dy)
        direction = wrap(math.atan2(dy, dx) - m["H"])
        direction = self._avoid_edges_route(direction, edges)
        md = min(speed, d)
        if md > 0.0:
            eff = md * bb_juke.BIOME_PEN.get(self._cur[0].get("biome"), 1.0) if self._cur else md
            X, Y, H = m["X"], m["Y"], m["H"]
            if self._blocked(X + eff * math.cos(H + direction), Y + eff * math.sin(H + direction), 6.0):
                for off in (0.35, -0.35, 0.7, -0.7, 1.05, -1.05, 1.4, -1.4, 1.75, -1.75, 2.1, -2.1):
                    cand = wrap(direction + off)
                    if not self._blocked(X + eff * math.cos(H + cand), Y + eff * math.sin(H + cand), 6.0) \
                            and self._dry(cand, swamp=False):
                        direction = cand
                        self.diag["route_dodge"] += 1
                        break
        return md, direction, d

    def _route_cost(self, X, Y, x, y):
        d = math.hypot(x - X, y - Y)
        bc = self.p["biome_cost"]
        mid = self._biome_at((X + x) / 2, (Y + y) / 2)
        end = self._biome_at(x, y)
        f = 0.5 * bc.get(mid, 1.0) + 0.5 * bc.get(end, 1.0)
        return d * f

    def _forage(self, a, m, fruits, trees, sibs, edges, stats):
        if not m.get("fix"):
            return super()._forage(a, m, fruits, trees, sibs, edges, stats)
        p = self.p
        aid = a["agent_id"]; energy = a["energy"]; max_e = a["max_energy"]; age = a["age"]; speed = a["speed"]
        X, Y, H = m["X"], m["Y"], m["H"]
        move_d, move_dir, turn = 0.0, 0.0, 0.0
        hungry = energy < p["eat_energy_frac"] * max_e
        target = None
        # 1. visible fruit with bb_juke's need-based claiming
        if fruits and hungry:
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
                m["t"] = m.get("t", 0) + 1
                return min(speed, target["distance"]), target["angle"], 0.0
        # 2. remembered fruit (not visible now, unclaimed this step)
        if hungry and self.fruits:
            vis_w = []
            for f in fruits:
                vis_w.append((X + f["distance"] * math.cos(H + f["angle"]), Y + f["distance"] * math.sin(H + f["angle"])))
            best, bc = None, p["fruit_map_d"]
            for e in self.fruits:
                if e["id"] in self._claims:
                    continue
                if any(math.hypot(e["x"] - vx, e["y"] - vy) < 8.0 for vx, vy in vis_w):
                    continue
                cost = self._route_cost(X, Y, e["x"], e["y"])
                if cost < bc:
                    best, bc = e, cost
            if best is not None:
                self._claims.add(best["id"])
                self.diag["map_fruit"] += 1
                md, mdir, _d = self._goto(m, best["x"], best["y"], speed, edges)
                m["t"] = m.get("t", 0) + 1
                return md, mdir, 0.0
        # 3. tree assignment: visible + remembered trees, capped occupancy, barren skip, hysteresis for the current tree
        cands = []
        for e in self.trees:
            if e["barren"] > self.step:
                continue
            cands.append(e)
        cur_id = m.get("tree_id")
        best, bc = None, p["tree_map_d"]
        for e in cands:
            if self._load.get(e["id"], 0) >= p["tree_cap"]:
                continue
            cost = self._route_cost(X, Y, e["x"], e["y"]) + 90.0 * self._load.get(e["id"], 0)
            if e["id"] == cur_id:
                cost -= p["tree_hyst"]
            if cost < bc:
                best, bc = e, cost
        if best is None and trees:
            # a visible tree not in the map yet (should be rare): go there
            tree = min(trees, key=lambda o: o["distance"])
            self.diag["vis_tree"] += 1
            if tree["distance"] > p["tree_stay"] + 10:
                m["t"] = m.get("t", 0) + 1
                return min(speed, tree["distance"] - p["tree_stay"]), tree["angle"], 0.0
            m["t"] = m.get("t", 0) + 1
            return 0.0, 0.0, 0.08 * m["scan_dir"]
        if best is not None:
            self._load[best["id"]] = self._load.get(best["id"], 0) + 1
            self.diag["map_tree"] += 1
            if best["id"] != cur_id:
                m["tree_id"] = best["id"]; m["tree_since"] = self.step; m["tree_fruit"] = self.step
            if fruits:
                m["tree_fruit"] = self.step
            md, mdir, d = self._goto(m, best["x"], best["y"], speed, edges)
            if d > p["tree_stay"] + 10:
                self._stuck_check(m, d)
                if m.get("stuck_until", 0) > self.step:
                    m["t"] = m.get("t", 0) + 1
                    return speed, m["stuck_dir"], 0.0
                m["t"] = m.get("t", 0) + 1
                return min(speed, d - p["tree_stay"]), mdir, 0.0
            m.pop("prog", None)
            # at the tree: scan; abandon a barren tree
            if self.step - m.get("tree_since", self.step) > p["barren_s"] * 10 and self.step - m.get("tree_fruit", self.step) > p["barren_s"] * 10:
                best["barren"] = self.step + int(p["barren_s"] * 10)
                m["tree_id"] = None
            move_d, turn = 0.0, 0.08 * m["scan_dir"]
            close = [s for s in sibs if s["distance"] < p["crowd_dist"]]
            if close:
                s = min(close, key=lambda o: o["distance"])
                move_dir = wrap(s["angle"] + math.pi); move_d = min(speed, 3.0)
            m["t"] = m.get("t", 0) + 1
            return move_d, move_dir, turn
        # 4. explore: least-recently-seen cell, biome prior, persistent target
        m["tree_id"] = None
        tgt = m.get("explore")
        if tgt is None or self.step - m.get("explore_t", -10 ** 9) > p["explore_reeval"] \
                or math.hypot(tgt[0] - X, tgt[1] - Y) < p["explore_reach"]:
            tgt = self._explore_target(a, m)
            m["explore"] = tgt; m["explore_t"] = self.step
        self.diag["explore"] += 1
        if tgt is None:
            return super()._forage(a, m, fruits, trees, sibs, edges, stats)
        md, mdir, d = self._goto(m, tgt[0], tgt[1], speed, edges)
        self._stuck_check(m, d)
        if m.get("stuck_until", 0) > self.step:
            m["t"] = m.get("t", 0) + 1
            return speed, m["stuck_dir"], 0.0
        m["t"] = m.get("t", 0) + 1
        return md, mdir, 0.0

    def _stuck_check(self, m, d):
        p = self.p
        pr = m.get("prog")
        if pr is None or self.step - pr[0] >= p["stuck_steps"]:
            if pr is not None and pr[1] - d < p["stuck_gain"]:
                m["stuck_until"] = self.step + 25
                m["stuck_dir"] = wrap((m.get("stuck_dir", 0.0) + 2.1))
                self.diag["stuck"] += 1
            m["prog"] = (self.step, d)

    def _explore_target(self, a, m):
        p = self.p
        X, Y = m["X"], m["Y"]
        c = self.p["cell"]
        R = p["explore_d"]
        r = int(R // c) + 1
        i0, j0 = int(X // c), int(Y // c)
        best, bs = None, -1e18
        tp = p["tree_prior"]
        for i in range(max(0, i0 - r), min(self.nx, i0 + r + 1)):
            cx = (i + 0.5) * c
            for j in range(max(0, j0 - r), min(self.ny, j0 + r + 1)):
                cy = (j + 0.5) * c
                d = math.hypot(cx - X, cy - Y)
                if d > R or d < 60.0:
                    continue
                k = i * self.ny + j
                age = min(self.step - self.seen[k], p["explore_age"])
                if age < p["explore_age"] * 0.5:
                    continue
                b = self.biome[k]
                prior = tp.get(b, 1.05) if b else 1.05
                score = age / p["explore_age"] * 300.0 - d * prior
                if score > bs:
                    best, bs = (cx, cy), score
        return best


def make(**kw):
    class H(Hivemind):
        def __init__(self, params=None):
            q = dict(kw)
            if params:
                q.update(params)
            super().__init__(q)
    return H
