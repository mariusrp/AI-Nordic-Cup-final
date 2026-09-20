"""end_pred (surv4, 20 Sep 2026) = lin_map + ENDGAME PREDATOR-PROOFING.  *** MEASURED LOSS - DEFAULT OFF ***

RESULT FIRST (frozen scorer, Mac, seeds 9000-9063, --max-time 3000, paired against a lin_map draw of the same
seeds in the same machine conditions):

    lin_map (base)  64 seeds  mean 1236.7  p25 1043.0  p50 1282.4  min 414.2  >=1500 20.3%  pred deaths 2336
    ep_on=1  (arm A) 64 seeds mean 1110.5  p25  892.3  p50 1132.0  min 604.2  >=1500 10.9%  pred deaths 3091
    PAIRED diff -126.3 se 37.6  z -3.36  wins 20/64                                          (predator deaths +32%)
    ep_on=1 + repo_on=1 (arm C), 32 seeds: -61.7 se 60.1, pred deaths 1538 vs 1199 (+28%)

So ep_on DEFAULTS TO 0 and this file is behaviourally identical to lin_map out of the box. The mechanisms below
are kept because the wall map itself is exact and cheap and may be useful for something else; the LOSS is in
using it to steer the flee direction. Turn the screened arm back on with EP_PARAMS='{"ep_on":1}'.

WHY IT LOSES (the useful part of this negative result): the premise is right - the map is remembered exactly, one
sighting of a border wall gives the whole hivemind a 1600/1200 px segment forever, and 304 of the map's 336 edges
are stored by t=120 s. What is wrong is the conclusion that a fleeing agent should therefore steer toward open
ground. bb_juke's juke (1.25 rad off the away vector, toward the side the chaser's heading already errs to)
beats the predator's 0.3 rad/step turn cap, and the plain back-pedal drains it 2.55/step; both point at "away",
and any wall-aware correction points somewhere else. With ~350 remembered segments most directions near an
obstacle have < ep_min_room px of free path, so the corridor fires often and each time it trades a working
evasion for a longer escape corridor - and predator deaths go UP 32%, not down. The agent being pinned against a
wall is where kills are OBSERVED, not what causes them; the simulator's collision handler already slides an agent
along a wall at full speed, which is as good as the corridor and costs no turn energy and no detour.

lin_map is kept byte-for-byte in behaviour when ep_on=0 (the default; EP_PARAMS='{"ep_on":0}' or END_PRED=0 also
force it); every mechanism below is switchable so the orchestrator can fall back.

The problem it attacks: by t=1500 there are 15-20 predators and 3-5 agents, so ONE encounter ends the run. The kill
autopsy (lin_tail, LEDGER 20 Sep 01:30) found fast agents are not out-run in a fair chase - they are pinned against
the map border or an obstacle while back-pedalling, because EDGES ARE ONLY OBSERVED INSIDE THE VISION CONE and a
fleeing agent faces the predator, i.e. away from the wall it is backing into. lin_tail's F3 tried a per-agent, 60 s
odometry wall memory and it acted on ~3% of close-predator steps (score flat). lin_map changed the premise: the
hivemind now has an EXACT SHARED dead-reckoned frame, so walls can be remembered globally and permanently.

Mechanisms:
  A wall_on - SHARED WALL MAP. Every observed Edge is the FULL obstacle edge (upstream sensing.compute_visibility
    returns the un-clipped segment), so one sighting of a border wall stores a 1600/1200 px segment that every agent
    then knows, for the rest of the game, from anywhere. Segments are stored in the shared map frame, deduplicated,
    transformed on frame merges, and indexed in a coarse grid. Obstacles never move, so entries never expire.
  B flee_on - OPEN-CORRIDOR FLEE. Inside the predator branches (juke at close_d, back-pedal at back_d) the direction
    is chosen from remembered + visible edges: keep bb_juke's direction when it has >= ep_min_room px of free path,
    otherwise take the direction closest to it that does, otherwise the most open one. Never back-pedal into a wall.
  C drift_on - LATE IDLE WALL DRIFT. From drift_t, an agent that would stand still (foraging move 0) and is within
    drift_min px of a remembered wall takes one small step toward open ground, so it keeps an escape corridor before
    a predator ever shows up. Capped by drift_px per step and only while it stays inside tree_stay of its tree.
  D nsd_on - NO BIRTH NEXT TO A PREDATOR (lin_tail F2): a child is born 10-30 px from the parent with 75 energy and
    no predator model; suppress scheduled/elder births while an awake threat is inside nsd_d.
  F repo_on - OPEN-GROUND REPOSITIONING. The predator closes 10.6 px/step even while PIVOTING around a facing agent
    (sprint 15 at 45 deg off the bearing), so facing only buys time proportional to how fast the agent can back away -
    and against a wall that is zero. While the chaser is still outside back_d the agent is idle anyway (the FDE far
    branch zeroes any foraging move toward it), so an idle agent with less than repo_gap px of remembered wall
    clearance walks repo_px/step toward the most open direction that also leads away from the predator, while facing
    it. Facing keeps the predator in the cheap pivot regime, so the sideways drift is nearly free (0.05/px).
  E pm_on - SHARED PREDATOR MEMORY. Predator sightings (pose + heading + rest) are kept in the shared frame and
    inflated by pm_grow px/s of elapsed time; from pm_t a foraging agent will not walk into the inflated disc of a
    predator it cannot currently see, and an idle agent inside it drifts out.
"""
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lin_map  # noqa: E402
from lin_map import MAPP  # noqa: E402
from bb_juke import wrap, TWO_PI  # noqa: E402

EP = dict(
    ep_on=0,   # measured -126.3 se 37.6 (z -3.36) on 64 seeds; see the module docstring
    # A + B: wall map and open-corridor flee
    wall_on=1, flee_on=1, seg_cell=120.0, seg_tol=5.0, seg_max=1500,
    ep_R=220.0, ep_min_room=60.0, ep_fan=36, ep_back=1, ep_close=1,
    # C: late idle wall drift
    drift_on=0, drift_t=600.0, drift_min=70.0, drift_px=3.0, drift_R=160.0,
    # D: no birth next to a predator
    nsd_on=0, nsd_d=150.0,
    # E: shared predator memory
    pm_on=0, pm_t=600.0, pm_keep=40.0, pm_grow=9.0, pm_r0=70.0, pm_max=220.0,
    # F: open-ground repositioning while the predator is still far
    repo_on=0, repo_d=260.0, repo_gap=110.0, repo_px=8.0, repo_R=220.0, repo_min_e=40.0,
    repo_room=120.0, repo_w=0.7, repo_fan=24,
)
if os.environ.get("EP_PARAMS"):
    EP.update(json.loads(os.environ["EP_PARAMS"]))
if os.environ.get("END_PRED") == "0":
    EP["ep_on"] = 0

_orig_avoid = lin_map.Hivemind._avoid_edges.__func__ if hasattr(lin_map.Hivemind._avoid_edges, "__func__") \
    else lin_map.Hivemind._avoid_edges


class Hivemind(lin_map.Hivemind):
    def __init__(self, params=None):
        q = dict(MAPP)
        q.update(EP)
        if params:
            q.update(params)
        super().__init__(q)
        self.segs = {}        # id -> [x1, y1, x2, y2, F]
        self.segkey = {}      # rounded key -> id
        self.sgrid = {}       # (F, cx, cy) -> set(ids)
        self.next_seg = 0
        self.preds = {}       # id -> {F, x, y, psi, t, rest}
        self.next_pm = 0
        self._m = None
        self._ep_danger = False
        self.diag.update({"segs": 0, "ep_calls": 0, "ep_veto": 0, "ep_fix": 0, "ep_open": 0,
                          "drift": 0, "repo": 0, "nsd_block": 0, "pm_veto": 0, "pm_drift": 0})

    # ------------------------------------------------------------------ segment store
    def _seg_cells(self, x1, y1, x2, y2, F):
        c = self.p["seg_cell"]
        L = math.hypot(x2 - x1, y2 - y1)
        n = max(1, int(L / (c * 0.5)) + 1)
        out = set()
        for k in range(n + 1):
            u = k / n
            out.add((F, int((x1 + u * (x2 - x1)) // c), int((y1 + u * (y2 - y1)) // c)))
        return out

    def _add_seg(self, F, x1, y1, x2, y2):
        if len(self.segs) >= self.p["seg_max"]:
            return
        if (x2, y2) < (x1, y1):
            x1, y1, x2, y2 = x2, y2, x1, y1
        key = (F, round(x1 / 4.0), round(y1 / 4.0), round(x2 / 4.0), round(y2 / 4.0))
        if key in self.segkey:
            return
        tol = self.p["seg_tol"]
        c = self.p["seg_cell"]
        for i in self.sgrid.get((F, int(x1 // c), int(y1 // c)), ()):
            s = self.segs.get(i)
            if s and abs(s[0] - x1) < tol and abs(s[1] - y1) < tol and abs(s[2] - x2) < tol and abs(s[3] - y2) < tol:
                return
        i = self.next_seg; self.next_seg += 1
        self.segs[i] = [x1, y1, x2, y2, F]
        self.segkey[key] = i
        for cc in self._seg_cells(x1, y1, x2, y2, F):
            self.sgrid.setdefault(cc, set()).add(i)
        self.diag["segs"] = len(self.segs)

    def _near_segs(self, F, x, y, r):
        c = self.p["seg_cell"]
        out = set()
        for cx in range(int((x - r) // c), int((x + r) // c) + 1):
            for cy in range(int((y - r) // c), int((y + r) // c) + 1):
                s = self.sgrid.get((F, cx, cy))
                if s:
                    out |= s
        return out

    def _regrid_segs(self):
        self.sgrid = {}
        self.segkey = {}
        for i, s in self.segs.items():
            x1, y1, x2, y2, F = s
            self.segkey[(F, round(x1 / 4.0), round(y1 / 4.0), round(x2 / 4.0), round(y2 / 4.0))] = i
            for cc in self._seg_cells(x1, y1, x2, y2, F):
                self.sgrid.setdefault(cc, set()).add(i)

    def _transform(self, Fsrc, Fdst, th, tx, ty):
        super()._transform(Fsrc, Fdst, th, tx, ty)
        c, s = math.cos(th), math.sin(th)
        moved = False
        for i, sg in self.segs.items():
            if sg[4] != Fsrc:
                continue
            moved = True
            x1, y1, x2, y2 = sg[0], sg[1], sg[2], sg[3]
            sg[0], sg[1] = c * x1 - s * y1 + tx, s * x1 + c * y1 + ty
            sg[2], sg[3] = c * x2 - s * y2 + tx, s * x2 + c * y2 + ty
            sg[4] = Fdst
        for e in self.preds.values():
            if e["F"] != Fsrc:
                continue
            x, y = e["x"], e["y"]
            e["x"], e["y"] = c * x - s * y + tx, s * x + c * y + ty
            e["psi"] += th
            e["F"] = Fdst
        if moved:
            self._regrid_segs()

    # ------------------------------------------------------------------ map update
    def _map_update(self, agents):
        super()._map_update(agents)
        p = self.p
        if not p["ep_on"]:
            return
        t = (self.step + 1) * 0.1
        if p["wall_on"]:
            for a in agents:
                m = self.mem.get(a["agent_id"])
                if m is None or m.get("unc", 0.0) > 15.0:
                    continue
                F, X, Y, H = m["F"], m["X"], m["Y"], m["H"]
                c, s = math.cos(H), math.sin(H)
                for o in a["observations"]:
                    if o["type"] != "Edge":
                        continue
                    (ax, ay), (bx, by) = o["coords"]
                    self._add_seg(F, X + c * ax - s * ay, Y + s * ax + c * ay,
                                  X + c * bx - s * by, Y + s * bx + c * by)
        if p["pm_on"]:
            for a in agents:
                m = self.mem.get(a["agent_id"])
                if m is None:
                    continue
                F, X, Y, H = m["F"], m["X"], m["Y"], m["H"]
                for o in a["observations"]:
                    if o["type"] != "Predator":
                        continue
                    ang = H + o["angle"]
                    px, py = X + o["distance"] * math.cos(ang), Y + o["distance"] * math.sin(ang)
                    psi = ang + math.pi - o.get("rel_dir", 0.0)
                    hit = None
                    for i, e in self.preds.items():
                        if e["F"] == F and (e["x"] - px) ** 2 + (e["y"] - py) ** 2 < 900.0:
                            hit = i; break
                    if hit is None:
                        hit = self.next_pm; self.next_pm += 1
                        self.preds[hit] = {"F": F, "x": px, "y": py, "psi": psi, "t": t}
                    else:
                        e = self.preds[hit]
                        e["x"], e["y"], e["psi"], e["t"] = px, py, psi, t
            if self.step % 10 == 0:
                for i in [i for i, e in self.preds.items() if t - e["t"] > p["pm_keep"]]:
                    self.preds.pop(i, None)

    # ------------------------------------------------------------------ wall queries
    def _local_walls(self, m, R):
        """Remembered segments within R, expressed in the agent's LOCAL (heading-rotated) frame."""
        F, X, Y, H = m["F"], m["X"], m["Y"], m["H"]
        c, s = math.cos(-H), math.sin(-H)
        out = []
        for i in self._near_segs(F, X, Y, R):
            sg = self.segs.get(i)
            if sg is None:
                continue
            ax, ay, bx, by = sg[0] - X, sg[1] - Y, sg[2] - X, sg[3] - Y
            if min(ax, bx) > R or max(ax, bx) < -R or min(ay, by) > R or max(ay, by) < -R:
                continue
            out.append({"type": "Edge", "coords": ((c * ax - s * ay, s * ax + c * ay),
                                                   (c * bx - s * by, s * bx + c * by))})
        return out

    def _corridor(self, target, edges, m):
        """Direction closest to `target` with >= ep_min_room px of free path (remembered + visible edges)."""
        p = self.p
        self.diag["ep_calls"] += 1
        alle = list(edges) + self._local_walls(m, p["ep_R"])
        base = _orig_avoid(target, alle)
        need = p["ep_min_room"]
        if self._edge_room(base, alle) >= need:
            return base
        self.diag["ep_veto"] += 1
        n = int(p["ep_fan"])
        step = TWO_PI / n
        best, bsc, bopen, bo = None, -1e9, None, -1.0
        for k in range(n):
            d = wrap(target + ((k + 1) // 2) * ((-1) ** k) * step)
            r = self._edge_room(d, alle)
            if r > bo:
                bo, bopen = r, d
            if r >= need:
                sc = math.cos(wrap(d - target))
                if sc > bsc:
                    best, bsc = d, sc
        if best is not None:
            self.diag["ep_fix"] += 1
            return best
        self.diag["ep_open"] += 1
        return bopen if bopen is not None else base

    # ------------------------------------------------------------------ hooks
    def _juke_dir(self, away, rd, edges, aid):
        p = self.p
        if not (p["ep_on"] and p["flee_on"] and p["ep_close"]) or self._m is None:
            return super()._juke_dir(away, rd, edges, aid)
        m = self._m
        alle = list(edges) + self._local_walls(m, p["ep_R"])
        d = super()._juke_dir(away, rd, alle, aid)
        if self._edge_room(d, alle) >= p["ep_min_room"]:
            return d
        return self._corridor(d, edges, m)

    def _avoid_edges(self, direction, edges):
        p = self.p
        if not (p["ep_on"] and p["flee_on"] and p["ep_back"]) or self._m is None or not self._ep_danger:
            return _orig_avoid(direction, edges)
        return self._corridor(direction, edges, self._m)

    def _act_one(self, a, m, n, permit, stats, cur_by, fde):
        p = self.p
        if not p["ep_on"]:
            self._m = None
            return super()._act_one(a, m, n, permit, stats, cur_by, fde)
        self._m = m
        self._a = a
        near, nbear = 1e9, None
        for o in a["observations"]:
            if o["type"] == "Predator" and o["distance"] < near:
                near, nbear = o["distance"], o["angle"]
        X, Y, H = m["X"], m["Y"], m["H"]
        for tr in m.get("tracks", ()):
            if tr["seen"] < self.step and tr["rest"] < p["rest_steps"]:
                dd = math.hypot(tr["wx"] - X, tr["wy"] - Y)
                if dd < near:
                    near, nbear = dd, math.atan2(tr["wy"] - Y, tr["wx"] - X) - H
        self._ep_danger = near <= p["back_d"] + 20.0
        if p["nsd_on"] and permit is not None and near <= p["nsd_d"]:
            permit = None
            self.diag["nsd_block"] += 1
        act = super()._act_one(a, m, n, permit, stats, cur_by, fde)
        if p["repo_on"] and act["move_distance"] == 0.0 and nbear is not None and near < p["repo_d"] \
                and a["energy"] > p["repo_min_e"] and not m.get("elder"):
            act = self._reposition(a, m, act, nbear)
        elif p["drift_on"] and act["move_distance"] == 0.0 and not self._ep_danger:
            act = self._wall_drift(a, m, act)
        self._m = None
        return act

    def _reposition(self, a, m, act, nbear):
        """Idle agent with a predator inside repo_d and too little remembered wall clearance: drift toward the most
        open direction that also leads away from it (facing is unchanged, so the predator stays in pivot mode)."""
        p = self.p
        walls = self._local_walls(m, p["repo_R"])
        if not walls:
            return act
        near_w = 1e9
        for e in walls:
            (x1, y1), (x2, y2) = e["coords"]
            dx, dy = x2 - x1, y2 - y1
            L = dx * dx + dy * dy
            if L == 0:
                continue
            u = max(0.0, min(1.0, -(x1 * dx + y1 * dy) / L))
            near_w = min(near_w, math.hypot(x1 + u * dx, y1 + u * dy))
        if near_w >= p["repo_gap"]:
            return act
        away = wrap(nbear + math.pi)
        R = p["repo_room"]
        n = int(p["repo_fan"])
        best, bsc = None, -1e9
        for k in range(n):
            d = wrap(away + k * (TWO_PI / n))
            r = min(self._edge_room(d, walls), R)
            sc = r / R + p["repo_w"] * math.cos(wrap(d - away))
            if sc > bsc:
                best, bsc = d, sc
        if best is None or self._edge_room(best, walls) < near_w + 10.0:
            return act
        self.diag["repo"] += 1
        act["move_distance"] = float(min(p["repo_px"], min(a["speed"], p["walk_cap"])))
        act["move_direction"] = float(best)
        return act

    def _wall_drift(self, a, m, act):
        """Idle late-game agent too close to a remembered wall: one small step toward open ground."""
        p = self.p
        t = self.step * 0.1
        if t < p["drift_t"] or m.get("elder") or a["energy"] < 30.0:
            return act
        walls = self._local_walls(m, p["drift_R"])
        if not walls:
            return act
        near, nang = 1e9, 0.0
        for e in walls:
            (x1, y1), (x2, y2) = e["coords"]
            dx, dy = x2 - x1, y2 - y1
            L = dx * dx + dy * dy
            if L == 0:
                continue
            u = max(0.0, min(1.0, -(x1 * dx + y1 * dy) / L))
            px, py = x1 + u * dx, y1 + u * dy
            d = math.hypot(px, py)
            if d < near:
                near, nang = d, math.atan2(py, px)
        if near >= p["drift_min"]:
            return act
        away = wrap(nang + math.pi)
        if self._edge_room(away, walls) < p["drift_min"]:
            return act
        self.diag["drift"] += 1
        act["move_distance"] = float(min(p["drift_px"], min(a["speed"], p["walk_cap"])))
        act["move_direction"] = float(away)
        return act

    # ------------------------------------------------------------------ predator memory
    def _forage(self, a, m, fruits, trees, sibs, edges, stats):
        md, mdir, turn = super()._forage(a, m, fruits, trees, sibs, edges, stats)
        p = self.p
        if not (p["ep_on"] and p["pm_on"]) or self.step * 0.1 < p["pm_t"] or not self.preds:
            return md, mdir, turn
        t = self.step * 0.1
        F, X, Y, H = m["F"], m["X"], m["Y"], m["H"]
        worst = None
        for e in self.preds.values():
            if e["F"] != F:
                continue
            r = min(p["pm_max"], p["pm_r0"] + p["pm_grow"] * (t - e["t"]))
            dx, dy = e["x"] - X, e["y"] - Y
            d = math.hypot(dx, dy)
            if d < r and (worst is None or d < worst[0]):
                worst = (d, math.atan2(dy, dx) - H, r)
        if worst is None:
            return md, mdir, turn
        d, bearing, r = worst
        if md > 0.0 and math.cos(wrap(mdir - bearing)) > 0.3:
            self.diag["pm_veto"] += 1
            return 0.0, mdir, turn
        if md == 0.0 and d < r * 0.6:
            self.diag["pm_drift"] += 1
            return min(a["speed"], p["walk_cap"]), self._avoid_edges(wrap(bearing + math.pi), edges), turn
        return md, mdir, turn


def make(**kw):
    class H(Hivemind):
        def __init__(self, params=None):
            q = dict(kw)
            if params:
                q.update(params)
            super().__init__(q)
    return H
