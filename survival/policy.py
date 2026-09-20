"""Hivemind policy v2 for the survival simulator.
Observation angles are relative to each agent's heading; move_direction and turn_angle are relative too.
Agents that observe each other share predator / fruit / tree sightings (one hop), after converting
between their local frames using the mutual bearings."""
import math

TWO_PI = 2 * math.pi
DEFAULT = dict(
    flee_dist=160.0,      # react to predators closer than this
    charge_dist=95.0,     # inside this the predator charges regardless of facing
    sprint_dist=75.0,     # sprint when predator closer than this
    threat_angle=1.2,     # predator heading towards us within this angle counts as threat
    eat_frac=0.9,         # eat fruit when energy below this fraction of max
    tree_stay=30.0,
    spawn_energy=300.0,
    spawn_age=60.0,
    spawn_energy_old=130.0,
    max_herd=10,
    min_herd=3,           # below this, spawn as soon as energy allows
    scan_turn=0.3,
    crowd_dist=25.0,
    share=1.0,            # share sightings between agents
    explore_energy=150.0, # only explore (walk blind) when energy above this
    fruit_max_dist=260.0,
)


def wrap(a):
    return (a + math.pi) % TWO_PI - math.pi


def to_xy(d, a):
    return d * math.cos(a), d * math.sin(a)


class Hivemind:
    def __init__(self, params=None):
        self.p = dict(DEFAULT)
        if params:
            self.p.update(params)
        self.mem = {}

    # ---------- sharing ----------
    def _shared(self, agents):
        """Return per-agent lists of (x,y) positions (own frame) for predators/fruits/trees incl. neighbours' sightings."""
        sees = {}
        own = {}
        for a in agents:
            aid = a["agent_id"]
            sees[aid] = {o["id"]: (o["distance"], o["angle"]) for o in a["observations"] if o["type"] == "Agent" and "id" in o}
            pr, fr, tr = [], [], []
            for o in a["observations"]:
                t = o["type"]
                if t == "Predator":
                    x, y = to_xy(o["distance"], o["angle"])
                    pr.append((x, y, o.get("rel_dir", 0.0), o["angle"]))  # rel_dir angle kept only in own frame
                elif t == "Fruit":
                    fr.append(to_xy(o["distance"], o["angle"]))
                elif t == "Tree":
                    tr.append(to_xy(o["distance"], o["angle"]))
            own[aid] = (pr, fr, tr)
        out = {}
        for a in agents:
            aid = a["agent_id"]
            pr = [(x, y, rd, True) for (x, y, rd, _) in own[aid][0]]
            fr = list(own[aid][1]); tr = list(own[aid][2])
            if self.p["share"] > 0.5:
                for bid, (dab_b, ang_b_in_a) in sees[aid].items():   # a sees b
                    if bid not in sees or aid not in sees[bid] or bid not in own:
                        continue
                    _, ang_a_in_b = sees[bid][aid]                   # b sees a
                    # heading_b - heading_a
                    dh = ang_b_in_a - ang_a_in_b - math.pi
                    bx, by = to_xy(dab_b, ang_b_in_a)
                    c, s = math.cos(dh), math.sin(dh)
                    def tf(x, y):
                        return bx + c * x - s * y, by + s * x + c * y
                    bpr, bfr, btr = own[bid]
                    for (x, y, rd, _) in bpr:
                        X, Y = tf(x, y)
                        # predator heading relative to line pred->a is unknown in a's frame; mark as shared
                        pr.append((X, Y, None, False))
                    for (x, y) in bfr:
                        fr.append(tf(x, y))
                    for (x, y) in btr:
                        tr.append(tf(x, y))
            out[aid] = (pr, fr, tr)
        return out

    def act(self, agents):
        p = self.p
        n = len(agents)
        shared = self._shared(agents)
        actions = []
        spawned = 0
        alive = set()
        for a in agents:
            aid = a["agent_id"]
            alive.add(aid)
            m = self.mem.setdefault(aid, {"scan": 1 if aid % 2 else -1, "t": 0, "explore": 0})
            m["t"] += 1
            obs = a["observations"]
            energy, max_e = a["energy"], a["max_energy"]
            speed, sprint = a["speed"], a["sprint_speed"]
            sibs = [o for o in obs if o["type"] == "Agent"]
            edges = [o for o in obs if o["type"] == "Edge"]
            preds, fruits, trees = shared[aid]

            move_d, move_dir, turn, spawn = 0.0, 0.0, 0.0, False

            # --- threat assessment ---
            threats = []
            for (x, y, rd, direct) in preds:
                d = math.hypot(x, y)
                if d > p["flee_dist"]:
                    continue
                heading_at_us = (rd is None) or abs(rd) < p["threat_angle"]
                if d < p["charge_dist"] or heading_at_us:
                    threats.append((d, math.atan2(y, x), direct))
            if threats:
                vx = vy = 0.0
                for d, ang, _ in threats:
                    w = 1.0 / max(d, 1.0) ** 2
                    vx -= w * math.cos(ang); vy -= w * math.sin(ang)
                flee = self._avoid_edges(math.atan2(vy, vx), edges)
                dmin, amin, _ = min(threats)
                can_sprint = energy > max_e / 5 + 6
                if dmin < p["sprint_dist"] and can_sprint:
                    move_d = sprint
                else:
                    move_d = speed
                move_dir = flee
                turn = wrap(amin)          # face the closest threat -> predator won't charge outside charge_dist
                turn = max(-math.pi, min(math.pi, turn))
            else:
                # --- spawning ---
                thr = p["spawn_energy"] if a["age"] < p["spawn_age"] else p["spawn_energy_old"]
                if n + spawned < p["min_herd"]:
                    thr = min(thr, 130.0)
                if energy > max(thr, 101.0) and n + spawned < p["max_herd"] and (trees or fruits):
                    spawn = True; spawned += 1
                elif n + spawned <= 1 and energy > 101.0:
                    spawn = True; spawned += 1

                hungry = energy < p["eat_frac"] * max_e
                fr = [(math.hypot(x, y), math.atan2(y, x)) for x, y in fruits]
                fr = [f for f in fr if f[0] < p["fruit_max_dist"]]
                tr = [(math.hypot(x, y), math.atan2(y, x)) for x, y in trees]
                if hungry and fr:
                    d, ang = min(fr)
                    move_dir, move_d = ang, min(speed, d)
                    m["explore"] = 0
                elif tr:
                    d, ang = min(tr)
                    if d > p["tree_stay"] + 8:
                        move_dir, move_d = ang, min(speed, d - p["tree_stay"])
                    else:
                        turn = 0.1 * m["scan"]
                    m["explore"] = 0
                else:
                    # nothing known: scan, then explore if we can afford it
                    if energy > p["explore_energy"] or not sibs:
                        phase = (m["t"] // 12) % 3
                        if phase == 0:
                            turn = p["scan_turn"] * m["scan"]
                        else:
                            move_d = speed
                            move_dir = self._avoid_edges(0.0, edges)
                    else:
                        turn = p["scan_turn"] * m["scan"]
                # separation when idle
                if move_d == 0.0:
                    close = [s for s in sibs if s["distance"] < p["crowd_dist"]]
                    if close:
                        s = min(close, key=lambda o: o["distance"])
                        move_dir, move_d = wrap(s["angle"] + math.pi), 2.0
            actions.append({"agent_id": aid, "move_distance": float(move_d), "move_direction": float(move_dir),
                            "turn_angle": float(turn), "spawn_agent": bool(spawn)})
        for k in list(self.mem):
            if k not in alive:
                del self.mem[k]
        return actions

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
