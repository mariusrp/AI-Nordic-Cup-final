"""Rule-based hivemind policy v1 (1733 on seed 1). Angles relative to heading; move_direction relative."""
import math

TWO_PI = 2 * math.pi

DEFAULT = dict(
    flee_dist=140.0, sprint_dist=70.0, face_pred=1.0, eat_energy_frac=0.85, tree_stay=35.0,
    spawn_energy=260.0, spawn_age=55.0, spawn_energy_old=140.0, max_herd=12, scan_turn=0.35,
    crowd_dist=40.0, crowd_n=3,
)


def wrap(a):
    return (a + math.pi) % TWO_PI - math.pi


class Hivemind:
    def __init__(self, params=None):
        self.p = dict(DEFAULT)
        if params:
            self.p.update(params)
        self.mem = {}

    def act(self, agents, n_total=None):
        p = self.p
        n = len(agents)
        actions = []
        spawned_this_step = 0
        alive_ids = set()
        for a in agents:
            aid = a["agent_id"]
            alive_ids.add(aid)
            m = self.mem.setdefault(aid, {"scan_dir": 1 if aid % 2 else -1, "wander": 0})
            obs = a["observations"]
            energy = a["energy"]; max_e = a["max_energy"]; speed = a["speed"]; sprint = a["sprint_speed"]
            preds = [o for o in obs if o["type"] == "Predator"]
            fruits = [o for o in obs if o["type"] == "Fruit"]
            trees = [o for o in obs if o["type"] == "Tree"]
            sibs = [o for o in obs if o["type"] == "Agent"]
            edges = [o for o in obs if o["type"] == "Edge"]
            move_d, move_dir, turn, spawn = 0.0, 0.0, 0.0, False
            thr = p["spawn_energy"] if a["age"] < p["spawn_age"] else p["spawn_energy_old"]
            if energy > max(thr, 101) and n + spawned_this_step < p["max_herd"]:
                spawn = True; spawned_this_step += 1
            elif n + spawned_this_step <= 1 and energy > 101:
                spawn = True; spawned_this_step += 1
            near_pred = [o for o in preds if o["distance"] < p["flee_dist"]]
            if near_pred:
                vx = vy = 0.0
                for o in near_pred:
                    w = 1.0 / max(o["distance"], 1.0)
                    vx -= w * math.cos(o["angle"]); vy -= w * math.sin(o["angle"])
                flee = self._avoid_edges(math.atan2(vy, vx), edges)
                closest = min(near_pred, key=lambda o: o["distance"])
                move_d = sprint if (closest["distance"] < p["sprint_dist"] and energy > max_e / 5 + 10) else speed
                move_dir = flee
                if p["face_pred"] > 0.5:
                    turn = wrap(closest["angle"])
                spawn = False
            else:
                hungry = energy < p["eat_energy_frac"] * max_e
                target = None
                if fruits and hungry:
                    target = min(fruits, key=lambda o: o["distance"])
                    move_dir = target["angle"]; move_d = min(speed, target["distance"])
                elif trees:
                    tree = min(trees, key=lambda o: o["distance"])
                    crowd = sum(1 for s in sibs if s["distance"] < 80)
                    if tree["distance"] > p["tree_stay"] + 10:
                        move_dir = tree["angle"]; move_d = min(speed, tree["distance"] - p["tree_stay"])
                    else:
                        move_d = 0.0; turn = 0.08 * m["scan_dir"]
                    if crowd > p["crowd_n"] and fruits == [] and m["wander"] <= 0:
                        m["wander"] = 60
                elif fruits:
                    move_d = 0.0; turn = 0.08 * m["scan_dir"]
                else:
                    m["wander"] = max(m["wander"], 1)
                if m["wander"] > 0 and target is None and not (trees and m["wander"] <= 0):
                    if not trees or m["wander"] > 0:
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
            actions.append({"agent_id": aid, "move_distance": float(move_d), "move_direction": float(move_dir),
                            "turn_angle": float(turn), "spawn_agent": bool(spawn)})
        for k in list(self.mem):
            if k not in alive_ids:
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
