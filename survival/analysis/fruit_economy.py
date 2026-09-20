#!/usr/bin/env python3
"""Understanding lane, cycle 5: FRUIT ECONOMY (analysis only; NOT a scorer).
Hooks spawn_fruit / remove_fruit on the unmodified simulator. For every fruit: spawned, eaten (energy and age at the
moment it is eaten, and the eater's age / energy / max energy / herd size / hidden max_age) or rotted (age > 100 = 50 s). Per 300-s
window: fruits spawned, eaten, rotted, energy eaten, and the ripeness mix of what is eaten. Also every birth (parent
age, energy and hidden max_age) and whether the newborn starved, was killed or aged out.
Usage: UPSTREAM=/workspace/upstream python3 survival/analysis/fruit_economy.py <policy.py> <a-b> [--procs N] [--out f]
Seeds 5000-5031 (holdout) are refused.
"""
import argparse, json, math, os, sys, time, multiprocessing as mp
from collections import defaultdict

UP = os.environ.get("UPSTREAM", "/home/claude/Nordic-AI-Cup-2026") + "/survival-simulator"


def one(args):
    policy_path, seed, max_time = args
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    sys.path.insert(0, UP)
    import io, contextlib, importlib.util
    from src.core import SimulationCore
    from src.utils.DTOs import ActionRequest
    from src.elements.environment import Environment
    from src.elements.predator import Predator
    spec = importlib.util.spec_from_file_location("pol", policy_path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    G = {"phase": "agents", "env": None}
    W = defaultdict(lambda: {"spawned": 0, "eaten": 0, "rotted": 0, "E_eaten": 0.0, "ripe": 0, "unripe_lt40": 0})
    eats = []
    births = {}
    fate = {}

    def win(t):
        return min(int(t // 300), 6)

    o_sf = Environment.spawn_fruit
    def sf(self, *a, **k):
        f = o_sf(self, *a, **k)
        if f is not None:
            W[win(self.time)]["spawned"] += 1
        return f
    Environment.spawn_fruit = sf

    o_rm = Environment.remove_fruit
    def rmf(self, fruit):
        if fruit in self.fruits:
            w = W[win(self.time)]
            if fruit.age > 100:
                w["rotted"] += 1
            else:
                best, bd = None, 1e9
                for ag in self.agents:
                    d = math.hypot(ag.x - fruit.x, ag.y - fruit.y)
                    if d < ag.size + fruit.radius + 1e-6 and d < bd:
                        best, bd = ag, d
                w["eaten"] += 1; w["E_eaten"] += fruit.energy
                w["ripe"] += int(fruit.energy >= 59.9); w["unripe_lt40"] += int(fruit.energy < 40)
                if best is not None:
                    # the eater's energy AFTER eating (the sim adds fruit.energy, capped at max, before remove_fruit)
                    eats.append((round(self.time, 1), round(fruit.energy, 1), best.agent_id, round(best.age, 1),
                                 round(best.energy, 1), round(best.max_energy), len(self.agents), round(best.max_age, 1)))
        return o_rm(self, fruit)
    Environment.remove_fruit = rmf

    o_pstep = Predator.step
    def pstep(self, observation=None):
        G["phase"] = "pred"
        return o_pstep(self, observation)
    Predator.step = pstep

    o_as = Environment.agent_step
    def astep(self, agent_id, move_distance, move_direction, turn_angle, spawn_agent=False):
        a = self.agents_dict.get(agent_id)
        n0 = len(self.agents)
        e0 = a.energy if a is not None else None
        r = o_as(self, agent_id, move_distance, move_direction, turn_angle, spawn_agent)
        if a is not None and len(self.agents) > n0:
            c = self.agents[-1]
            births[c.agent_id] = (round(self.time, 1), round(a.age, 1), round(e0, 1), round(a.max_energy), round(a.max_age, 1))
        return r
    Environment.agent_step = astep

    o_kill = Environment.kill_agent
    def kill(self, agent):
        if agent in self.agents:
            if G["phase"] == "pred" and agent.energy > 0:
                c = "predator"
            else:
                c = "old" if agent.age > agent.max_age else "starve"
            fate[agent.agent_id] = (c, round(agent.age, 1))
        o_kill(self, agent)
    Environment.kill_agent = kill

    t0 = time.time(); err = None
    with contextlib.redirect_stdout(io.StringIO()):
        sim = SimulationCore(seed=seed)
        env = sim.env
        hm = mod.Hivemind()
        actions = []
        while True:
            G["phase"] = "agents"
            state = sim.step(actions)
            obs = [o for o in state["observations"] if o is not None]
            if state["num_agents"] == 0 or env.time > max_time:
                break
            try:
                acts = hm.act(obs)
                actions = [(a["agent_id"], ActionRequest(**a)) for a in acts]
            except Exception as e:
                err = repr(e); break
    bl = []
    for cid, (t, pa, pe, pme, pma) in births.items():
        f = fate.get(cid)
        bl.append((t, pa, pe, pme, None if f is None else f[0], None if f is None else f[1], pma))
    return {"seed": seed, "score": state["score"], "time": env.time, "err": err,
            "windows": {str(k): v for k, v in W.items()}, "eats": eats, "births": bl, "wall": round(time.time() - t0, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("policy"); ap.add_argument("seeds")
    ap.add_argument("--procs", type=int, default=8); ap.add_argument("--out", default="fruit_economy.json")
    ap.add_argument("--max-time", type=float, default=3000)
    a = ap.parse_args()
    lo, hi = map(int, a.seeds.split("-"))
    seeds = list(range(lo, hi + 1))
    if any(5000 <= s < 5032 for s in seeds):
        sys.exit("refusing holdout seeds 5000-5031")
    with mp.get_context("spawn").Pool(min(a.procs, len(seeds))) as pool:
        res = pool.map(one, [(os.path.abspath(a.policy), s, a.max_time) for s in seeds])
    json.dump({"policy": a.policy, "results": res}, open(a.out, "w"))
    sc = [r["score"] for r in res]
    print(f"done {a.policy} n={len(res)} mean={sum(sc)/len(sc):.1f} errs={sum(1 for r in res if r['err'])} -> {a.out}")


if __name__ == "__main__":
    main()
