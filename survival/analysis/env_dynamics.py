#!/usr/bin/env python3
"""Understanding lane: policy-free world dynamics over the full 3000 s (analysis only, not a scorer).
One immortal, motionless agent is parked (energy 1e9, no aging) so the game never ends; it never eats
(it stays put far from trees is NOT enforced, so a few fruits may be eaten: negligible).
Records every 50 s: trees, fruiting trees (age>=20), fruits, ripe fruits (energy>=60), predators, awake predators,
fruit spawned / rotted in the last 50 s, and the per-biome area share of the map.
Usage: UPSTREAM=... python3 survival/analysis/env_dynamics.py <seeds a-b> [--procs N] [--out f.json]
"""
import argparse, json, os, sys, multiprocessing as mp

UP = os.environ.get("UPSTREAM", "/home/claude/Nordic-AI-Cup-2026") + "/survival-simulator"


def one(seed):
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    sys.path.insert(0, UP)
    import io, contextlib, collections
    from src.core import SimulationCore
    from src.elements.environment import Environment
    from src.utils.DTOs import ActionRequest
    cnt = {"spawned": 0, "rotted": 0}
    o_sf = Environment.spawn_fruit
    def sf(self, *a, **k):
        r = o_sf(self, *a, **k)
        if r is not None:
            cnt["spawned"] += 1
        return r
    Environment.spawn_fruit = sf
    o_rm = Environment.remove_fruit
    def rm(self, fruit):
        if fruit in self.fruits and fruit.age > 100:
            cnt["rotted"] += 1
        return o_rm(self, fruit)
    Environment.remove_fruit = rm
    with contextlib.redirect_stdout(io.StringIO()):
        sim = SimulationCore(seed=seed)
    env = sim.env
    for a in env.agents[1:]:
        env.kill_agent(a)
    a0 = env.agents[0]
    a0.energy = 1e9; a0.max_energy = 1e9; a0.max_age = 1e12
    bt = collections.Counter(env.biome_map[x, y].type for x in range(0, env.width, 10) for y in range(0, env.height, 10))
    tot = sum(bt.values())
    series, step, last = [], 0, dict(cnt)
    fruit_life = []
    with contextlib.redirect_stdout(io.StringIO()):
        while env.time <= 3000:
            sim.step([(a0.agent_id, ActionRequest(agent_id=a0.agent_id, move_distance=0.0, move_direction=0.0, turn_angle=0.0, spawn_agent=False))])
            a0.energy = 1e9
            step += 1
            if step % 500 == 0:
                series.append({"t": round(env.time), "trees": len(env.trees), "ftrees": sum(1 for t in env.trees if t.age >= 20),
                               "fruits": len(env.fruits), "ripe": sum(1 for f in env.fruits if f.energy >= 59.9),
                               "P": len(env.predators), "Pawake": sum(1 for p in env.predators if not p.resting),
                               "spawned50": cnt["spawned"] - last["spawned"], "rotted50": cnt["rotted"] - last["rotted"]})
                last = dict(cnt)
    return {"seed": seed, "biomes": {k: round(v / tot, 3) for k, v in bt.items()}, "series": series}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("seeds"); ap.add_argument("--procs", type=int, default=8); ap.add_argument("--out", default="env_dynamics.json")
    a = ap.parse_args()
    lo, hi = map(int, a.seeds.split("-"))
    seeds = [s for s in range(lo, hi + 1) if not 5000 <= s < 5032]
    with mp.get_context("spawn").Pool(min(a.procs, len(seeds))) as pool:
        res = pool.map(one, seeds)
    json.dump(res, open(a.out, "w"))
    import statistics as st
    for i, row in enumerate(res[0]["series"]):
        vals = {k: st.mean(r["series"][i][k] for r in res) for k in row if k != "t"}
        print(row["t"], " ".join(f"{k}={v:.1f}" for k, v in vals.items()))


if __name__ == "__main__":
    main()
