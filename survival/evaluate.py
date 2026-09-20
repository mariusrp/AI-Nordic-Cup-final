#!/usr/bin/env python3
"""FROZEN survival scorer. Do not edit during search (orchestrator only).
Usage: evaluate.py <policy.py> [--seeds train|holdout|smoke|a,b,c] [--procs N] [--max-time 3000] [--json out.json]
policy.py must define class Hivemind(params=None) with .act(list_of_agent_state_dicts) -> list of action dicts.
Prints one grep-able line:  RESULT mean=<> se=<> p25=<> min=<> n=<> deaths=<pred/starve/old>"""
import argparse, importlib.util, json, math, os, statistics, sys, time, multiprocessing as mp

UP = os.environ.get("UPSTREAM", "/home/claude/Nordic-AI-Cup-2026") + "/survival-simulator"
SEEDS = {"smoke": [11], "train": list(range(1000, 1016)), "train32": list(range(1000, 1032))}  # holdout seeds are auditor-only


def one(args):
    policy_path, seed, max_time = args
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    sys.path.insert(0, UP)
    import io, contextlib
    from src.core import SimulationCore
    from src.utils.DTOs import ActionRequest
    from src.elements.environment import Environment
    spec = importlib.util.spec_from_file_location("pol", policy_path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    deaths = {"predator": 0, "starve": 0, "old": 0}
    orig = Environment.kill_agent
    def kill(self, agent):
        if agent in self.agents:
            c = "predator" if agent.energy > 0 else ("old" if agent.age > agent.max_age else "starve")
            deaths[c] += 1
        orig(self, agent)
    Environment.kill_agent = kill
    t0 = time.time()
    with contextlib.redirect_stdout(io.StringIO()):
        sim = SimulationCore(seed=seed)
        hm = mod.Hivemind()
        actions, err = [], None
        while True:
            state = sim.step(actions)
            obs = [o for o in state["observations"] if o is not None]
            try:
                acts = hm.act(obs)
                actions = [(a["agent_id"], ActionRequest(**a)) for a in acts]
            except Exception as e:  # a crashing policy scores what it has so far
                err = repr(e); break
            if state["num_agents"] == 0 or sim.env.time > max_time:
                break
    return {"seed": seed, "score": state["score"], "time": sim.env.time, "deaths": deaths, "err": err, "wall": time.time() - t0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("policy"); ap.add_argument("--seeds", default="train"); ap.add_argument("--procs", type=int, default=16)
    ap.add_argument("--max-time", type=float, default=3000); ap.add_argument("--json")
    a = ap.parse_args()
    seeds = SEEDS.get(a.seeds) or [int(s) for s in a.seeds.split(",")]
    with mp.get_context("spawn").Pool(min(a.procs, len(seeds))) as pool:
        res = pool.map(one, [(os.path.abspath(a.policy), s, a.max_time) for s in seeds])
    sc = [r["score"] for r in res]
    mean = statistics.mean(sc); se = statistics.stdev(sc) / math.sqrt(len(sc)) if len(sc) > 1 else 0.0
    p25 = sorted(sc)[len(sc) // 4]
    d = {k: sum(r["deaths"][k] for r in res) for k in ("predator", "starve", "old")}
    errs = [r["err"] for r in res if r["err"]]
    print(f"RESULT mean={mean:.1f} se={se:.1f} p25={p25:.1f} min={min(sc):.1f} n={len(sc)} deaths={d['predator']}/{d['starve']}/{d['old']} errors={len(errs)} wall={max(r['wall'] for r in res):.0f}s")
    if a.json:
        json.dump({"policy": a.policy, "seeds": seeds, "results": res, "mean": mean, "se": se}, open(a.json, "w"))


if __name__ == "__main__":
    main()
