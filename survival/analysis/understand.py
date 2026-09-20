#!/usr/bin/env python3
"""Understanding-lane instrumentation (analysis only, NOT a scorer; never used for keep/discard).

Runs a policy on seeds with the unmodified simulator and records, per game:
  * score decomposition: survival time, fruit bonus (+energy/1000), eaten penalty (-energy/100)
  * every death: time, cause (predator phase vs energy<=0; old = age > max_age), age, energy, sprint-locked
  * every birth: time, parent energy/age
  * fruit fate: eaten vs rotted, age (s) and energy when eaten
  * a 10 s time series: agents, predators (awake/resting), trees (fruiting = age>=20), fruits, mean agent energy
  * the extinction: cause/time of the last deaths, population 50/100/200 s before the end

Usage: UPSTREAM=/workspace/upstream python3 survival/analysis/understand.py <policy.py> <seeds a-b|a,b> [--procs N] [--out f.json] [--max-time T]
Seeds 5000-5031 (holdout) are refused.
"""
import argparse, json, math, os, sys, time, multiprocessing as mp

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
    deaths, births, eaten, rotted = [], [], [], 0
    fruit_spawned = [0]
    pen = [0.0]

    o_nas = Environment.non_agent_step
    def nas(self, dt):
        G["phase"] = "agents"
        return o_nas(self, dt)
    Environment.non_agent_step = nas

    o_pstep = Predator.step
    def pstep(self, observation=None):
        G["phase"] = "pred"
        return o_pstep(self, observation)
    Predator.step = pstep

    o_kill = Environment.kill_agent
    def kill(self, agent):
        if agent in self.agents:
            if G["phase"] == "pred":
                cause = "predator"
                pen[0] += agent.energy / 100.0
                pr = min(self.predators, key=lambda p: math.hypot(p.x - agent.x, p.y - agent.y))
                pe = pr.energy
            else:
                cause = "old" if agent.age > agent.max_age else "starve"
                pe = None
            deaths.append({"t": round(self.time, 1), "cause": cause, "age": round(agent.age, 1), "max_age": round(agent.max_age, 1),
                           "E": round(agent.energy, 1), "locked": agent.energy < agent.max_energy / 5, "n_before": len(self.agents),
                           "pred_E_after": None if pe is None else round(pe, 1)})
        o_kill(self, agent)
    Environment.kill_agent = kill

    o_spawn = Environment.spawn_agent
    def spawn(self, x=None, y=None, parent=None):
        if parent is not None:
            births.append({"t": round(self.time, 1), "pE": round(parent.energy, 1), "page": round(parent.age, 1), "n": len(self.agents)})
        return o_spawn(self, x, y, parent)
    Environment.spawn_agent = spawn

    o_rm = Environment.remove_fruit
    def rm(self, fruit):
        nonlocal rotted
        if fruit in self.fruits:
            if fruit.age > 100:
                rotted += 1
            else:
                eaten.append((round(self.time, 1), round(fruit.age / 2.0, 1), round(fruit.energy, 1)))
        return o_rm(self, fruit)
    Environment.remove_fruit = rm

    o_sf = Environment.spawn_fruit
    def sf(self, *a, **k):
        r = o_sf(self, *a, **k)
        if r is not None:
            fruit_spawned[0] += 1
        return r
    Environment.spawn_fruit = sf

    series = []
    t0 = time.time()
    with contextlib.redirect_stdout(io.StringIO()):
        sim = SimulationCore(seed=seed)
        env = sim.env
        hm = mod.Hivemind()
        actions, step, err = [], 0, None
        while True:
            state = sim.step(actions)
            step += 1
            if step % 100 == 0:
                ag = env.agents
                series.append({"t": round(env.time), "n": len(ag), "P": len(env.predators),
                               "Pawake": sum(1 for p in env.predators if not p.resting),
                               "trees": len(env.trees), "ftrees": sum(1 for tr in env.trees if tr.age >= 20),
                               "fruits": len(env.fruits), "ripe": sum(1 for f in env.fruits if f.energy >= 59.9),
                               "Emean": round(sum(a.energy for a in ag) / len(ag), 1) if ag else 0,
                               "old": sum(1 for a in ag if a.age > a.max_age),
                               "locked": sum(1 for a in ag if a.energy < a.max_energy / 5)})
            obs = [o for o in state["observations"] if o is not None]
            try:
                acts = hm.act(obs)
                actions = [(a["agent_id"], ActionRequest(**a)) for a in acts]
            except Exception as e:
                err = repr(e); break
            if state["num_agents"] == 0 or env.time > max_time:
                break
    T = env.time
    score = state["score"]
    return {"seed": seed, "score": score, "time": T, "eaten_pen": pen[0], "fruit_bonus": score - T + pen[0],
            "deaths": deaths, "births": births, "eaten": eaten, "rotted": rotted, "fruit_spawned": fruit_spawned[0],
            "series": series, "err": err, "wall": time.time() - t0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("policy"); ap.add_argument("seeds")
    ap.add_argument("--procs", type=int, default=16); ap.add_argument("--out", default="understand.json")
    ap.add_argument("--max-time", type=float, default=3000)
    a = ap.parse_args()
    if "-" in a.seeds and "," not in a.seeds:
        lo, hi = map(int, a.seeds.split("-")); seeds = list(range(lo, hi + 1))
    else:
        seeds = [int(s) for s in a.seeds.split(",")]
    if any(5000 <= s < 5032 for s in seeds):
        sys.exit("refusing holdout seeds 5000-5031")
    with mp.get_context("spawn").Pool(min(a.procs, len(seeds))) as pool:
        res = pool.map(one, [(os.path.abspath(a.policy), s, a.max_time) for s in seeds])
    json.dump({"policy": a.policy, "results": res}, open(a.out, "w"))
    sc = [r["score"] for r in res]
    print(f"done {a.policy} n={len(res)} mean={sum(sc)/len(sc):.1f} -> {a.out}")


if __name__ == "__main__":
    main()
