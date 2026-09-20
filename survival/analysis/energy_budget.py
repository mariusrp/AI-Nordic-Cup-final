#!/usr/bin/env python3
"""Understanding lane, cycle 5: HERD ENERGY BUDGET (analysis only; NOT a scorer).
Where does the herd's energy go? Read-only hooks on the unmodified simulator, per 300-s window, summed over agents:
  income : fruit energy eaten (fruit.energy; eats by an agent already at max energy are counted as cap_hits), newborn endowment (+75)
  spend  : base drain (dt x biome drain), aging drain (0.01 x age per step once age > max_age), walking (0.05/px up to
           speed), sprinting (0.5/px above speed, after the sim's caps and the max/5 lock), turning (|turn|/2pi, capped
           at pi), spawning (100 per birth), energy carried into predators (victim energy at death).
Also the share of agent-steps standing / walking / sprinting. Mirrors environment.update_entity_position exactly.
Usage: UPSTREAM=/workspace/upstream python3 survival/analysis/energy_budget.py <policy.py> <a-b> [--procs N] [--out f]
Seeds 5000-5031 (holdout) are refused.
"""
import argparse, json, math, os, sys, time, multiprocessing as mp
from collections import defaultdict

UP = os.environ.get("UPSTREAM", "/home/claude/Nordic-AI-Cup-2026") + "/survival-simulator"
KEYS = ("fruit", "endow", "base", "aging", "walk", "sprint", "turn", "spawn", "eaten",
        "steps", "st_stand", "st_walk", "st_sprint")


def one(args):
    policy_path, seed, max_time = args
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    sys.path.insert(0, UP)
    import io, contextlib, importlib.util
    from src.core import SimulationCore
    from src.utils.DTOs import ActionRequest
    from src.elements.environment import Environment
    from src.elements.predator import Predator
    from src.elements.agent import Agent
    spec = importlib.util.spec_from_file_location("pol", policy_path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    B = defaultdict(lambda: {k: 0.0 for k in KEYS})
    G = {"phase": "agents"}

    def w(env):
        return B[min(int(env.time // 300), 6)]

    o_pos = Environment.update_entity_position
    def upos(self, entity, distance, direction=None, local_obstacles=None):
        if isinstance(entity, Agent):
            d = max(0.0, float(distance))
            d = min(d, entity.sprint_speed)
            if entity.energy < entity.max_energy / 5 and d > entity.speed:
                d = entity.speed
            b = w(self)
            b["walk"] += min(d, entity.speed) * 0.05
            b["sprint"] += max(0.0, d - entity.speed) * 0.5
            b["steps"] += 1
            if d <= 1e-9:
                b["st_stand"] += 1
            elif d <= entity.speed:
                b["st_walk"] += 1
            else:
                b["st_sprint"] += 1
        return o_pos(self, entity, distance, direction, local_obstacles)
    Environment.update_entity_position = upos

    o_dir = Environment.update_entity_direction
    def udir(self, entity, turn_angle):
        if isinstance(entity, Agent):
            w(self)["turn"] += min(math.pi, abs(turn_angle)) / (2 * math.pi)
        return o_dir(self, entity, turn_angle)
    Environment.update_entity_direction = udir

    o_nas = Environment.non_agent_step
    def nas(self, dt):
        b = w(self)
        for a in self.agents:
            ix = min(max(int(a.x), 0), self.width - 1); iy = min(max(int(a.y), 0), self.height - 1)
            b["base"] += dt * self.biome_map[ix, iy].energy_drain_rate
            if a.age + dt > a.max_age:
                b["aging"] += 0.01 * (a.age + dt)
        return o_nas(self, dt)
    Environment.non_agent_step = nas

    o_rm = Environment.remove_fruit
    def rmf(self, fruit):
        if fruit in self.fruits and fruit.age <= 100:
            b = w(self)
            b["fruit"] += fruit.energy
            for ag in self.agents:
                if math.hypot(ag.x - fruit.x, ag.y - fruit.y) < ag.size + fruit.radius + 1e-6 and ag.energy >= ag.max_energy - 1e-9:
                    b.setdefault("cap_hits", 0.0); b["cap_hits"] += 1
                    break
        return o_rm(self, fruit)
    Environment.remove_fruit = rmf

    o_as = Environment.agent_step
    def astep(self, agent_id, move_distance, move_direction, turn_angle, spawn_agent=False):
        n0 = len(self.agents)
        r = o_as(self, agent_id, move_distance, move_direction, turn_angle, spawn_agent)
        if len(self.agents) > n0:
            b = w(self); b["spawn"] += 100; b["endow"] += 75
        return r
    Environment.agent_step = astep

    o_pstep = Predator.step
    def pstep(self, observation=None):
        G["phase"] = "pred"
        return o_pstep(self, observation)
    Predator.step = pstep

    o_kill = Environment.kill_agent
    def kill(self, agent):
        if agent in self.agents and G["phase"] == "pred" and agent.energy > 0:
            w(self)["eaten"] += agent.energy
        o_kill(self, agent)
    Environment.kill_agent = kill

    t0 = time.time(); err = None
    with contextlib.redirect_stdout(io.StringIO()):
        sim = SimulationCore(seed=seed)
        env = sim.env
        start_E = sum(a.energy for a in env.agents)
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
    return {"seed": seed, "score": state["score"], "time": env.time, "err": err, "start_E": start_E,
            "budget": {str(k): v for k, v in B.items()}, "wall": round(time.time() - t0, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("policy"); ap.add_argument("seeds")
    ap.add_argument("--procs", type=int, default=8); ap.add_argument("--out", default="energy_budget.json")
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
    tot = defaultdict(float)
    for r in res:
        for v in r["budget"].values():
            for k, x in v.items():
                tot[k] += x
    spend = sum(tot[k] for k in ("base", "aging", "walk", "sprint", "turn", "spawn", "eaten"))
    inc = tot["fruit"] + tot["endow"] + sum(r["start_E"] for r in res)
    print(f"income {inc:.0f} (fruit {tot['fruit']:.0f}, endow {tot['endow']:.0f}); spend {spend:.0f}: " +
          ", ".join(f"{k} {100 * tot[k] / spend:.1f}%" for k in ("base", "aging", "walk", "sprint", "turn", "spawn", "eaten")))
    st_ = tot["steps"] or 1
    print(f"agent-steps: stand {100 * tot['st_stand'] / st_:.1f}%, walk {100 * tot['st_walk'] / st_:.1f}%, sprint {100 * tot['st_sprint'] / st_:.1f}%; "
          f"fruit cap hits {tot.get('cap_hits', 0):.0f}")


if __name__ == "__main__":
    main()
