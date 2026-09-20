#!/usr/bin/env python3
"""Understanding lane: measure the predator state machine with the REAL simulator code (1 agent vs 1 predator,
obstacle-free uniform forest, oracle geometry for the agent). Analysis only, not a scorer.

Scenarios (agent energy 400, no aging, predator awake at E0 placed D px away, facing the agent):
  face_back_walk   face the predator and backpedal at walk speed (10)
  face_back_sprint face the predator and backpedal at sprint speed (20)
  face_stand       face the predator, do not move
  flee_walk        turn the back and walk away (10)
  flee_sprint      turn the back and sprint away (20)
Prints: caught?, steps to catch, steps until the predator rests, predator energy trace, agent energy used.
Usage: UPSTREAM=... python3 survival/analysis/predator_mech.py
"""
import math, os, sys, io, contextlib
UP = os.environ.get("UPSTREAM", "/home/claude/Nordic-AI-Cup-2026") + "/survival-simulator"
sys.path.insert(0, UP)
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
from src.core import SimulationCore
from src.elements.biome import Forest_biome
from src.elements.predator import Predator


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def setup(D, E0, seed=1):
    with contextlib.redirect_stdout(io.StringIO()):
        sim = SimulationCore(seed=seed)
    env = sim.env
    env.agents, env.agents_dict, env.predators, env.trees, env.fruits = [], {}, [], [], []
    env.obstacles = env.obstacles[:4]
    env.edges = set()
    for o in env.obstacles:
        env.edges.update([((o.x, o.y), (o.x + o.width, o.y)), ((o.x + o.width, o.y), (o.x + o.width, o.y + o.height)),
                          ((o.x, o.y + o.height), (o.x + o.width, o.y + o.height)), ((o.x, o.y), (o.x, o.y + o.height))])
    fb = Forest_biome()
    env.biome_map[:, :] = fb
    a = env.spawn_agent(x=1400, y=600)
    a.energy = 400.0; a.max_age = 1e9; a.direction = 0.0
    env.predators.append(Predator(1400 + D * math.cos(0.0), 600 + D * math.sin(0.0), rng=env.rng))
    p = env.predators[0]
    p.energy = E0; p.resting = False; p.direction = math.pi
    env._update_spatial_grid()
    env.agents_dict = {a.agent_id: a}
    env.time = 0.0
    return sim, env, a, p


def run(scn, D, E0, steps=130):
    sim, env, a, p = setup(D, E0)
    aid = a.agent_id
    E_start = a.energy
    rest_step, trace, mind = None, [], 1e9
    for k in range(steps):
        if aid not in env.agents_dict:
            return dict(scn=scn, D=D, E0=E0, caught=True, t_catch=k, rest_step=rest_step, mind=round(mind, 1), trace=trace)
        bearing = math.atan2(p.y - a.y, p.x - a.x)
        rel = wrap(bearing - a.direction)  # predator angle relative to agent heading
        if scn.startswith("face"):
            turn = rel + 0.3  # realistic imperfect facing (exact rel_dir==0 makes np.sign()=0: predator then charges straight)
            md = {"face_back_walk": 10.0, "face_back_sprint": 20.0, "face_stand": 0.0}[scn]
            mdir = wrap(rel + math.pi)
        else:
            turn = wrap(rel + math.pi)  # turn back to predator
            md = 10.0 if scn == "flee_walk" else 20.0
            mdir = wrap(rel + math.pi)
        env.agent_step(aid, md, mdir, turn, False)
        env.non_agent_step(0.1)
        env.agents_dict = {x.agent_id: x for x in env.agents}
        env.predators = env.predators[:1]
        d = math.hypot(p.x - a.x, p.y - a.y)
        mind = min(mind, d)
        if p.resting and rest_step is None:
            rest_step = k
        if k % 8 == 0:
            trace.append((k, round(d), round(p.energy), int(p.resting)))
    return dict(scn=scn, D=D, E0=E0, caught=False, t_catch=None, rest_step=rest_step, mind=round(mind, 1),
                agent_E_used=round(E_start - a.energy, 1), trace=trace[:14])


if __name__ == "__main__":
    for E0 in (102.0, 200.0):
        for D in (100.0, 150.0, 170.0):
            for scn in ("face_back_walk", "face_back_sprint", "face_stand", "flee_walk", "flee_sprint"):
                r = run(scn, D, E0)
                print(f"E0={E0:5.0f} D={D:4.0f} {scn:17s} caught={r['caught']!s:5s} t_catch={r['t_catch']} rest_step={r['rest_step']} "
                      f"min_d={r['mind']} agentE_used={r.get('agent_E_used')} trace={r['trace'][:8]}")
