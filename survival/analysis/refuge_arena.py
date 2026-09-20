#!/usr/bin/env python3
"""Understanding lane, cycle 4: REFUGE ARENA (analysis only; a measurement harness, never a policy).

refuge_geometry.py finds, on every map, cells an agent (radius 5) can reach but that are >= 15 px from every centre a
predator (radius 10) can reach. This harness checks the claim in the real simulator: on the seed's own map it keeps ONE
agent (energy 500, no aging, standing still, facing away = the predator's 'charge' branch), puts it at the deepest
refuge cell, and puts ONE awake predator (energy 200) at the nearest predator-reachable cell, heading at the agent.
New predator spawns are disabled. It runs 300 s and reports whether the agent survives, how close the predator stays
(does the decoy HOLD it), how much of the time the predator rests, and its energy. Control arm: the same agent put in
open ground at the same distance from the predator.
Usage: python3 survival/analysis/refuge_arena.py 2000-2009 [steps]
"""
import os, sys, math
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
UP = os.environ.get("UPSTREAM", "/home/claude/Nordic-AI-Cup-2026") + "/survival-simulator"
sys.path.insert(0, UP)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import io, contextlib
import numpy as np
from scipy import ndimage
from refuge_geometry import free_mask, main_component


def arena(seed, steps=3000, control=False):
    from src.core import SimulationCore
    from src.utils.DTOs import ActionRequest
    with contextlib.redirect_stdout(io.StringIO()):
        sim = SimulationCore(seed=seed)
    env = sim.env
    W, H = env.width, env.height
    ra = main_component(free_mask(env.obstacles, W, H, 5))
    rp = main_component(free_mask(env.obstacles, W, H, 10))
    dist, (ix, iy) = ndimage.distance_transform_edt(~rp, return_indices=True)
    refuge = ra & (dist >= 15.0)
    if not refuge.any():
        return None
    d_ref = np.where(refuge, dist, -1)
    ax, ay = np.unravel_index(int(np.argmax(d_ref)), d_ref.shape)
    px, py = int(ix[ax, ay]), int(iy[ax, ay])       # nearest predator-reachable centre
    gap = float(dist[ax, ay])
    if control:  # open ground: a predator-reachable cell far from obstacles, predator at the same distance
        dfree = ndimage.distance_transform_edt(rp)
        ox, oy = np.unravel_index(int(np.argmax(dfree)), dfree.shape)
        ax, ay = ox, oy
        px, py = int(ox + gap), int(oy)
    with contextlib.redirect_stdout(io.StringIO()):
        a = env.agents[0]
        for b in list(env.agents[1:]):
            env.kill_agent(b)
        a.x, a.y = float(ax), float(ay)
        a.energy = a.max_energy = 500.0
        a.max_age = 1e9
        for p in list(env.predators):
            env.predators.remove(p)
        P = env.spawn_predator(x=float(px), y=float(py))
        if P is None:
            from src.elements.predator import Predator
            P = Predator(float(px), float(py), rng=env.rng); env.predators.append(P)
        P.resting = False; P.energy = 200.0
        P.direction = math.atan2(a.y - P.y, a.x - P.x)
        a.direction = P.direction  # agent faces away from the predator (back turned -> charge branch)
        env.spawn_predator = lambda *a_, **k_: None
        env._update_agent_grid(); env._update_predator_grid()
        aid = a.agent_id
        dmin, rest, dsum, alive_steps, left_at, near = 1e9, 0, 0.0, 0, None, 0
        for s in range(steps):
            st = sim.step([(aid, ActionRequest(agent_id=aid, move_distance=0.0, move_direction=0.0, turn_angle=0.0,
                                               spawn_agent=False))])
            if st["num_agents"] == 0:
                break
            alive_steps += 1
            d = math.hypot(P.x - a.x, P.y - a.y)
            dmin = min(dmin, d); dsum += d
            rest += P.resting
            near += d < 100
            if left_at is None and d > 150:
                left_at = s
    return {"seed": seed, "control": control, "gap": round(gap, 1), "agent_xy": (int(ax), int(ay)),
            "survived_s": round(alive_steps / 10, 1), "alive": alive_steps == steps,
            "pred_dmin": round(dmin, 1), "pred_dmean": round(dsum / max(alive_steps, 1), 1),
            "pred_rest_share": round(rest / max(alive_steps, 1), 2), "pred_E_end": round(float(P.energy), 1),
            "held_s": round((left_at if left_at is not None else alive_steps) / 10, 1),
            "share_within100": round(near / max(alive_steps, 1), 2)}


if __name__ == "__main__":
    lo, hi = map(int, sys.argv[1].split("-"))
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
    for s in range(lo, hi + 1):
        if 5000 <= s < 5032:
            continue
        for c in (False, True):
            print(arena(s, steps, control=c), flush=True)
