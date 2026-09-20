#!/usr/bin/env python3
"""Understanding lane, cycle 4: are there PREDATOR-PROOF REFUGES on the map? (analysis only; map generation, no game)

Collision rule (environment.py _in_obstacle): a move is refused if the new centre lies inside any obstacle rectangle
expanded by the entity's radius (agent 5, predator 10; open boxes, square corners). Kills need centre distance
< 15 (predator.size + agent.size), through walls too. So a gap between obstacles of width 10-20 px lets an agent
through but not a predator. A REFUGE cell = a centre an agent can reach from the main free region (r=5) that is
>= 15 px from every centre a predator can reach from ITS main free region (r=10). This script rasterises each
seed's obstacles at 1 px, labels the connected free regions and measures the refuge cells.
Usage: python3 survival/analysis/refuge_geometry.py 2000-2049 [--trees]
"""
import os, sys, math
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
UP = os.environ.get("UPSTREAM", "/home/claude/Nordic-AI-Cup-2026") + "/survival-simulator"
sys.path.insert(0, UP)
import io, contextlib
import numpy as np
from scipy import ndimage


def free_mask(obstacles, W, H, r):
    blocked = np.zeros((W, H), dtype=bool)
    for o in obstacles:
        x0 = int(math.floor(o.x - r)) + 1; x1 = int(math.ceil(o.x + o.width + r)) - 1
        y0 = int(math.floor(o.y - r)) + 1; y1 = int(math.ceil(o.y + o.height + r)) - 1
        blocked[max(0, x0):min(W, x1 + 1), max(0, y0):min(H, y1 + 1)] = True
    # _keep_agent_in_bounds clamps centres to [r, W-r]
    blocked[:r, :] = True; blocked[W - r:, :] = True; blocked[:, :r] = True; blocked[:, H - r:] = True
    return ~blocked


def main_component(free):
    lab, n = ndimage.label(free)
    if n == 0:
        return np.zeros_like(free)
    sizes = ndimage.sum(free, lab, index=np.arange(1, n + 1))
    return lab == (1 + int(np.argmax(sizes)))


def analyse(seed):
    from src.core import SimulationCore
    with contextlib.redirect_stdout(io.StringIO()):
        sim = SimulationCore(seed=seed)
    env = sim.env
    W, H = env.width, env.height
    fa = free_mask(env.obstacles, W, H, 5)
    fp = free_mask(env.obstacles, W, H, 10)
    ra = main_component(fa)
    rp = main_component(fp)
    dist_to_pred = ndimage.distance_transform_edt(~rp)
    refuge = ra & (dist_to_pred >= 15.0)
    lab, n = ndimage.label(refuge)
    comps = []
    if n:
        sizes = ndimage.sum(refuge, lab, index=np.arange(1, n + 1))
        coms = ndimage.center_of_mass(refuge, lab, index=np.arange(1, n + 1))
        maxd = ndimage.maximum(dist_to_pred, lab, index=np.arange(1, n + 1))
        for k in range(n):
            comps.append((int(sizes[k]), round(float(maxd[k]), 1), (round(coms[k][0]), round(coms[k][1]))))
    comps.sort(reverse=True)
    trees = [(t.x, t.y) for t in env.trees]
    near = []
    for c in comps[:5]:
        cx, cy = c[2]
        near.append(min((math.hypot(tx - cx, ty - cy) for tx, ty in trees), default=None))
    return {"seed": seed, "agent_free_main": int(ra.sum()), "pred_free_main": int(rp.sum()),
            "refuge_px": int(refuge.sum()), "n_refuges": n, "top": comps[:5],
            "tree_dist_top": [None if d is None else round(d) for d in near]}


if __name__ == "__main__":
    lo, hi = map(int, sys.argv[1].split("-"))
    seeds = [s for s in range(lo, hi + 1) if not (5000 <= s < 5032)]
    rows = []
    for s in seeds:
        r = analyse(s)
        rows.append(r)
        print(r, flush=True)
    has = [r for r in rows if r["n_refuges"] > 0]
    big = [r for r in rows if r["top"] and r["top"][0][0] >= 50]
    print(f"SUMMARY seeds={len(rows)} with_any_refuge={len(has)} with_refuge>=50px={len(big)} "
          f"median_refuge_px={sorted(r['refuge_px'] for r in rows)[len(rows)//2]}")
