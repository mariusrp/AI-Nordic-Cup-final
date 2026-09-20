#!/usr/bin/env python3
"""Understanding lane: close-range evasion arena (analysis only, NOT a scorer).

1 agent vs 1 CHARGING predator with the REAL simulator code (Environment.agent_step + non_agent_step), uniform
forest, only the 4 boundary walls (never within 350 px), no tree/predator spawning; both are re-centred together when the agent drifts (translation
only, so no wall effects). The agent acts on its REAL observation list, which the sim computes before the
predator moves, i.e. one predator move stale (as in the game).

Start: predator at d0 ~ U(dmin, dmax) px, heading at the agent + U(-0.3, 0.3) rad (it is charging);
agent heading = facing the predator + U(-0.6, 0.6). Predator energy E0 (101 = just woke, 200 = fed,
35 = exhausted, walks at 11). Agent energy 300 (can sprint) or 90 (sprint-locked, max 500).
Rules (all keep facing the predator with a 0.15 rad offset):
  stand          do not move
  radial_walk    walk straight away (10)
  radial_sprint  sprint straight away (20) when not locked (BB2 fde inside 95 px)
  jukeK_walk     walk at K rad off the away vector, toward the side the predator's heading errs to (apred_g3_1)
  jukeK_sD       as jukeK_walk, sprint when d < D (apred_g3_1 default K=1.25, D=40)
  fde_juke       d < 90: juke 1.25 (sprint < 40), d >= 90: walk straight away (backpedal)
Outcome per trial (max 300 steps): caught / escaped (d > 250 or predator resting) / timeout, agent energy used.
Usage: UPSTREAM=... python3 survival/analysis/evasion_arena.py [--trials N] [--procs P] [--dmin 15 --dmax 90]
"""
import argparse, math, os, sys, io, contextlib, random, json, multiprocessing as mp

UP = os.environ.get("UPSTREAM", "/home/claude/Nordic-AI-Cup-2026") + "/survival-simulator"
RULES = ["stand", "radial_walk", "radial_sprint", "juke0.8_walk", "juke1.25_walk", "juke1.57_walk",
         "juke1.25_s40", "juke1.57_s40", "fde_juke"]


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


_ENV = {}


def _base_env():
    """One simulator per worker process (map generation is slow); each trial resets its entities."""
    if "env" not in _ENV:
        sys.path.insert(0, UP)
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        from src.core import SimulationCore
        from src.elements.biome import Forest_biome
        with contextlib.redirect_stdout(io.StringIO()):
            sim = SimulationCore(seed=1)
        env = sim.env
        env.obstacles = env.obstacles[:4]  # the 4 boundary walls only (compute_visibility needs >= 1 edge nearby)
        env.edges = set()
        for o in env.obstacles:
            env.edges.update([((o.x, o.y), (o.x + o.width, o.y)), ((o.x + o.width, o.y), (o.x + o.width, o.y + o.height)),
                              ((o.x, o.y + o.height), (o.x + o.width, o.y + o.height)), ((o.x, o.y), (o.x, o.y + o.height))])
        env.spawn_tree = lambda *a, **k: None
        env.spawn_predator = lambda *a, **k: None
        env.biome_map[:, :] = Forest_biome()
        _ENV["env"] = env
    return _ENV["env"]


def trial(args):
    rule, E0, aE, seed, dmin, dmax = args
    env = _base_env()
    from src.elements.predator import Predator
    R = random.Random(seed)
    env.agents, env.agents_dict, env.predators, env.trees, env.fruits = [], {}, [], [], []
    env.agent_observations = {}
    cx, cy = 800.0, 600.0
    a = env.spawn_agent(x=cx, y=cy)
    a.energy = aE; a.max_age = 1e9
    d0 = R.uniform(dmin, dmax); th = R.uniform(-math.pi, math.pi)
    px, py = cx + d0 * math.cos(th), cy + d0 * math.sin(th)
    p = Predator(px, py, rng=env.rng)
    p.energy = E0; p.resting = False
    p.direction = math.atan2(cy - py, cx - px) + R.uniform(-0.3, 0.3)
    a.direction = math.atan2(py - cy, px - cx) + R.uniform(-0.6, 0.6)
    env.predators.append(p)
    env._update_spatial_grid()
    env.agents_dict = {a.agent_id: a}
    env.time = 0.0
    aid = a.agent_id
    lo = env._get_local_objects(a)
    obs = a.observe(agents=lo[0], fruits=lo[1], trees=lo[2], obstacles=lo[3], predators=lo[4], edges=lo[5])
    E_start = a.energy
    last = (0.0, 0.0)
    for k in range(300):
        if aid not in env.agents_dict:
            return dict(rule=rule, E0=E0, aE=aE, d0=d0, out="caught", k=k, used=E_start)
        po = [o for o in obs if o["type"] == "Predator"]
        md, mdir, turn = 0.0, 0.0, 0.0
        if po:
            o = po[0]
            d, ang, rd = o["distance"], o["angle"], o.get("rel_dir", 0.0)
            can_sprint = a.energy > a.max_energy / 5 + 10
            away = wrap(ang + math.pi)
            side = 1.0 if rd >= 0 else -1.0
            turn = wrap(ang + 0.15)
            if rule == "stand":
                md = 0.0
            elif rule == "radial_walk":
                md, mdir = a.speed, away
            elif rule == "radial_sprint":
                md, mdir = (a.sprint_speed if can_sprint else a.speed), away
            elif rule.startswith("juke"):
                K = float(rule[4:].split("_")[0])
                mdir = wrap(away + side * K)
                sd = float(rule.split("_s")[1]) if "_s" in rule else 0.0  # jukeK_sD: sprint when d < D
                md = a.sprint_speed if (d < sd and can_sprint) else a.speed
            elif rule == "fde_juke":
                if d < 90:
                    mdir = wrap(away + side * 1.25)
                    md = a.sprint_speed if (d < 40 and can_sprint) else a.speed
                else:
                    md, mdir = a.speed, away
            last = (md, mdir)
        else:
            md, mdir = last
        env.agent_step(aid, md, mdir, turn, False)
        env.non_agent_step(0.1)
        env.agents_dict = {x.agent_id: x for x in env.agents}
        if aid not in env.agents_dict:
            return dict(rule=rule, E0=E0, aE=aE, d0=d0, out="caught", k=k, used=E_start - a.energy)
        obs = env.agent_observations.get(aid, [])
        dd = math.hypot(p.x - a.x, p.y - a.y)
        if p.resting or dd > 250:
            return dict(rule=rule, E0=E0, aE=aE, d0=d0, out="escaped", k=k, used=E_start - a.energy)
        # re-centre both (translation only)
        if math.hypot(a.x - cx, a.y - cy) > 200:
            sx, sy = cx - a.x, cy - a.y
            a.x += sx; a.y += sy; p.x += sx; p.y += sy
            env._update_spatial_grid()
    return dict(rule=rule, E0=E0, aE=aE, d0=d0, out="timeout", k=300, used=E_start - a.energy)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=60); ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--dmin", type=float, default=15.0); ap.add_argument("--dmax", type=float, default=90.0)
    ap.add_argument("--out", default="evasion_arena.json")
    ap.add_argument("--rules", default=",".join(RULES), help="comma list; jukeK_walk / jukeK_s40 accept any K")
    ap.add_argument("--E0", default="101,200,35"); ap.add_argument("--aE", default="300,90")
    a = ap.parse_args()
    rules = a.rules.split(","); E0s = [float(x) for x in a.E0.split(",")]; aEs = [float(x) for x in a.aE.split(",")]
    jobs = [(r, E0, aE, 1000 * i + 7, a.dmin, a.dmax) for r in rules for E0 in E0s for aE in aEs
            for i in range(a.trials)]
    with mp.get_context("spawn").Pool(a.procs) as pool:
        res = pool.map(trial, jobs, chunksize=8)
    json.dump(res, open(a.out, "w"))
    from collections import defaultdict
    g = defaultdict(list)
    for r in res:
        g[(r["rule"], r["E0"], r["aE"])].append(r)
    print(f"d0 ~ U({a.dmin},{a.dmax}); trials per cell {a.trials}; cells: caught% (mean agent energy used)")
    hdr = "rule            " + "  ".join(f"E0={E0:3.0f}/{'free' if aE > 100 else 'lock'}" for E0 in E0s for aE in aEs)
    print(hdr)
    for rname in rules:
        cells = []
        for E0 in E0s:
            for aE in aEs:
                L = g[(rname, E0, aE)]
                c = sum(1 for r in L if r["out"] == "caught") / len(L)
                u = sum(r["used"] for r in L) / len(L)
                cells.append(f"{100*c:5.0f}% ({u:4.0f})")
        print(f"{rname:15s} " + "  ".join(cells))
    # caught% by start distance (all cells pooled per rule)
    bins = [(15, 30), (30, 45), (45, 60), (60, 75), (75, 90)]
    print("caught% by d0 bin: " + " ".join(f"{lo}-{hi}" for lo, hi in bins))
    for rname in rules:
        L = [r for r in res if r["rule"] == rname]
        row = []
        for lo, hi in bins:
            B = [r for r in L if lo <= r["d0"] < hi]
            row.append(f"{100 * sum(r['out'] == 'caught' for r in B) / max(1, len(B)):5.0f}%")
        print(f"{rname:15s} " + " ".join(row))


if __name__ == "__main__":
    main()
