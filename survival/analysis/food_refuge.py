#!/usr/bin/env python3
"""Understanding lane, cycle 5: FOOD-MISS BRANCHES + REFUGE REACH (analysis only; NOT a scorer, never keep/discard).

Runs a policy with the bb_mpc structure (Hivemind._act_one / _forage) on the unmodified simulator with read-only hooks.
A) FOOD MISSES. Every agent-step in which a HUNGRY agent (energy < eat_energy_frac * max) perceives >= 1 fruit gets a
   branch label from wrapping _act_one/_forage (the policy code is not changed):
     flee          danger branch (close sprint / back-pedal): _forage was never called
     chaser_block  _forage wanted to move but the final move is 0 (far-chaser or watcher 'never step toward it')
     sleep_divert  final move direction differs from _forage's (sleep_escape walks away from a resting predator)
     target        moving to a perceived fruit (claim won)
     deferred      every perceived fruit is claimed by a needier / lower-id visible sibling (food_r2_2 rule)
     other         anything else (e.g. crowd push)
   plus: did the agent eat within the next 30 steps (fruit removal next to it, rot excluded), energy bucket, and the
   last-30-s label mix of every agent that starved (not predator, not old).
B) REFUGE REACH. refuge cells as in refuge_geometry.py (agent-reachable, >= 15 px from every predator-reachable centre).
   Geodesic distance to the nearest refuge cell through agent-free space (dilation with alternating 4/8-neighbourhoods
   ~ octagonal metric, up to 400 px). For each predator kill: the victim's refuge distance and the killer's distance at
   t-1/-3/-5/-10 s; base rate over all agent-steps (every 10 steps).
C) deaths by cause and score.
Usage: UPSTREAM=/workspace/upstream python3 survival/analysis/food_refuge.py <policy.py> <a-b> [--procs N] [--out f]
Seeds 5000-5031 (holdout) are refused.
"""
import argparse, json, math, os, sys, time, multiprocessing as mp
from collections import deque, defaultdict

UP = os.environ.get("UPSTREAM", "/home/claude/Nordic-AI-Cup-2026") + "/survival-simulator"
HERE = os.path.dirname(os.path.abspath(__file__))
LAGS = (10, 30, 50, 100)            # steps before the kill
DMAX = 400
BUCKETS = (15, 30, 60, 100, 150, 250, 400)


def geodesic_refuge(env):
    import numpy as np
    from scipy import ndimage
    sys.path.insert(0, HERE)
    from refuge_geometry import free_mask, main_component
    W, H = env.width, env.height
    ra = main_component(free_mask(env.obstacles, W, H, 5))
    rp = main_component(free_mask(env.obstacles, W, H, 10))
    refuge = ra & (ndimage.distance_transform_edt(~rp) >= 15.0)
    dist = np.full((W, H), 9999, dtype=np.int16)
    if not refuge.any():
        return dist, dist, 0
    dist[refuge] = 0
    cur = refuge.copy()
    s4 = ndimage.generate_binary_structure(2, 1); s8 = ndimage.generate_binary_structure(2, 2)
    for k in range(1, DMAX + 1):
        nxt = ndimage.binary_dilation(cur, structure=(s8 if k % 2 == 0 else s4), mask=ra)
        new = nxt & ~cur
        if not new.any():
            break
        dist[new] = k
        cur = nxt
    euc = np.minimum(ndimage.distance_transform_edt(~refuge), 9999).astype(np.int16)
    return dist, euc, int(refuge.sum())


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
    Base = mod.Hivemind
    wrap = mod.wrap

    G = {"step": 0, "phase": "agents", "cur_pred": None}
    FOOD = []                   # (step, aid, energy, max_e, n, nearest_fruit_d, label)
    EAT = defaultdict(list)     # aid -> steps it ate
    kills, deaths = [], []
    hist_a = {}                 # aid -> deque[(step, x, y)]
    hist_p = {}                 # id(pred) -> deque[(step, x, y, resting)]
    base_rate = [0] * (len(BUCKETS) + 1)

    class H(Base):
        def _forage(self, a, m, fruits, trees, sibs, edges, stats):
            r = Base._forage(self, a, m, fruits, trees, sibs, edges, stats)
            self._fcall = (r, fruits, sibs, stats)
            return r

        def _act_one(self, a, m, n, spawned, fits, med, stats, cur_by, fde):
            self._fcall = None
            act, sp = Base._act_one(self, a, m, n, spawned, fits, med, stats, cur_by, fde)
            p = self.p
            fr = [o for o in a["observations"] if o["type"] == "Fruit"]
            e, me = a["energy"], a["max_energy"]
            if fr and e < p["eat_energy_frac"] * me:
                if self._fcall is None:
                    lab = "flee"
                else:
                    (fm_d, fm_dir, _t), fruits, sibs, st = self._fcall
                    if act["move_distance"] == 0.0 and fm_d > 0:
                        lab = "chaser_block"
                    elif act["move_distance"] > 0 and abs(wrap(act["move_direction"] - fm_dir)) > 1e-9:
                        lab = "sleep_divert"
                    else:
                        own = mod.need_rank(e, me, a["age"], p)
                        usable = []
                        for f in fruits:
                            fx, fy = mod.to_xy(f["distance"], f["angle"])
                            radius = f["distance"] + p["claim_radius"]
                            beaten = False
                            for s in sibs:
                                sx, sy = mod.to_xy(s["distance"], s["angle"])
                                if math.hypot(fx - sx, fy - sy) > radius:
                                    continue
                                se, sme, sag = st.get(s["id"], (me, me, 999.0))
                                sr = mod.need_rank(se, sme, sag, p)
                                if sr < own or (sr == own and s["id"] < a["agent_id"]):
                                    beaten = True; break
                            if not beaten:
                                usable.append(f)
                        if not p.get("claim", 1):
                            usable = fruits
                        if usable:
                            tg = min(usable, key=lambda o: o["distance"])
                            lab = "target" if (fm_d > 0 and abs(wrap(fm_dir - tg["angle"])) < 1e-6) else "other"
                        else:
                            lab = "deferred"
                FOOD.append((G["step"], a["agent_id"], round(e, 1), round(me), n,
                             round(min(o["distance"] for o in fr), 1), lab))
            return act, sp

    o_sp = Environment.spawn_predator
    def spp(self, *a_, **k_):
        return o_sp(self, *a_, **k_)
    Environment.spawn_predator = spp

    o_pstep = Predator.step
    def pstep(self, observation=None):
        G["phase"] = "pred"; G["cur_pred"] = self
        return o_pstep(self, observation)
    Predator.step = pstep

    o_rm = Environment.remove_fruit
    def rmf(self, fruit):
        if fruit in self.fruits and fruit.age <= 100:
            best, bd = None, 1e9
            for ag in self.agents:
                d = math.hypot(ag.x - fruit.x, ag.y - fruit.y)
                if d < ag.size + fruit.radius + 1e-6 and d < bd:
                    best, bd = ag, d
            if best is not None:
                EAT[best.agent_id].append(G["step"])
        return o_rm(self, fruit)
    Environment.remove_fruit = rmf

    DG = {}
    o_kill = Environment.kill_agent
    def kill(self, agent):
        if agent in self.agents:
            s = G["step"]
            if G["phase"] == "pred" and agent.energy > 0:
                P = G["cur_pred"]
                ha = hist_a.get(agent.agent_id, deque()); hp = hist_p.get(id(P), deque())
                rec = {"t": round(self.time, 1), "E": round(agent.energy, 1), "maxE": round(agent.max_energy),
                       "age": round(agent.age, 1), "locked": agent.energy < agent.max_energy / 5, "n": len(self.agents)}
                dg, de = DG["geo"], DG["euc"]
                ix = lambda x: min(max(int(x), 0), self.width - 1)
                iy = lambda y: min(max(int(y), 0), self.height - 1)
                rec["ref0"] = int(dg[ix(agent.x), iy(agent.y)])
                for L in LAGS:
                    pa = next((h for h in ha if h[0] == s - L), None)
                    pp = next((h for h in hp if h[0] == s - L), None)
                    rec[f"ref{L}"] = None if pa is None else int(dg[ix(pa[1]), iy(pa[2])])
                    rec[f"refe{L}"] = None if pa is None else int(de[ix(pa[1]), iy(pa[2])])
                    rec[f"dp{L}"] = None if (pa is None or pp is None) else round(math.hypot(pa[1] - pp[1], pa[2] - pp[2]))
                    rec[f"rest{L}"] = None if pp is None else bool(pp[3])
                kills.append(rec)
                cause = "predator"
            else:
                cause = "old" if agent.age > agent.max_age else "starve"
            deaths.append({"t": round(self.time, 1), "step": s, "cause": cause, "age": round(agent.age, 1),
                           "id": agent.agent_id})
        o_kill(self, agent)
    Environment.kill_agent = kill

    t0 = time.time()
    err = None
    with contextlib.redirect_stdout(io.StringIO()):
        sim = SimulationCore(seed=seed)
        env = sim.env
        DG["geo"], DG["euc"], ref_px = geodesic_refuge(env)
        hm = H()
        actions = []
        while True:
            G["phase"] = "agents"; G["step"] += 1
            state = sim.step(actions)
            s = G["step"]
            for a in env.agents:
                h = hist_a.get(a.agent_id)
                if h is None:
                    h = hist_a[a.agent_id] = deque(maxlen=101)
                h.append((s, a.x, a.y))
                if s % 10 == 0:
                    d = int(DG["geo"][min(max(int(a.x), 0), env.width - 1), min(max(int(a.y), 0), env.height - 1)])
                    k = next((i for i, b in enumerate(BUCKETS) if d <= b), len(BUCKETS))
                    base_rate[k] += 1
            for p in env.predators:
                h = hist_p.get(id(p))
                if h is None:
                    h = hist_p[id(p)] = deque(maxlen=101)
                h.append((s, p.x, p.y, p.resting))
            if s % 500 == 0:
                alive = {a.agent_id for a in env.agents}
                for k in [k for k in hist_a if k not in alive]:
                    del hist_a[k]
            obs = [o for o in state["observations"] if o is not None]
            if state["num_agents"] == 0 or env.time > max_time:
                break
            try:
                acts = hm.act(obs)
                actions = [(a["agent_id"], ActionRequest(**a)) for a in acts]
            except Exception as e:
                err = repr(e); break
    end = G["step"]

    # ---- food-miss aggregation ----
    def ate_after(aid, st, w=30):
        return any(st < x <= st + w for x in EAT.get(aid, ()))
    agg = defaultdict(lambda: [0, 0])      # (phase, ebucket, label) -> [n, ate30]
    for (st, aid, e, me, n, fd, lab) in FOOD:
        ph = "tail120" if st > end - 1200 else "early"
        eb = "locked" if e < me / 5 else ("low" if e < 0.4 * me else "mid")
        c = agg[(ph, eb, lab)]
        c[0] += 1; c[1] += int(ate_after(aid, st))
    by_aid = defaultdict(list)
    for r in FOOD:
        by_aid[r[1]].append(r)
    starv = []
    for d in deaths:
        if d["cause"] != "starve":
            continue
        rs = [r for r in by_aid.get(d["id"], []) if r[0] > d["step"] - 300]
        mix = defaultdict(int)
        for r in rs:
            mix[r[6]] += 1
        starv.append({"t": d["t"], "age": d["age"], "tail": d["step"] > end - 1200, "fruit_steps30": len(rs),
                      "mix": dict(mix), "nearest_fd": min((r[5] for r in rs), default=None),
                      "ate30s": sum(1 for x in EAT.get(d["id"], ()) if x > d["step"] - 300)})
    return {"seed": seed, "score": state["score"], "time": env.time, "end_step": end, "err": err, "ref_px": ref_px,
            "kills": kills, "deaths": deaths, "base_rate": base_rate,
            "food": [[k[0], k[1], k[2], v[0], v[1]] for k, v in agg.items()], "starv": starv,
            "eats": sum(len(v) for v in EAT.values()), "wall": round(time.time() - t0, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("policy"); ap.add_argument("seeds")
    ap.add_argument("--procs", type=int, default=8); ap.add_argument("--out", default="food_refuge.json")
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
