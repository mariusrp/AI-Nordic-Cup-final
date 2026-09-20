#!/usr/bin/env python3
"""Understanding lane, cycle 4: CONTACT + TAIL census (analysis only, NOT a scorer; never used for keep/discard).

Runs a policy on the unmodified simulator with read-only hooks and records, per game:
  * contact EPISODES per predator: maximal runs of steps in which the predator is within R_ENG (250 px, its vision
    range) of the nearest agent, resting or awake, gaps < 5 s merged. Per episode: predator spawn index, start time,
    duration, kills, awake share, share of awake steps in which it perceived an agent, how it began (spawned in /
    resting / awake), and the number of wakes inside it. From these: the share of predator-time in contact, kills per
    contact-second, and the gap until a predator re-engages (the re-encounter process that 'break contact' must beat).
  * kills (time, killer index, killer's episode age) and deaths by cause, births.
  * a 10 s series: n agents, predators, awake, engaged, herd centroid and radius, trees, fruit.
  * the TAIL: a ring buffer of the last 300 s with, per agent-step, age, energy, max energy, px actually moved,
    requested move/turn/spawn, visible trees/fruit/predators, and the TRUE distance to the nearest tree and fruit.
    Summaries for the last 120 s, for each starvation death in the tail (its last 30 s) and for the final agent.

Usage: UPSTREAM=/workspace/upstream python3 survival/analysis/contact_tail.py <policy.py> <seeds a-b> [--procs N] [--out f.json]
Seeds 5000-5031 (holdout) are refused.
"""
import argparse, json, math, os, sys, time, multiprocessing as mp
from collections import deque

UP = os.environ.get("UPSTREAM", "/home/claude/Nordic-AI-Cup-2026") + "/survival-simulator"
R_ENG = 250.0
GAP = 50          # steps: episode gaps shorter than this are merged
TAIL = 3000       # steps kept in the tail ring buffer (300 s)


def one(args):
    policy_path, seed, max_time = args
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    sys.path.insert(0, UP)
    import io, contextlib, importlib.util
    import numpy as np
    from src.core import SimulationCore
    from src.utils.DTOs import ActionRequest
    from src.elements.environment import Environment
    from src.elements.predator import Predator
    spec = importlib.util.spec_from_file_location("pol", policy_path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)

    G = {"step": 0, "cur_pred": None, "phase": "agents"}
    pidx, pspawn = {}, {}          # id(pred) -> spawn order, spawn step
    perceiving = {}                # id(pred) -> this step's (n agents perceived, closest distance)
    req = {}                       # agent_id -> (req_move, turn, spawn, moved px)
    kills, deaths, births = [], [], []
    open_ep, episodes = {}, []
    was_rest = {}

    o_sp = Environment.spawn_predator
    def sp(self, *a_, **k_):
        r_ = o_sp(self, *a_, **k_)
        if r_ is not None:
            pidx[id(r_)] = len(pidx); pspawn[id(r_)] = G["step"]
        return r_
    Environment.spawn_predator = sp

    o_pstep = Predator.step
    def pstep(self, observation=None):
        G["phase"] = "pred"; G["cur_pred"] = self
        ag = [o for o in (observation or []) if o.get("type") == "Agent"]
        perceiving[id(self)] = (len(ag), min((o["distance"] for o in ag), default=None))
        return o_pstep(self, observation)
    Predator.step = pstep

    o_as = Environment.agent_step
    def astep(self, agent_id, move_distance, move_direction, turn_angle, spawn_agent=False):
        a = self.agents_dict.get(agent_id)
        if a is None:
            return o_as(self, agent_id, move_distance, move_direction, turn_angle, spawn_agent)
        x0, y0, n0 = a.x, a.y, len(self.agents)
        r_ = o_as(self, agent_id, move_distance, move_direction, turn_angle, spawn_agent)
        req[agent_id] = (float(move_distance), float(turn_angle), bool(spawn_agent), math.hypot(a.x - x0, a.y - y0))
        if len(self.agents) > n0:
            births.append({"t": round(self.time, 1), "parent_age": round(a.age, 1), "parent_E": round(a.energy, 1),
                           "n": len(self.agents)})
        return r_
    Environment.agent_step = astep

    o_kill = Environment.kill_agent
    def kill(self, agent):
        if agent in self.agents:
            if G["phase"] == "pred" and agent.energy > 0:
                P = G["cur_pred"]; pid = id(P)
                ep = open_ep.get(pid)
                kills.append({"t": round(self.time, 1), "killer": pidx.get(pid), "E": round(agent.energy, 1),
                              "age": round(agent.age, 1),
                              "ep_age": None if ep is None else round((G["step"] - ep["s0"]) / 10, 1),
                              "pred_age": None if pspawn.get(pid) is None else round((G["step"] - pspawn[pid]) / 10, 1)})
                if ep is not None:
                    ep["kills"] += 1
                cause = "predator"
            else:
                cause = "old" if agent.age > agent.max_age else "starve"
            deaths.append({"t": round(self.time, 1), "cause": cause, "age": round(agent.age, 1), "id": agent.agent_id,
                           "maxE": round(agent.max_energy)})
        o_kill(self, agent)
    Environment.kill_agent = kill

    def close(pid, ep):
        dur = (ep["last"] - ep["s0"] + 1)
        episodes.append({"pred": pidx.get(pid), "t0": round(ep["s0"] / 10, 1), "dur": round(dur / 10, 1),
                         "kills": ep["kills"], "awake": ep["awake"], "perc": ep["perc"], "steps": ep["steps"],
                         "wakes": ep["wakes"], "how": ep["how"], "d0": round(ep["d0"], 1),
                         "pred_age0": round((ep["s0"] - pspawn.get(pid, ep["s0"])) / 10, 1), "open_end": ep.get("open_end", False)})

    series = []
    tail = deque(maxlen=TAIL)
    kills_c = 0

    t0 = time.time()
    with contextlib.redirect_stdout(io.StringIO()):
        sim = SimulationCore(seed=seed)
        env = sim.env
        o_nas = Environment.non_agent_step
        hm = mod.Hivemind()
        actions, err = [], None
        while True:
            G["phase"] = "agents"; G["step"] += 1
            perceiving.clear(); req.clear()
            state = sim.step(actions)
            s = G["step"]
            ag = env.agents
            preds = env.predators
            # ---- contact bookkeeping (after this step's moves and kills) ----
            if ag:
                A = np.array([[a.x, a.y] for a in ag])
            else:
                A = np.zeros((0, 2))
            for p in preds:
                pid = id(p)
                if len(A):
                    dmin = float(np.min(np.hypot(A[:, 0] - p.x, A[:, 1] - p.y)))
                else:
                    dmin = 1e9
                woke = was_rest.get(pid, True) and not p.resting
                was_rest[pid] = p.resting
                ep = open_ep.get(pid)
                if dmin < R_ENG:
                    if ep is not None and s - ep["last"] > GAP:
                        close(pid, ep); ep = None
                    if ep is None:
                        how = "spawned" if s - pspawn.get(pid, -999) <= 10 else ("resting" if p.resting else "awake")
                        ep = {"s0": s, "last": s, "kills": 0, "awake": 0, "perc": 0, "steps": 0, "wakes": 0, "how": how, "d0": dmin}
                        open_ep[pid] = ep
                    ep["last"] = s; ep["steps"] += 1
                    if not p.resting:
                        ep["awake"] += 1
                        if perceiving.get(pid, (0, None))[0] > 0:
                            ep["perc"] += 1
                    if woke:
                        ep["wakes"] += 1
                elif ep is not None and s - ep["last"] > GAP:
                    close(pid, ep); del open_ep[pid]
            if s % 100 == 0:
                if len(A):
                    c = A.mean(axis=0); rad = float(np.median(np.hypot(A[:, 0] - c[0], A[:, 1] - c[1])))
                else:
                    c, rad = (None, None), None
                series.append({"t": round(env.time), "n": len(ag), "P": len(preds),
                               "awake": sum(1 for p in preds if not p.resting),
                               "eng": sum(1 for p in preds if id(p) in open_ep and s - open_ep[id(p)]["last"] == 0),
                               "eng_awake": sum(1 for p in preds if id(p) in open_ep and s - open_ep[id(p)]["last"] == 0 and not p.resting),
                               "cx": None if c[0] is None else round(float(c[0])), "cy": None if c[1] is None else round(float(c[1])),
                               "rad": None if rad is None else round(rad), "trees": len(env.trees), "fruits": len(env.fruits),
                               "kills": len(kills), "births": len(births)})
            # ---- tail ring buffer ----
            if ag:
                T = np.array([[t_.x, t_.y] for t_ in env.trees]) if env.trees else None
                F = np.array([[f.x, f.y] for f in env.fruits]) if env.fruits else None
                rec = []
                for i, a in enumerate(ag):
                    ob = env.agent_observations.get(a.agent_id, [])
                    nt = sum(1 for o in ob if o.get("type") == "Tree")
                    nf = sum(1 for o in ob if o.get("type") == "Fruit")
                    npd = sum(1 for o in ob if o.get("type") == "Predator")
                    dt_ = float(np.min(np.hypot(T[:, 0] - a.x, T[:, 1] - a.y))) if T is not None else 9999.0
                    df_ = float(np.min(np.hypot(F[:, 0] - a.x, F[:, 1] - a.y))) if F is not None else 9999.0
                    rq = req.get(a.agent_id, (0.0, 0.0, False, 0.0))
                    rec.append((a.agent_id, round(a.age, 1), round(a.energy, 1), round(a.max_energy), round(rq[3], 2),
                                round(rq[0], 2), round(rq[1], 3), rq[2], nt, nf, npd, round(dt_), round(df_),
                                round(a.speed, 1), round(a.max_age, 1)))
                tail.append((round(env.time, 1), rec))
            obs = [o for o in state["observations"] if o is not None]
            if state["num_agents"] == 0 or env.time > max_time:
                break
            try:
                acts = hm.act(obs)
                actions = [(a["agent_id"], ActionRequest(**a)) for a in acts]
            except Exception as e:
                err = repr(e); break
        for pid, ep in list(open_ep.items()):
            ep["open_end"] = True
            close(pid, ep)
        pred_life = [{"pred": pidx[k], "spawn_t": round(v / 10, 1)} for k, v in pspawn.items()]

    # ---- tail summaries ----
    T_end = env.time
    tl = list(tail)
    def win(t_lo):
        rs = [(t, r) for t, rr in tl if t >= t_lo for r in rr]
        if not rs:
            return None
        n_ = len(rs)
        return {"agent_steps": n_,
                "moving": round(sum(1 for _, r in rs if r[4] > 0.5) / n_, 3),
                "req_move": round(sum(1 for _, r in rs if r[5] > 0.5) / n_, 3),
                "turning": round(sum(1 for _, r in rs if abs(r[6]) > 0.01) / n_, 3),
                "tree_vis": round(sum(1 for _, r in rs if r[8] > 0) / n_, 3),
                "fruit_vis": round(sum(1 for _, r in rs if r[9] > 0) / n_, 3),
                "pred_vis": round(sum(1 for _, r in rs if r[10] > 0) / n_, 3),
                "tree_true_lt100": round(sum(1 for _, r in rs if r[11] < 100) / n_, 3),
                "tree_true_lt250": round(sum(1 for _, r in rs if r[11] < 250) / n_, 3),
                "fruit_true_lt100": round(sum(1 for _, r in rs if r[12] < 100) / n_, 3),
                "fruit_true_lt250": round(sum(1 for _, r in rs if r[12] < 250) / n_, 3),
                "E_mean": round(sum(r[2] for _, r in rs) / n_, 1),
                "locked": round(sum(1 for _, r in rs if r[2] < r[3] / 5) / n_, 3),
                "spawn_req": sum(1 for _, r in rs if r[7])}
    per_agent = {}
    for t, rr in tl:
        for r in rr:
            per_agent.setdefault(r[0], []).append((t, r))
    starve_tail = []
    for d in deaths:
        if d["cause"] != "predator" and d["t"] >= T_end - 120 and d["id"] in per_agent:
            h = [x for x in per_agent[d["id"]] if x[0] >= d["t"] - 30]
            if not h:
                continue
            n_ = len(h)
            first = per_agent[d["id"]][0][1]
            starve_tail.append({"t": d["t"], "cause": d["cause"], "age": d["age"], "maxE": d["maxE"],
                                "max_age": h[-1][1][14],
                                "moving30": round(sum(1 for _, r in h if r[4] > 0.5) / n_, 2),
                                "fruit_vis30": round(sum(1 for _, r in h if r[9] > 0) / n_, 2),
                                "tree_vis30": round(sum(1 for _, r in h if r[8] > 0) / n_, 2),
                                "tree_true_end": h[-1][1][11], "fruit_true_end": h[-1][1][12],
                                "tree_true_min30": min(r[11] for _, r in h), "fruit_true_min30": min(r[12] for _, r in h),
                                "E_30s_before": h[0][1][2], "E_first_in_buffer": first[2],
                                "t_first_in_buffer": per_agent[d["id"]][0][0]})
    nser = []
    for t, rr in tl[::100]:
        nser.append((t, len(rr)))
    last = tl[-1][1] if tl else []
    summ = {"T_end": round(T_end, 1), "last120": win(T_end - 120), "last300": win(T_end - 300),
            "n_series_tail": nser, "starve_tail": starve_tail,
            "last_death": deaths[-1] if deaths else None,
            "trees_end": len(env.trees), "fruits_end": len(env.fruits), "preds_end": len(env.predators)}
    return {"seed": seed, "score": state["score"], "time": env.time, "kills": kills, "deaths": deaths, "births": births,
            "episodes": episodes, "series": series, "pred_life": pred_life, "tail": summ, "err": err,
            "wall": time.time() - t0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("policy"); ap.add_argument("seeds")
    ap.add_argument("--procs", type=int, default=8); ap.add_argument("--out", default="contact.json")
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
