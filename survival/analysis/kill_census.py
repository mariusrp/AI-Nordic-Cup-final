#!/usr/bin/env python3
"""Understanding lane: per-kill census (analysis only, NOT a scorer; never used for keep/discard).

Runs a policy with the unmodified simulator plus read-only hooks and records, for every predator kill:
  * the killer's chase: consecutive steps (back from the kill) in which the victim was the killer's target
    (its closest perceived agent), the predator branch per step (charge_back = victim's back turned,
    charge_close = victim < 90 px, pivot = victim faced it at >= 90 px), chase start distance, killer energy
  * the victim's perception of the killer (matched from the victim's own observation list): perceived at all
    in the chase / in the 60-step window, first-perceived distance, perceived in the last 5 steps
  * multi-predator context: awake predators within 90/250 px of the victim at the kill, distinct predators that
    targeted the victim in the last 30 steps
  * victim state: age, energy, sprint-locked, traits, mean speed over the last 10 steps, siblings within 250 px
  * (cycle 3) the killer's PRE-CHASE history: steps since it last woke from rest (gap in its step records) and
    since it spawned, whether it targeted ANOTHER agent in the 30 steps before the chase (target switch), how
    the chase started (victim inside its 60 px hearing or seen in its 250 px cone), and the victim's own state in
    the 10 steps before the chase (speed px/step, heading change rad/step, whether it saw any predator)
Per game it also logs a 50 s series of herd traits (hearing/speed/sprint/max_energy/vision/cone; counts of
hearing >= 90 and walking speed >= 15) and all deaths by cause.

Usage: UPSTREAM=/workspace/upstream python3 survival/analysis/kill_census.py <policy.py> <seeds a-b> [--procs N] [--out f.json]
Seeds 5000-5031 (holdout) are refused.
"""
import argparse, json, math, os, sys, time, multiprocessing as mp
from collections import deque

UP = os.environ.get("UPSTREAM", "/home/claude/Nordic-AI-Cup-2026") + "/survival-simulator"


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
    from src.elements.creature import Creature
    spec = importlib.util.spec_from_file_location("pol", policy_path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)

    G = {"phase": "agents", "step": 0, "cur_pred": None, "env": None}
    aring = {}   # agent_id -> deque of (step, {pred_id: dist}, energy, x, y, dir)
    pring = {}   # id(pred) -> deque of (step, target_id, branch, d, energy, x, y)
    kills, deaths = [], []

    o_nas = Environment.non_agent_step
    def nas(self, dt):
        G["phase"] = "agents"; G["step"] += 1; G["env"] = self
        return o_nas(self, dt)
    Environment.non_agent_step = nas

    o_obs = Creature.observe
    def observe(self, agents=None, fruits=None, trees=None, obstacles=None, edges=None, predators=None):
        res = o_obs(self, agents=agents, fruits=fruits, trees=trees, obstacles=obstacles, edges=edges, predators=predators)
        if not isinstance(self, Predator):
            seen = {}
            pl = list(predators) if predators else []
            for o in res:
                if o.get("type") == "Predator":
                    for p in pl:
                        if abs(math.hypot(p.x - self.x, p.y - self.y) - o["distance"]) < 1e-6:
                            seen[id(p)] = o["distance"]; break
            aring.setdefault(self.agent_id, deque(maxlen=200)).append(
                (G["step"], seen, self.energy, self.x, self.y, self.direction))
        return res
    Creature.observe = observe

    o_pstep = Predator.step
    def pstep(self, observation=None):
        G["phase"] = "pred"; G["cur_pred"] = self
        agents = [o for o in (observation or []) if o.get("type") == "Agent"]
        edges = [o for o in (observation or []) if o.get("type") == "Edge"]
        if agents:
            c = min(agents, key=lambda f: f["distance"])
            if abs(c["rel_dir"]) > np.pi / 2:
                br = "charge_back"
            elif c["distance"] < self.hearing_radius * 1.5:
                br = "charge_close"
            else:
                br = "pivot"
            tgt, d = c.get("id"), c["distance"]
        else:
            br = "edge" if edges else "wander"; tgt, d = None, None
        pring.setdefault(id(self), deque(maxlen=400)).append((G["step"], tgt, br, d, self.energy, self.x, self.y))
        return o_pstep(self, observation)
    Predator.step = pstep

    spawn_step = {}
    pidx = {}        # id(pred) -> spawn order (stable killer id)
    pkills = {}      # id(pred) -> list of kill times
    o_sp = Environment.spawn_predator
    def sp(self, *a_, **k_):
        r_ = o_sp(self, *a_, **k_)
        if r_ is not None:
            spawn_step[id(r_)] = G["step"]
            pidx[id(r_)] = len(pidx)
        return r_
    Environment.spawn_predator = sp

    o_kill = Environment.kill_agent
    def kill(self, agent):
        if agent in self.agents:
            locked = agent.energy < agent.max_energy / 5
            if G["phase"] == "pred":
                P = G["cur_pred"]; pid = id(P)
                pr = list(pring.get(pid, []))
                ar = list(aring.get(agent.agent_id, []))
                # chase = consecutive most-recent steps with target == victim
                chase = []
                for rec in reversed(pr):
                    if rec[1] == agent.agent_id:
                        chase.append(rec)
                    else:
                        break
                chase.reverse()
                br = [r[2] for r in chase]
                start = chase[0] if chase else None
                steps_chase = {r[0] for r in chase}
                perc = [(r[0], r[1].get(pid)) for r in ar if pid in r[1]]
                perc_chase = [x for x in perc if x[0] in steps_chase]
                first_seen_d = perc_chase[0][1] if perc_chase else None
                last5 = any(x[0] >= G["step"] - 5 for x in perc)
                # distinct predators that targeted the victim in the last 30 steps
                tg = 0
                for q in self.predators:
                    rq = pring.get(id(q), [])
                    if any(r[1] == agent.agent_id and r[0] >= G["step"] - 30 for r in rq):
                        tg += 1
                aw90 = sum(1 for q in self.predators if not q.resting and math.hypot(q.x - agent.x, q.y - agent.y) < 90)
                aw250 = sum(1 for q in self.predators if not q.resting and math.hypot(q.x - agent.x, q.y - agent.y) < 250)
                sib250 = sum(1 for b in self.agents if b is not agent and math.hypot(b.x - agent.x, b.y - agent.y) < 250)
                last = ar[-11:]
                spd = [math.hypot(b[3] - a_[3], b[4] - a_[4]) for a_, b in zip(last, last[1:])]
                # facing the killer over the last 10 steps (bearing within +-90 deg of heading), from ring positions
                pos = {r[0]: (r[5], r[6]) for r in pr}
                fac = []
                for r in last:
                    if r[0] in pos:
                        bx, by = pos[r[0]]
                        rel = (math.atan2(by - r[4], bx - r[3]) - r[5] + math.pi) % (2 * math.pi) - math.pi
                        fac.append(abs(rel) < math.pi / 2)
                # ---- pre-chase history (cycle 3) ----
                s0 = start[0] if start else G["step"]
                recs = [r for r in pr if r[0] <= s0]
                woke = None
                for i in range(len(recs) - 1, 0, -1):
                    if recs[i][0] - recs[i - 1][0] > 1:
                        woke = recs[i][0]; break
                if woke is None and recs and len(pr) < 400 and spawn_step.get(pid) is not None \
                        and recs[0][0] - spawn_step[pid] > 1:
                    woke = recs[0][0]  # first wake after spawning (rested since spawn)
                pre30 = [r for r in pr if s0 - 30 <= r[0] < s0]
                prev_other = any(r[1] is not None and r[1] != agent.agent_id for r in pre30)
                prev_wander = sum(1 for r in pre30 if r[1] is None)
                arec = {r[0]: r for r in ar}
                a0, a10 = arec.get(s0), arec.get(s0 - 10)
                pre_v = pre_turn = None
                if a0 and a10:
                    pre_v = math.hypot(a0[3] - a10[3], a0[4] - a10[4]) / 10
                    pre_turn = abs(a0[5] - a10[5]) / 10
                pre_seen_any = any(len(arec[st_][1]) > 0 for st_ in range(s0 - 20, s0 + 1) if st_ in arec)
                pre_seen_killer = any(pid in arec[st_][1] for st_ in range(s0 - 20, s0 + 1) if st_ in arec)
                kills.append({
                    "t": round(self.time, 1), "age": round(agent.age, 1), "E": round(agent.energy, 1), "locked": locked,
                    "maxE": round(agent.max_energy), "hear": round(agent.hearing_radius, 1), "speed": round(agent.speed, 1),
                    "sprint": round(agent.sprint_speed, 1), "n": len(self.agents),
                    "chase_len": len(chase), "chase_d0": None if not start else round(start[3], 1),
                    "chase_br0": None if not start else start[2], "chase_E0": None if not start else round(start[4], 1),
                    "n_back": br.count("charge_back"), "n_close": br.count("charge_close"), "n_pivot": br.count("pivot"),
                    "seen_chase": bool(perc_chase), "seen_window": bool(perc), "first_seen_d": None if first_seen_d is None else round(first_seen_d, 1),
                    "seen_last5": last5, "targeters30": tg, "awake90": aw90, "awake250": aw250, "sib250": sib250,
                    "v10": round(sum(spd) / len(spd), 2) if spd else None,
                    "face10": round(sum(fac) / len(fac), 2) if fac else None,
                    "killer": pidx.get(pid), "killer_prev_kills": len(pkills.get(pid, [])),
                    "killer_since_prev": None if not pkills.get(pid) else round(self.time - pkills[pid][-1], 1),
                    "sib150_killer": sum(1 for b in self.agents if b is not agent and math.hypot(b.x - P.x, b.y - P.y) < 150),
                    "awake_at_chase": None if woke is None else s0 - woke,
                    "pred_age": None if spawn_step.get(pid) is None else G["step"] - spawn_step[pid],
                    "prev_other": prev_other, "prev_wander30": prev_wander, "n_pre30": len(pre30),
                    "start_hear": None if not start else bool(start[3] <= 60.0),
                    "pre_v": None if pre_v is None else round(pre_v, 2),
                    "pre_turn": None if pre_turn is None else round(pre_turn, 3),
                    "pre_seen_any": pre_seen_any, "pre_seen_killer": pre_seen_killer,
                })
                pkills.setdefault(pid, []).append(self.time)
                cause = "predator"
            else:
                cause = "old" if agent.age > agent.max_age else "starve"
            deaths.append({"t": round(self.time, 1), "cause": cause, "age": round(agent.age, 1), "locked": locked})
            aring.pop(agent.agent_id, None)
        o_kill(self, agent)
    Environment.kill_agent = kill

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
            if step % 500 == 0:
                ag = env.agents
                n = len(ag)
                def m(f):
                    return round(sum(f(a) for a in ag) / n, 2) if n else None
                series.append({"t": round(env.time), "n": n, "P": len(env.predators),
                               "Pawake": sum(1 for p in env.predators if not p.resting),
                               "hear": m(lambda a: a.hearing_radius), "hear_max": round(max((a.hearing_radius for a in ag), default=0), 1),
                               "hear90": sum(1 for a in ag if a.hearing_radius >= 90),
                               "speed": m(lambda a: a.speed), "speed15": sum(1 for a in ag if a.speed >= 15),
                               "sprint": m(lambda a: a.sprint_speed), "maxE": m(lambda a: a.max_energy),
                               "vis": m(lambda a: a.vision_radius), "cone": m(lambda a: a.cone_angle)})
            obs = [o for o in state["observations"] if o is not None]
            try:
                acts = hm.act(obs)
                actions = [(a["agent_id"], ActionRequest(**a)) for a in acts]
            except Exception as e:
                err = repr(e); break
            if state["num_agents"] == 0 or env.time > max_time:
                break
    return {"seed": seed, "score": state["score"], "time": env.time, "kills": kills, "deaths": deaths,
            "series": series, "err": err, "wall": time.time() - t0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("policy"); ap.add_argument("seeds")
    ap.add_argument("--procs", type=int, default=8); ap.add_argument("--out", default="census.json")
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
    print(f"done {a.policy} n={len(res)} mean={sum(sc)/len(sc):.1f} -> {a.out}")


if __name__ == "__main__":
    main()
