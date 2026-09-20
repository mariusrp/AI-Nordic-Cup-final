#!/usr/bin/env python3
"""Understanding lane, cycle 5: can an agent detect its own (hidden) max_age from what it observes? (analysis only)
From one reported state to the next, energy changes by exactly
  -(0.05 * min(d, speed) + 0.5 * max(0, d - speed))   d = requested move capped at sprint_speed, and at speed if energy < max/5
  - min(pi, |turn|) / (2 pi)  - 100 if the spawn succeeded  - 0.1 (base drain, 1.0 in every biome)
  - 0.01 * age_next if age_next > max_age  + fruit eaten (capped at max_energy).
The policy knows everything but the last two terms. This script runs bb_mpc and, for every agent-step without a birth by
that agent, computes the residual r = e_next - (e_now - known costs) and flags 'elder' when r is within 0.05 of
-0.01 * age_next (only from age 55 on: max_age is U(60,120)). It reports the flag's precision and recall against the true max_age, and the detection delay.
Usage: UPSTREAM=... python3 survival/analysis/elder_detect.py <policy.py> <seed> [max_time]
"""
import math, os, sys, io, contextlib, importlib.util
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
UP = os.environ.get("UPSTREAM", "/home/claude/Nordic-AI-Cup-2026") + "/survival-simulator"
sys.path.insert(0, UP)


def main():
    policy, seed = sys.argv[1], int(sys.argv[2])
    if 5000 <= seed < 5032:
        sys.exit("refusing holdout seeds 5000-5031")
    max_time = float(sys.argv[3]) if len(sys.argv) > 3 else 300.0
    from src.core import SimulationCore
    from src.utils.DTOs import ActionRequest
    spec = importlib.util.spec_from_file_location("pol", policy)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    tp = fp = fn = tn = eat_masked = 0
    first_flag, first_true = {}, {}
    with contextlib.redirect_stdout(io.StringIO()):
        sim = SimulationCore(seed=seed)
        env = sim.env
        hm = mod.Hivemind()
        actions, prev = [], {}
        while True:
            state = sim.step(actions)
            obs = [o for o in state["observations"] if o is not None]
            cur = {o["agent_id"]: o for o in obs}
            for aid, (o0, act) in prev.items():
                o1 = cur.get(aid)
                ag = env.agents_dict.get(aid)
                if o1 is None or ag is None:
                    continue
                e0, me, sp, ss = o0["energy"], o0["max_energy"], o0["speed"], o0["sprint_speed"]
                d = min(max(0.0, act["move_distance"]), ss)
                if e0 < me / 5 and d > sp:
                    d = sp
                cost = 0.05 * min(d, sp) + 0.5 * max(0.0, d - sp) + min(math.pi, abs(act["turn_angle"])) / (2 * math.pi) + 0.1
                e_after_move = e0 - (0.05 * min(d, sp) + 0.5 * max(0.0, d - sp)) - min(math.pi, abs(act["turn_angle"])) / (2 * math.pi)
                if act["spawn_agent"] and e_after_move > 100:
                    continue  # birth step: skip (the child's id tells the policy anyway)
                r = o1["energy"] - (e0 - cost)
                a1 = o1["age"]
                truth = a1 > ag.max_age
                flag = a1 >= 55.0 and abs(r + 0.01 * a1) < 0.05   # max_age >= 60, so aging is >= 0.6 per step when on
                if r > 0.5:
                    eat_masked += 1
                    continue
                if truth:
                    first_true.setdefault(aid, a1)
                if flag:
                    first_flag.setdefault(aid, a1)
                tp += flag and truth; fp += flag and not truth; fn += (not flag) and truth; tn += (not flag) and (not truth)
            if state["num_agents"] == 0 or env.time > max_time:
                break
            acts = hm.act(obs)
            actions = [(a["agent_id"], ActionRequest(**a)) for a in acts]
            prev = {a["agent_id"]: (cur[a["agent_id"]], a) for a in acts if a["agent_id"] in cur}
    delays = [first_flag[k] - first_true[k] for k in first_true if k in first_flag]
    print(f"seed {seed}: agent-steps scored {tp + fp + fn + tn}, eat steps skipped {eat_masked}")
    print(f"elder flag: precision {tp / max(tp + fp, 1):.4f}, recall {tp / max(tp + fn, 1):.4f} (tp {tp}, fp {fp}, fn {fn})")
    if delays:
        print(f"detection delay after max_age: max {max(delays):.2f} s, mean {sum(delays) / len(delays):.2f} s over {len(delays)} agents")


if __name__ == "__main__":
    main()
