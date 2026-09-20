#!/usr/bin/env python3
"""Paired confirmation of survival candidates on many FRESH seeds (never used for tuning; holdout 5000-5031 untouched).

  python3 survival/confirm.py --pod cpu2 --seeds 2000-2127 \
      --baseline v1=main:survival/policies/v1.py \
      --cand food_r1_1=survival-food-r1-1:survival/policies/food_r1_1.py ...

Every policy runs on the SAME seeds with main's frozen survival/evaluate.py (copied from main, never from the
candidate branch). Verdict per candidate uses the PAIRED per-seed difference vs the baseline:
  CLEAR WIN  if mean_diff > 2 * se_diff   (only these may be validated on the portal)
  NO CLEAR WIN otherwise.
Reports: /home/claude/handoff/confirm/<name>.json and a summary table on stdout. Runs policies one after another,
each using all pod cores; progress is polled (never blocks > 60 s per poll).
"""
import argparse, json, math, os, statistics, subprocess, sys, time

REPO = "/home/claude/nac"
OUT = "/home/claude/handoff/confirm"
os.makedirs(OUT, exist_ok=True)


def sh(cmd, env=None, timeout=600):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout, env={**os.environ, **(env or {})})
    return r.stdout + r.stderr


def parse_spec(s):
    name, rest = s.split("=", 1)
    branch, path = rest.split(":", 1)
    return name, branch, path


def seeds_from(s):
    if "-" in s and "," not in s:
        a, b = map(int, s.split("-"))
        return list(range(a, b + 1))
    return [int(x) for x in s.split(",")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pod", default="cpu2")
    ap.add_argument("--seeds", default="2000-2127")
    ap.add_argument("--procs", type=int, default=32)
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--cand", action="append", default=[])
    ap.add_argument("--tag", default=time.strftime("%H%M"))
    a = ap.parse_args()
    seeds = seeds_from(a.seeds)
    if any(5000 <= s < 5032 for s in seeds):
        sys.exit("refusing: seeds overlap the hidden holdout (5000-5031)")
    specs = [parse_spec(a.baseline)] + [parse_spec(c) for c in a.cand]
    env = {"POD": a.pod}
    stage = f"/tmp/confirm_{a.tag}"
    sh(f"rm -rf {stage} && mkdir -p {stage}")
    for name, branch, path in specs:
        d = f"{stage}/{name}"
        sh(f"mkdir -p {d} && git -C {REPO} archive {branch} survival | tar -x -C {d}")
        sh(f"git -C {REPO} show main:survival/evaluate.py > {d}/survival/evaluate.py")  # frozen scorer from main
        if not os.path.exists(f"{d}/{path}"):
            sys.exit(f"missing {path} in {branch}")
    print(sh(f"python3 {REPO}/infra/pod.py push {stage} /workspace", env))
    seedarg = ",".join(map(str, seeds))
    base = f"/workspace/{os.path.basename(stage)}"
    script = " ; ".join(
        f"cd {base}/{n} && UPSTREAM=/workspace/upstream python3 survival/evaluate.py {p} --seeds {seedarg} --procs {a.procs} --json {base}/{n}.json > {base}/{n}.log 2>&1"
        for n, _, p in specs)
    print(sh(f"python3 {REPO}/infra/pod.py bg confirm_{a.tag} \"{script}\"", env))
    res = {}
    while len(res) < len(specs):
        time.sleep(60)
        for n, _, _ in specs:
            if n in res:
                continue
            loc = f"{stage}/{n}.result.json"  # pod.py get: exec output is capped at 20 KB
            sh(f"python3 {REPO}/infra/pod.py get {base}/{n}.json {loc}", env)
            got = open(loc).read() if os.path.exists(loc) else ""
            try:
                j = json.loads(got[got.index("{"):])
                res[n] = {r["seed"]: r["score"] for r in j["results"]}
                print(f"{time.strftime('%H:%M')} done {n}: mean {statistics.mean(res[n].values()):.1f}", flush=True)
            except Exception:
                pass
    bname = specs[0][0]
    b = res[bname]
    rows = []
    for n, br, p in specs:
        s = res[n]
        common = [k for k in seeds if k in s and k in b]
        vals = [s[k] for k in common]
        mean = statistics.mean(vals); se = statistics.stdev(vals) / math.sqrt(len(vals))
        rep = {"name": n, "branch": br, "policy": p, "seeds": f"{seeds[0]}-{seeds[-1]}", "n": len(common), "mean": mean, "se": se}
        if n != bname:
            d = [s[k] - b[k] for k in common]
            md = statistics.mean(d); sd = statistics.stdev(d) / math.sqrt(len(d))
            rep.update({"baseline": bname, "mean_diff": md, "se_diff": sd, "z": md / sd if sd else 0.0,
                        "verdict": "CLEAR WIN" if md > 2 * sd else "NO CLEAR WIN"})
        json.dump(rep, open(f"{OUT}/{n}.json", "w"), indent=1)
        rows.append(rep)
    print(f"\n{'policy':22s} {'mean':>8s} {'se':>6s} {'diff':>8s} {'se_d':>6s} {'z':>5s}  verdict   (n={len(seeds)} fresh seeds)")
    for r in rows:
        print(f"{r['name']:22s} {r['mean']:8.1f} {r['se']:6.1f} {r.get('mean_diff', 0):8.1f} {r.get('se_diff', 0):6.1f} {r.get('z', 0):5.2f}  {r.get('verdict', 'baseline')}")


if __name__ == "__main__":
    main()
