#!/usr/bin/env python3
"""Paired comparison of two survival/evaluate.py --json outputs on the same seeds.
Usage: paired.py BASE.json CAND.json [NAME]
Prints mean/se of the per-seed difference, win rate, worst-20% averages and the lower confidence bound
(mean - 2 se), plus early wipe-outs (games < 600 s). Missing seeds are ignored."""
import json
import math
import statistics
import sys


def load(p):
    d = json.load(open(p))
    rows = d if isinstance(d, list) else d.get("results") or d.get("res") or []
    return {r["seed"]: r for r in rows}


def summ(rows):
    sc = sorted(r["score"] for r in rows)
    k = max(1, len(sc) // 5)
    return dict(mean=statistics.mean(sc), sd=statistics.stdev(sc) if len(sc) > 1 else 0.0, min=sc[0],
                worst20=statistics.mean(sc[:k]), early=sum(1 for r in rows if r["time"] < 600),
                deaths={c: sum(r["deaths"][c] for r in rows) for c in ("predator", "starve", "old")})


def main():
    a, b = load(sys.argv[1]), load(sys.argv[2])
    name = sys.argv[3] if len(sys.argv) > 3 else "cand"
    seeds = sorted(set(a) & set(b))
    if not seeds:
        sys.exit("no common seeds")
    diffs = [b[s]["score"] - a[s]["score"] for s in seeds]
    m = statistics.mean(diffs)
    se = statistics.stdev(diffs) / math.sqrt(len(diffs)) if len(diffs) > 1 else 0.0
    wins = sum(1 for d in diffs if d > 0)
    sa, sb = summ([a[s] for s in seeds]), summ([b[s] for s in seeds])
    verdict = "CLEAR WIN" if m > 2 * se and m > 0 else ("CLEAR LOSS" if m < -2 * se else "NOISE")
    print(f"{name}: n={len(seeds)} paired diff {m:+.1f} se {se:.1f} z {m / se if se else 0:.2f} wins {wins}/{len(seeds)} -> {verdict}")
    for lab, s in (("base", sa), (name, sb)):
        print(f"  {lab:10s} mean {s['mean']:.1f} sd {s['sd']:.0f} min {s['min']:.0f} worst20 {s['worst20']:.1f} "
              f"early<600s {s['early']} deaths pred/starve/old {s['deaths']['predator']}/{s['deaths']['starve']}/{s['deaths']['old']}")
    print(f"  LCB (mean-2se) of diff: {m - 2 * se:+.1f}")
    json.dump({"name": name, "n": len(seeds), "diff": m, "se": se, "wins": wins, "verdict": verdict, "base": sa, "cand": sb},
              open(sys.argv[2].replace(".json", "_paired.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
