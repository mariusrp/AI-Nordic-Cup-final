#!/usr/bin/env python3
"""Paired analysis over survival/harvest.py JSONL files, restricted to the seeds EVERY file has.

The first file is the baseline; every other file is reported as a paired diff against it. Also reports the
sd and p25 of a 3-GAME AVERAGE (bootstrap), which is what the final actually scores.
Usage: python3 survival/harvest_stats.py <base.jsonl> <arm.jsonl> ...
"""
import json, math, random, statistics, sys

def load(p):
    d = {}
    for line in open(p):
        line = line.strip()
        if not line: continue
        try: r = json.loads(line)
        except Exception: continue
        d[r["seed"]] = r
    return d

def boot3(sc, n=100000, seed=7):
    rnd = random.Random(seed); k = len(sc)
    a = [(sc[rnd.randrange(k)] + sc[rnd.randrange(k)] + sc[rnd.randrange(k)]) / 3.0 for _ in range(n)]
    a.sort(); return a

def q(a, f): return a[min(len(a) - 1, int(f * len(a)))]

paths = sys.argv[1:]
ds = [(p.split("/")[-1].replace(".jsonl", "").replace("base_", ""), load(p)) for p in paths]
common = set(ds[0][1])
for _, d in ds[1:]: common &= set(d)
seeds = sorted(common)
print(f"common seeds n={len(seeds)}: {seeds[0] if seeds else '-'}..{seeds[-1] if seeds else '-'}")
if not seeds: sys.exit()
base = ds[0][1]
print(f"{'policy':<12}{'n':>4}{'mean':>8}{'se':>7}{'sd':>7}{'p25':>8}{'p50':>8}{'min':>8}{'max':>8}"
      f"{'>=1500':>8}{'>=2000':>8}{'diff':>8}{'se_d':>7}{'z':>6}{'win':>8}{'pred/starve/old':>18}{'a3sd':>7}{'a3p25':>8}")
for name, d in ds:
    sc = [d[s]["score"] for s in seeds]; n = len(sc); ss = sorted(sc)
    se = statistics.stdev(sc) / math.sqrt(n) if n > 1 else 0.0
    diffs = [d[s]["score"] - base[s]["score"] for s in seeds]
    dm = statistics.mean(diffs); dse = statistics.stdev(diffs) / math.sqrt(n) if n > 1 else 0.0
    de = {k: sum(d[s]["deaths"][k] for s in seeds) for k in ("predator", "starve", "old")}
    a3 = boot3(sc)
    print(f"{name:<12}{n:>4}{statistics.mean(sc):>8.1f}{se:>7.1f}{statistics.stdev(sc) if n>1 else 0:>7.0f}"
          f"{q(ss,0.25):>8.1f}{q(ss,0.5):>8.1f}{ss[0]:>8.1f}{ss[-1]:>8.1f}"
          f"{sum(1 for x in sc if x>=1500)/n*100:>7.1f}%{sum(1 for x in sc if x>=2000)/n*100:>7.1f}%"
          f"{dm:>8.1f}{dse:>7.1f}{(dm/dse if dse else 0):>6.2f}{sum(1 for x in diffs if x>0):>5}/{n:<2}"
          f"{de['predator']:>6}/{de['starve']}/{de['old']:<7}{statistics.stdev(a3):>7.0f}{q(a3,0.25):>8.0f}")
