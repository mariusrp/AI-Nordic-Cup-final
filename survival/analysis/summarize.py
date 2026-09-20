#!/usr/bin/env python3
"""Summarize understand.py output (analysis only). Usage: python3 survival/analysis/summarize.py a.json [b.json ...]"""
import json, sys, statistics as st, collections


def summ(path):
    R = json.load(open(path))["results"]
    n = len(R)
    print(f"\n=== {path}  n={n}")
    sc = [r["score"] for r in R]; T = [r["time"] for r in R]
    print(f"score {st.mean(sc):.1f} (se {st.stdev(sc)/n**0.5:.1f})  time {st.mean(T):.1f}  fruit_bonus {st.mean(r['fruit_bonus'] for r in R):.1f}"
          f"  eaten_penalty {st.mean(r['eaten_pen'] for r in R):.1f}  time quantiles {[round(x) for x in st.quantiles(T, n=4)]} max {max(T):.0f}")
    # deaths by cause and time bucket
    B = [0, 150, 300, 450, 600, 900, 1200, 1800, 3001]
    tab = collections.defaultdict(lambda: collections.Counter())
    for r in R:
        for d in r["deaths"]:
            b = max(x for x in B if x <= d["t"])
            tab[b][d["cause"]] += 1
    print("deaths per game by window (pred/starve/old) and births per game:")
    for b in B[:-1]:
        nb = sum(1 for r in R for x in r["births"] if b <= x["t"] < B[B.index(b) + 1])
        alive = sum(1 for r in R if r["time"] > b)
        c = tab[b]
        print(f"  {b:5d}-{B[B.index(b)+1]:<5d} games_alive={alive:3d}  pred {c['predator']/n:5.1f} starve {c['starve']/n:5.1f} old {c['old']/n:5.1f}  births {nb/n:5.1f}")
    dd = collections.Counter(d["cause"] for r in R for d in r["deaths"])
    print(f"total deaths/game pred {dd['predator']/n:.1f} starve {dd['starve']/n:.1f} old {dd['old']/n:.1f}; births/game {st.mean(len(r['births']) for r in R):.1f}")
    pk = [d for r in R for d in r["deaths"] if d["cause"] == "predator"]
    if pk:
        print(f"predator kills: victim energy mean {st.mean(d['E'] for d in pk):.0f}, sprint-locked {sum(d['locked'] for d in pk)/len(pk):.0%}, "
              f"victim age<20s {sum(d['age']<20 for d in pk)/len(pk):.0%}, age>max_age {sum(d['age']>d['max_age'] for d in pk)/len(pk):.0%}")
    sv = [d for r in R for d in r["deaths"] if d["cause"] == "starve"]
    if sv:
        print(f"starve deaths: age quantiles {[round(x) for x in st.quantiles([d['age'] for d in sv], n=4)]}")
    od = [d for r in R for d in r["deaths"] if d["cause"] == "old"]
    if od:
        print(f"old deaths: age-max_age quantiles {[round(x) for x in st.quantiles([d['age']-d['max_age'] for d in od], n=4)]}, age quantiles {[round(x) for x in st.quantiles([d['age'] for d in od], n=4)]}")
    # extinction
    last = collections.Counter(); last3 = collections.Counter()
    for r in R:
        ds = r["deaths"]
        if ds:
            last[ds[-1]["cause"]] += 1
            for d in ds[-3:]:
                last3[d["cause"]] += 1
    print(f"LAST death cause: {dict(last)}; last-3 deaths causes: {dict(last3)}")
    # population before the end
    pre = collections.defaultdict(list)
    for r in R:
        S = r["series"]
        for lag in (0, 50, 100, 200, 300):
            t = r["time"] - lag
            row = max((x for x in S if x["t"] <= t), key=lambda x: x["t"], default=None)
            if row:
                pre[lag].append(row)
    for lag in (300, 200, 100, 50, 0):
        rows = pre[lag]
        if rows:
            print(f"  {lag:3d}s before end: agents {st.mean(x['n'] for x in rows):4.1f}  preds {st.mean(x['P'] for x in rows):4.1f} (awake {st.mean(x['Pawake'] for x in rows):4.1f})"
                  f"  trees {st.mean(x['trees'] for x in rows):4.1f} fruiting {st.mean(x['ftrees'] for x in rows):4.1f} fruits {st.mean(x['fruits'] for x in rows):4.1f}"
                  f"  E_mean {st.mean(x['Emean'] for x in rows):5.0f} old {st.mean(x['old'] for x in rows):3.1f} locked {st.mean(x['locked'] for x in rows):3.1f}")
    # births: last birth before extinction
    gap = [r["time"] - (r["births"][-1]["t"] if r["births"] else 0) for r in R]
    print(f"time from LAST birth to extinction: quantiles {[round(x) for x in st.quantiles(gap, n=4)]}")
    # world over time (games still running)
    print("world over time (mean over running games): t agents preds awake trees fruiting fruits ripe Emean")
    for t in (50, 150, 300, 450, 600, 800, 1000, 1200, 1500):
        rows = [x for r in R for x in r["series"] if x["t"] == t]
        if rows:
            print(f"  t={t:5d} (n={len(rows):2d}) {st.mean(x['n'] for x in rows):5.1f} {st.mean(x['P'] for x in rows):5.1f} {st.mean(x['Pawake'] for x in rows):5.1f}"
                  f" {st.mean(x['trees'] for x in rows):5.1f} {st.mean(x['ftrees'] for x in rows):5.1f} {st.mean(x['fruits'] for x in rows):5.1f} {st.mean(x['ripe'] for x in rows):5.1f} {st.mean(x['Emean'] for x in rows):5.0f}")
    ea = [e for r in R for e in r["eaten"]]
    if ea:
        print(f"fruit: spawned/game {st.mean(r['fruit_spawned'] for r in R):.0f}, eaten/game {len(ea)/n:.0f}, rotted/game {st.mean(r['rotted'] for r in R):.0f}; "
              f"eaten energy mean {st.mean(e[2] for e in ea):.1f} (full=60: {sum(e[2]>=59.9 for e in ea)/len(ea):.0%}), eaten age(s) quantiles {[round(x,1) for x in st.quantiles([e[1] for e in ea], n=4)]}")


for p in sys.argv[1:]:
    summ(p)
