#!/usr/bin/env python3
"""Summarise food_refuge.py outputs (understanding lane, cycle 5). Analysis numbers only, never a scorer.
Usage: python3 survival/analysis/summarize_food_refuge.py fr_bbmpc.json [fr_other.json]
"""
import json, math, statistics as st, sys
from collections import defaultdict

LABELS = ("target", "deferred", "flee", "chaser_block", "sleep_divert", "other")
BUCKETS = (15, 30, 60, 100, 150, 250, 400)


def pct(a, b):
    return f"{100.0 * a / b:5.1f}%" if b else "  n/a"


def food(res, name):
    agg = defaultdict(lambda: [0, 0])
    for r in res:
        for ph, eb, lab, n, ate in r["food"]:
            agg[(ph, eb, lab)][0] += n; agg[(ph, eb, lab)][1] += ate
    print(f"\n== {name}: hungry agent-steps WITH a perceived fruit, by branch (share of steps | ate within 3 s)")
    for ph in ("early", "tail120"):
        for eb in ("locked", "low", "mid"):
            tot = sum(agg[(ph, eb, l)][0] for l in LABELS)
            if not tot:
                continue
            cells = []
            for l in LABELS:
                n, ate = agg[(ph, eb, l)]
                if n:
                    cells.append(f"{l} {pct(n, tot)} | {pct(ate, n)}")
            print(f"  {ph:8s} {eb:6s} n={tot:7d}  " + "; ".join(cells))
    sv = [s for r in res for s in r["starv"]]
    if sv:
        tail = [s for s in sv if s["tail"]]
        mix = defaultdict(int)
        for s in sv:
            for k, v in s["mix"].items():
                mix[k] += v
        tot = sum(mix.values())
        saw = sum(1 for s in sv if s["fruit_steps30"] > 0)
        dom = defaultdict(int)
        for s in sv:
            if s["mix"]:
                dom[max(s["mix"].items(), key=lambda kv: kv[1])[0]] += 1
        near = [s["nearest_fd"] for s in sv if s["nearest_fd"] is not None]
        ate = [s["ate30s"] for s in sv]
        print(f"  STARVERS n={len(sv)} (in last 120 s: {len(tail)}); perceived a fruit while hungry in their last 30 s: {saw} "
              f"({pct(saw, len(sv))}); median age {st.median(s['age'] for s in sv):.1f}")
        print("   last-30-s branch mix over fruit-perceived steps: " + ", ".join(f"{k} {pct(v, tot)}" for k, v in sorted(mix.items(), key=lambda kv: -kv[1])))
        print("   dominant branch per starver: " + ", ".join(f"{k} {v}" for k, v in sorted(dom.items(), key=lambda kv: -kv[1])))
        if near:
            print(f"   nearest perceived fruit in last 30 s: median {st.median(near):.1f} px, <=15 px in {sum(1 for x in near if x <= 15)}; "
                  f"ate >= 1 fruit in last 30 s: {sum(1 for x in ate if x > 0)} of {len(ate)}")


def refuge(res, name):
    ks = [k for r in res for k in r["kills"]]
    base = [0] * (len(BUCKETS) + 1)
    for r in res:
        for i, v in enumerate(r["base_rate"]):
            base[i] += v
    btot = sum(base)
    print(f"\n== {name}: refuge reach (geodesic px through agent-free space to the nearest refuge cell); kills n={len(ks)}")
    cum = 0
    line = []
    for i, b in enumerate(BUCKETS):
        cum += base[i]
        line.append(f"<= {b}: {pct(cum, btot)}")
    print("  base rate, all agent-time: " + ", ".join(line))
    for L, lab in ((0, "at kill"), (10, "t-1 s"), (30, "t-3 s"), (50, "t-5 s"), (100, "t-10 s")):
        key = "ref0" if L == 0 else f"ref{L}"
        v = [k[key] for k in ks if k.get(key) is not None]
        if not v:
            continue
        print(f"  victim {lab:7s} n={len(v):5d} median {st.median(v):6.0f}  " +
              ", ".join(f"<= {b}: {pct(sum(1 for x in v if x <= b), len(v))}" for b in BUCKETS[:6]))
    for L in (30, 50):
        rows = [k for k in ks if k.get(f"ref{L}") is not None and k.get(f"dp{L}") is not None]
        if not rows:
            continue
        win_walk = sum(1 for k in rows if k[f"ref{L}"] <= max(0.0, (k[f"dp{L}"] - 15)) * 10 / 15)
        win_mix = sum(1 for k in rows if k[f"ref{L}"] <= max(0.0, (k[f"dp{L}"] - 15)) * (10 if k["locked"] else 20) / 15)
        dps = [k[f"dp{L}"] for k in rows]
        rest = sum(1 for k in rows if k.get(f"rest{L}"))
        print(f"  race at t-{L // 10} s (victim reaches refuge before killer closes to 15 px, straight lines): walk-only "
              f"{pct(win_walk, len(rows))}, sprint unless locked {pct(win_mix, len(rows))}; killer dist median {st.median(dps):.0f} px, "
              f"killer resting {pct(rest, len(rows))}")
    lk = sum(1 for k in ks if k["locked"])
    if ks:
        print(f"  victims sprint-locked at death: {pct(lk, len(ks))}; median victim age {st.median(k['age'] for k in ks):.1f}")


def census(res, name):
    T = sum(r["time"] for r in res)
    c = defaultdict(int)
    win = defaultdict(int)
    for r in res:
        for d in r["deaths"]:
            c[d["cause"]] += 1
        for k in r["kills"]:
            win[min(int(k["t"] // 300), 5)] += 1
    sc = [r["score"] for r in res]
    print(f"\n== {name}: n={len(res)} score mean {st.mean(sc):.1f} sd {st.pstdev(sc):.1f} median {st.median(sc):.1f} max {max(sc):.1f}; "
          f"errs {sum(1 for r in res if r['err'])}")
    print("  deaths per game: " + ", ".join(f"{k} {v / len(res):.1f}" for k, v in sorted(c.items())) +
          " | per 1000 s: " + ", ".join(f"{k} {1000 * v / T:.1f}" for k, v in sorted(c.items())))
    print("  kills per game by 300-s window: " + ", ".join(f"{300 * w}-{300 * w + 300 if w < 5 else ''}: {win[w] / len(res):.1f}" for w in range(6)))
    print(f"  eats per game {sum(r['eats'] for r in res) / len(res):.0f}; refuge px per map median {st.median(r['ref_px'] for r in res)}")


def paired(a, b):
    A = {r["seed"]: r["score"] for r in a["results"]}
    B = {r["seed"]: r["score"] for r in b["results"]}
    common = sorted(set(A) & set(B))
    d = [B[s] - A[s] for s in common]
    if len(d) > 1:
        print(f"\n== paired {b['policy']} - {a['policy']} on {len(d)} seeds: {st.mean(d):+.1f} se {st.stdev(d) / math.sqrt(len(d)):.1f} "
              f"(wins {sum(1 for x in d if x > 0)}/{len(d)})")


if __name__ == "__main__":
    runs = [json.load(open(p)) for p in sys.argv[1:]]
    for R in runs:
        name = R["policy"].split("/")[-1]
        census(R["results"], name)
        food(R["results"], name)
        refuge(R["results"], name)
    if len(runs) == 2:
        paired(runs[0], runs[1])
