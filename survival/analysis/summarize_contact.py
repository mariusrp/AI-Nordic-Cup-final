#!/usr/bin/env python3
"""Summarise contact_tail.py output: contact episodes (share of predator-time engaged, kills per contact-second,
re-encounter gaps, how episodes start), kills by predator spawn order (exposure vs lethality), and the end-game tail.
Usage: python3 survival/analysis/summarize_contact.py ct.json"""
import json, sys, statistics as st
from collections import defaultdict


def q(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * len(xs)))] if xs else None


def main(path):
    R = json.load(open(path))["results"]
    print(f"games={len(R)} mean score={st.mean(r['score'] for r in R):.1f} mean T={st.mean(r['time'] for r in R):.1f}")
    E = [dict(e, seed=r["seed"], T=r["time"]) for r in R for e in r["episodes"]]
    K = [dict(k, seed=r["seed"]) for r in R for k in r["kills"]]
    nk = len(K)
    # --- predator time budget ---
    life = sum(max(0.0, r["time"] - p["spawn_t"]) for r in R for p in r["pred_life"])
    eng = sum(e["dur"] for e in E)
    awake_eng = sum(e["awake"] for e in E) / 10
    perc = sum(e["perc"] for e in E) / 10
    print(f"predator-seconds alive {life:.0f}; engaged (<250 px of an agent) {eng:.0f} ({eng/life:.1%}); "
          f"awake+engaged {awake_eng:.0f}; awake+perceiving an agent {perc:.0f} ({perc/life:.1%})")
    print(f"kills {nk}: per 1000 engaged-s {1000*nk/eng:.1f}; per 1000 perceiving-s {1000*nk/perc:.1f}; "
          f"kills outside any episode {sum(1 for k in K if k['ep_age'] is None)}")
    # --- episodes ---
    durs = [e["dur"] for e in E]
    ek = [e for e in E if e["kills"] > 0]
    print(f"episodes {len(E)} ({len(E)/len(R):.1f}/game): dur median {q(durs,.5)} p75 {q(durs,.75)} p90 {q(durs,.9)} max {max(durs)}; "
          f"with >=1 kill {len(ek)} ({len(ek)/len(E):.1%})")
    for lo, hi in ((0, 10), (10, 30), (30, 60), (60, 120), (120, 1e9)):
        es = [e for e in E if lo <= e["dur"] < hi]
        if es:
            print(f"   dur {lo:>4}-{hi if hi < 1e9 else 'inf':>4}s: n={len(es):4d} share_time={sum(e['dur'] for e in es)/eng:.1%} "
                  f"kills={sum(e['kills'] for e in es):4d} ({sum(e['kills'] for e in es)/max(nk,1):.1%}) "
                  f"kills/ep={st.mean(e['kills'] for e in es):.2f} wakes/ep={st.mean(e['wakes'] for e in es):.2f}")
    for how in ("spawned", "resting", "awake"):
        es = [e for e in E if e["how"] == how]
        if es:
            print(f"   start={how:8s}: n={len(es):4d} kills={sum(e['kills'] for e in es):4d} "
                  f"({sum(e['kills'] for e in es)/max(nk,1):.1%}) median dur {q([e['dur'] for e in es],.5)} "
                  f"median d0 {q([e['d0'] for e in es],.5)}")
    # kills vs time inside the episode
    ka = [k["ep_age"] for k in K if k["ep_age"] is not None]
    print(f"kill time since episode start: p25 {q(ka,.25)} median {q(ka,.5)} p75 {q(ka,.75)} p90 {q(ka,.9)}")
    # kills by predator's multi-kill episodes
    multi = [e for e in E if e["kills"] >= 2]
    print(f"episodes with >=2 kills: {len(multi)} carry {sum(e['kills'] for e in multi)} kills ({sum(e['kills'] for e in multi)/max(nk,1):.1%})")
    # --- re-encounter ---
    gaps, first = [], []
    for r in R:
        by = defaultdict(list)
        for e in r["episodes"]:
            by[e["pred"]].append(e)
        sp = {p["pred"]: p["spawn_t"] for p in r["pred_life"]}
        for pid, es in by.items():
            es.sort(key=lambda e: e["t0"])
            first.append(es[0]["t0"] - sp.get(pid, es[0]["t0"]))
            for a, b in zip(es, es[1:]):
                gaps.append(b["t0"] - (a["t0"] + a["dur"]))
        never = [p for p in r["pred_life"] if p["pred"] not in by]
        first += [None] * 0
    print(f"re-encounter gap (end of one episode to the same predator's next): n={len(gaps)} median {q(gaps,.5)} "
          f"p25 {q(gaps,.25)} p75 {q(gaps,.75)}; spawn->first contact median {q(first,.5)} p75 {q(first,.75)}")
    # --- kills by spawn order: exposure vs lethality ---
    print("by predator spawn index: kills, engaged s, kills per 1000 engaged s, alive s")
    for lo, hi in ((0, 1), (1, 2), (2, 4), (4, 8), (8, 100)):
        kk = sum(1 for k in K if k["killer"] is not None and lo <= k["killer"] < hi)
        ee = sum(e["dur"] for e in E if lo <= e["pred"] < hi)
        al = sum(max(0.0, r["time"] - p["spawn_t"]) for r in R for p in r["pred_life"] if lo <= p["pred"] < hi)
        print(f"   #{lo}-{hi-1}: kills {kk:4d} ({kk/max(nk,1):.1%}) engaged {ee:6.0f}s ({ee/max(al,1):.1%} of alive) "
              f"k/1000eng {1000*kk/max(ee,1):5.1f} alive {al:6.0f}s")
    # --- herd structure ---
    rad = [s["rad"] for r in R for s in r["series"] if s["rad"] is not None and s["n"] >= 6]
    ea = [s["eng_awake"] for r in R for s in r["series"] if s["n"] >= 1]
    print(f"herd radius (median dist to centroid, n>=6): median {q(rad,.5)} p25 {q(rad,.25)} p75 {q(rad,.75)}; "
          f"awake engaged predators per sample mean {st.mean(ea):.2f}")
    for lo, hi in ((0, 300), (300, 600), (600, 900), (900, 1200), (1200, 3000)):
        ss = [s for r in R for s in r["series"] if lo <= s["t"] < hi and s["n"] >= 1]
        if ss:
            print(f"   t {lo}-{hi}: samples {len(ss)} n {st.mean(s['n'] for s in ss):.1f} P {st.mean(s['P'] for s in ss):.1f} "
                  f"awake-engaged {st.mean(s['eng_awake'] for s in ss):.2f} engaged {st.mean(s['eng'] for s in ss):.2f} "
                  f"share of P engaged {sum(s['eng'] for s in ss)/max(1,sum(s['P'] for s in ss)):.1%}")
    # --- tail ---
    print("TAIL (last 120 s, agent-step shares averaged over games):")
    keys = ["moving", "req_move", "turning", "tree_vis", "fruit_vis", "pred_vis", "tree_true_lt100", "tree_true_lt250",
            "fruit_true_lt100", "fruit_true_lt250", "E_mean", "locked"]
    L = [r["tail"]["last120"] for r in R if r["tail"]["last120"]]
    print("   " + " ".join(f"{k}={st.mean(x[k] for x in L):.2f}" for k in keys))
    L3 = [r["tail"]["last300"] for r in R if r["tail"]["last300"]]
    print("   last300: " + " ".join(f"{k}={st.mean(x[k] for x in L3):.2f}" for k in keys))
    ld = [r["tail"]["last_death"] for r in R if r["tail"]["last_death"]]
    print(f"   last death causes: " + ", ".join(f"{c}={sum(1 for x in ld if x['cause']==c)}" for c in ("predator", "starve", "old"))
          + f"; last agent age median {q([x['age'] for x in ld],.5)}")
    ST = [x for r in R for x in r["tail"]["starve_tail"]]
    for c in ("starve", "old"):
        xs = [x for x in ST if x["cause"] == c]
        if xs:
            print(f"   {c} deaths in the last 120 s: n={len(xs)} age median {q([x['age'] for x in xs],.5)} "
                  f"moving30 {st.mean(x['moving30'] for x in xs):.2f} fruit_vis30 {st.mean(x['fruit_vis30'] for x in xs):.2f} "
                  f"tree_vis30 {st.mean(x['tree_vis30'] for x in xs):.2f} | true nearest tree at death median {q([x['tree_true_end'] for x in xs],.5)} "
                  f"fruit median {q([x['fruit_true_end'] for x in xs],.5)}; min over last 30 s: tree {q([x['tree_true_min30'] for x in xs],.5)} "
                  f"fruit {q([x['fruit_true_min30'] for x in xs],.5)}; fruit within 100 px at some point in last 30 s: "
                  f"{sum(1 for x in xs if x['fruit_true_min30'] < 100)}/{len(xs)}; E 30 s before median {q([x['E_30s_before'] for x in xs],.5)}")
    print(f"   world at the end: trees median {q([r['tail']['trees_end'] for r in R],.5)} fruits median "
          f"{q([r['tail']['fruits_end'] for r in R],.5)} predators median {q([r['tail']['preds_end'] for r in R],.5)}")
    # births vs deaths per window
    print("births / deaths (pred, starve, old) per game by window:")
    for lo, hi in ((0, 300), (300, 600), (600, 900), (900, 1200), (1200, 3000)):
        b = sum(1 for r in R for x in r["births"] if lo <= x["t"] < hi) / len(R)
        d = {c: sum(1 for r in R for x in r["deaths"] if lo <= x["t"] < hi and x["cause"] == c) / len(R) for c in ("predator", "starve", "old")}
        live = sum(1 for r in R if r["time"] > lo)
        print(f"   t {lo}-{hi}: games alive at start {live}; births {b:.1f} pred {d['predator']:.1f} starve {d['starve']:.1f} old {d['old']:.1f}")


if __name__ == "__main__":
    main(sys.argv[1])
