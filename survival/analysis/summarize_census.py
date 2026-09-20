#!/usr/bin/env python3
"""Summarize kill_census.py outputs (analysis only). Usage: python3 summarize_census.py a.json [b.json ...]
Kill classes (exclusive, first match wins):
  blind     the victim never perceived its killer during the final chase
  late      first perceived inside 90 px (the predator charges there whatever the facing)
  multi     >= 2 predators targeted the victim in its last 30 steps
  seen_far  perceived at >= 90 px by a single chaser: a RESPONSE failure (facing + backpedal/juke should survive)
"""
import json, sys, statistics as st
from collections import Counter


def q(v, p):
    v = sorted(v)
    return v[int(p * (len(v) - 1))] if v else None


def cls(k):
    if not k["seen_chase"]:
        return "blind"
    if k["first_seen_d"] is not None and k["first_seen_d"] < 90:
        return "late"
    if k["targeters30"] >= 2:
        return "multi"
    return "seen_far"


for f in sys.argv[1:]:
    d = json.load(open(f))
    R = d["results"]
    G = len(R)
    K = [k for r in R for k in r["kills"]]
    D = Counter(x["cause"] for r in R for x in r["deaths"])
    sc = [r["score"] for r in R]
    print(f"\n=== {d['policy']}  games={G} mean score={st.mean(sc):.1f} errors={sum(1 for r in R if r['err'])}")
    print(f"deaths/game: pred {D['predator']/G:.1f} starve {D['starve']/G:.1f} old {D['old']/G:.1f}")
    n = len(K)
    if not n:
        continue
    c = Counter(cls(k) for k in K)
    print("kill class %: " + " ".join(f"{a} {100*c[a]/n:.0f}" for a in ("blind", "late", "multi", "seen_far")))
    print(f"chase start branch %: " + " ".join(f"{b} {100*sum(1 for k in K if k['chase_br0']==b)/n:.0f}" for b in ("charge_back", "charge_close", "pivot")))
    cd = [k["chase_d0"] for k in K if k["chase_d0"] is not None]
    print(f"chase start d px: p25 {q(cd,.25):.0f} med {q(cd,.5):.0f} p75 {q(cd,.75):.0f}; chase len steps med {q([k['chase_len'] for k in K],.5)}")
    fs = [k["first_seen_d"] for k in K if k["first_seen_d"] is not None]
    print(f"first seen d (seen in chase): p25 {q(fs,.25):.0f} med {q(fs,.5):.0f} p75 {q(fs,.75):.0f}; seen in last 5 steps {100*sum(k['seen_last5'] for k in K)/n:.0f}%")
    print(f"locked {100*sum(k['locked'] for k in K)/n:.0f}%  newborn<20s {100*sum(k['age']<20 for k in K)/n:.0f}%  "
          f"targeters30>=2 {100*sum(k['targeters30']>=2 for k in K)/n:.0f}%  awake250>=2 {100*sum(k['awake250']>=2 for k in K)/n:.0f}%  "
          f"awake90>=2 {100*sum(k['awake90']>=2 for k in K)/n:.0f}%  alone(sib250=0) {100*sum(k['sib250']==0 for k in K)/n:.0f}%")
    v = [k["v10"] for k in K if k["v10"] is not None]
    print(f"victim speed last 10 steps: <2 {100*sum(x<2 for x in v)/len(v):.0f}%  2-10.5 {100*sum(2<=x<=10.5 for x in v)/len(v):.0f}%  >10.5 {100*sum(x>10.5 for x in v)/len(v):.0f}%")
    fc = [k["face10"] for k in K if k["face10"] is not None]
    print(f"victim faced killer (share of last 10 steps): mean {st.mean(fc):.2f}; any back-turned charge in chase {100*sum(k['n_back']>0 for k in K)/n:.0f}%")
    for a in ("blind", "late", "multi", "seen_far"):
        S = [k for k in K if cls(k) == a]
        if not S:
            continue
        print(f"  {a:8s} n={len(S):4d} locked {100*sum(k['locked'] for k in S)/len(S):3.0f}% newborn {100*sum(k['age']<20 for k in S)/len(S):3.0f}% "
              f"br0 back/close/pivot {sum(k['chase_br0']=='charge_back' for k in S)}/{sum(k['chase_br0']=='charge_close' for k in S)}/{sum(k['chase_br0']=='pivot' for k in S)} "
              f"d0 med {q([k['chase_d0'] for k in S if k['chase_d0'] is not None],.5)} pivot-steps med {q([k['n_pivot'] for k in S],.5)} "
              f"back-steps med {q([k['n_back'] for k in S],.5)} v10 med {q([k['v10'] for k in S if k['v10'] is not None],.5)} "
              f"sib250=0 {100*sum(k['sib250']==0 for k in S)/len(S):.0f}%")
    # kills by time window
    wins = [(0, 300), (300, 600), (600, 900), (900, 1200), (1200, 3000)]
    print("kills/game by window: " + " ".join(f"{lo}-{hi}:{sum(1 for k in K if lo<=k['t']<hi)/G:.1f}" for lo, hi in wins))
    # herd traits over time (mean over games alive at t)
    ts = sorted({s["t"] for r in R for s in r["series"]})
    print("t     n    hear  hear_max hear90 speed speed15 sprint maxE   vis   cone")
    for t in ts:
        S = [s for r in R for s in r["series"] if s["t"] == t and s["n"] > 0]
        if len(S) < max(3, G // 4) or t % 250:
            continue
        m = lambda key: st.mean(s[key] for s in S)
        print(f"{t:5d} {m('n'):4.1f} {m('hear'):5.1f} {m('hear_max'):6.1f} {m('hear90'):5.2f} {m('speed'):5.2f} {m('speed15'):5.2f} "
              f"{m('sprint'):5.1f} {m('maxE'):5.0f} {m('vis'):5.0f} {m('cone'):5.2f}  (games {len(S)})")
