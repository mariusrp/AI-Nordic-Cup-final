#!/usr/bin/env python3
"""Understanding lane cycle 3: where do late kills come from? Summarises the pre-chase fields of kill_census.py
(cycle-3 version). Usage: python3 prechase.py census.json [...]
Origin classes of each predator kill (exclusive, first match wins):
  woke      the killer woke from rest <= 60 steps before the chase started (incl. first wake after spawning)
  switch    the killer targeted ANOTHER agent in the 30 steps before the chase (target switch)
  far       the chase started at >= 105 px (a sighting there lets FDE stop the charge)
  near      the rest: an awake wandering killer first targeted the victim inside 105 px
"""
import json, sys, statistics as st
from collections import Counter


def q(v, p):
    v = sorted(x for x in v if x is not None)
    return None if not v else v[int(p * (len(v) - 1))]


def origin(k):
    if k.get("awake_at_chase") is not None and k["awake_at_chase"] <= 60:
        return "woke"
    if k.get("prev_other"):
        return "switch"
    if k["chase_d0"] is not None and k["chase_d0"] >= 105:
        return "far"
    return "near"


for f in sys.argv[1:]:
    d = json.load(open(f)); R = d["results"]; G = len(R)
    K = [k for r in R for k in r["kills"] if "awake_at_chase" in k]
    if not K:
        print(f, "no cycle-3 fields"); continue
    n = len(K)
    sc = [r["score"] for r in R]
    print(f"\n=== {f} ({d['policy']}) games={G} mean={st.mean(sc):.0f} kills={n}")
    late = [k for k in K if k["seen_chase"] and k["first_seen_d"] is not None and k["first_seen_d"] < 90]
    print(f"late (first seen < 90 px) {100*len(late)/n:.0f}%; first-seen med {q([k['first_seen_d'] for k in K], .5)}; chase d0 med {q([k['chase_d0'] for k in K], .5)}")
    c = Counter(origin(k) for k in K)
    print("origin % (all kills): " + " ".join(f"{o} {100*c[o]/n:.0f}" for o in ("woke", "switch", "far", "near")))
    cl = Counter(origin(k) for k in late)
    print("origin % (late kills): " + " ".join(f"{o} {100*cl[o]/max(1,len(late)):.0f}" for o in ("woke", "switch", "far", "near")))
    for o in ("woke", "switch", "far", "near"):
        S = [k for k in K if origin(k) == o]
        if not S:
            continue
        print(f"  {o:6s} n={len(S):4d} d0 med {q([k['chase_d0'] for k in S], .5)} first-seen med {q([k['first_seen_d'] for k in S], .5)} "
              f"start_hear {100*sum(bool(k['start_hear']) for k in S)/len(S):.0f}% killer seen in 20 steps before chase {100*sum(k['pre_seen_killer'] for k in S)/len(S):.0f}% "
              f"any pred seen {100*sum(k['pre_seen_any'] for k in S)/len(S):.0f}% victim pre-speed med {q([k['pre_v'] for k in S], .5)} "
              f"locked {100*sum(k['locked'] for k in S)/len(S):.0f}% newborn {100*sum(k['age'] < 20 for k in S)/len(S):.0f}% "
              f"awake_at_chase med {q([k['awake_at_chase'] for k in S], .5)} pred_age med {q([k['pred_age'] for k in S], .5)}")
    aw = [k["awake_at_chase"] for k in K if k["awake_at_chase"] is not None]
    print(f"steps awake at chase start: <=20 {sum(a <= 20 for a in aw)} 21-60 {sum(20 < a <= 60 for a in aw)} 61-200 {sum(60 < a <= 200 for a in aw)} >200 {sum(a > 200 for a in aw)} unknown(>ring) {n - len(aw)}")
    pv = [k["pre_v"] for k in K if k["pre_v"] is not None]
    print(f"victim speed in the 10 steps BEFORE the chase: idle(<1) {100*sum(v < 1 for v in pv)/len(pv):.0f}% slow(1-5) {100*sum(1 <= v < 5 for v in pv)/len(pv):.0f}% walking(>=5) {100*sum(v >= 5 for v in pv)/len(pv):.0f}%")
