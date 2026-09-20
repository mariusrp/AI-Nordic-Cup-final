#!/usr/bin/env python3
"""Summarise fruit_economy.py output (understanding lane, cycle 5). Analysis numbers only.
Usage: python3 survival/analysis/summarize_fruit.py fe_bbmpc.json
"""
import json, statistics as st, sys
from collections import defaultdict

R = json.load(open(sys.argv[1]))
res = R["results"]
print(f"{R['policy']}: n={len(res)} mean score {st.mean(r['score'] for r in res):.1f}")
W = defaultdict(lambda: defaultdict(float))
for r in res:
    for k, v in r["windows"].items():
        for kk, vv in v.items():
            W[int(k)][kk] += vv
print("window   spawned  eaten  rotted  eaten%  meanE_eaten  ripe(60)%  <40%   (totals over games)")
for k in sorted(W):
    w = W[k]
    sp, ea, ro = w["spawned"], w["eaten"], w["rotted"]
    print(f"{300 * k:4d}-{300 * k + 300:<4d} {sp:8.0f} {ea:6.0f} {ro:7.0f}  {100 * ea / max(sp, 1):5.1f}  {w['E_eaten'] / max(ea, 1):9.1f}  "
          f"{100 * w['ripe'] / max(ea, 1):8.1f}  {100 * w['unripe_lt40'] / max(ea, 1):5.1f}")
tot = defaultdict(float)
for w in W.values():
    for kk, vv in w.items():
        tot[kk] += vv
print(f"ALL: spawned {tot['spawned']:.0f}, eaten {tot['eaten']:.0f} ({100 * tot['eaten'] / tot['spawned']:.1f}%), rotted {tot['rotted']:.0f} "
      f"({100 * tot['rotted'] / tot['spawned']:.1f}%), mean energy eaten {tot['E_eaten'] / tot['eaten']:.1f}, ripe {100 * tot['ripe'] / tot['eaten']:.1f}%")
eats = [e for r in res for e in r["eats"]]
# e = (t, fruit_E, aid, age, energy_after, max_e, n)
before = [e[4] - e[1] for e in eats]
buck = defaultdict(list)
for e in eats:
    b = e[4] - e[1]
    k = "locked(<max/5)" if b < e[5] / 5 else ("<0.5max" if b < 0.5 * e[5] else ">=0.5max")
    buck[k].append(e[1])
for k, v in buck.items():
    print(f"  eater energy {k:15s}: {len(v):6d} eats ({100 * len(v) / len(eats):4.1f}%), mean fruit energy {st.mean(v):.1f}")
full = sum(1 for e in eats if e[4] >= e[5] - 0.01)
print(f"  eats that hit the max-energy cap: {100 * full / len(eats):.1f}%")
yb = defaultdict(list)
for e in eats:
    yb["newborn<15s" if e[3] < 15 else ("young<40s" if e[3] < 40 else "adult")].append(e[1])
for k, v in yb.items():
    print(f"  eater age {k:12s}: {len(v):6d} eats, mean fruit energy {st.mean(v):.1f}")
if eats and len(eats[0]) > 7:
    old = [e for e in eats if e[3] > e[7]]
    E_all = sum(e[1] for e in eats); E_old = sum(e[1] for e in old)
    print(f"  eats by agents PAST their hidden max_age (aging drain on): {len(old)} ({100 * len(old) / len(eats):.1f}% of eats, "
          f"{100 * E_old / E_all:.1f}% of fruit energy); their median age {st.median(e[3] for e in old):.1f}, "
          f"median energy after eating {st.median(e[4] for e in old):.0f}")
    near = [e for e in eats if e[7] - 10 < e[3] <= e[7]]
    print(f"  eats in the last 10 s before max_age: {len(near)} ({100 * len(near) / len(eats):.1f}%)")
B = [b for r in res for b in r["births"]]
if B and len(B[0]) > 6:
    ob = [b for b in B if b[1] > b[6]]
    print(f"  births by parents past max_age: {len(ob)} ({100 * len(ob) / len(B):.1f}%)")
fates = defaultdict(int)
for b in B:
    fates[b[4]] += 1
print(f"births {len(B)} ({len(B) / len(res):.1f}/game); newborn fates: " + ", ".join(f"{k} {v} ({100 * v / len(B):.0f}%)" for k, v in fates.items()))
ages = defaultdict(list)
for b in B:
    if b[4] is not None:
        ages[b[4]].append(b[5])
for k, v in ages.items():
    print(f"  {k}: median age at death {st.median(v):.1f} s")
low = [b for b in B if b[2] - 100 < b[3] / 5]
print(f"  parents left sprint-locked by the birth (energy-100 < max/5): {len(low)} ({100 * len(low) / max(len(B), 1):.0f}%)")
