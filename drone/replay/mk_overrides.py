"""overrides_v4 = overrides_v3 + crisp-calibrated demotion of the eye-unjudged clusters that rank in the top 40 of
some frame. Calibration (this session): lap/ring of the 43 eye-CONFIRMED objects vs the 287 eye-DEMOTED clutter
clusters of overrides_v3 -> a threshold keeps X% of real / kills Y% of clutter. Demotion is SOFT (support x FAC) so a
mistakenly demoted real object still sits in its class's tail instead of vanishing.
Usage: python mk_overrides.py BASE_OVERRIDES.json CLUSTERS.json IDS.json CRISP.json OUT.json [--thr 1.8] [--fac 0.3] [--rank 40]
(CLUSTERS/IDS = the fuse.py run the CRISP.json was measured on, i.e. the previous table.)"""
import json, sys, collections
BASE, CLP, IDP, CRP, OUT = sys.argv[1:6]
argv = sys.argv[6:]


def opt(n, d):
    return float(argv[argv.index(n) + 1]) if n in argv else d


THR = opt('--thr', 1.8); FAC = opt('--fac', 0.3); RANK = int(opt('--rank', 40))
CL = {c['id']: c for c in json.load(open(CLP))}
CR = {int(k): v for k, v in json.load(open(CRP)).items()}
IDS = json.load(open(IDP))
top = collections.Counter()
for f, ids in IDS.items():
    for cid in ids[:RANK]: top[cid] += 1
ov = json.load(open(BASE))
n = 0
for cid, cr in CR.items():
    c = CL[cid]
    if c.get('judged') or cid not in top or c['support'] >= 0.4: continue
    if cr['lap'] / max(cr['ring'], 1e-6) >= THR: continue
    ov.append(dict(t=c['t_star'], x=c['x_star'], support=round(c['support'] * FAC, 4),
                   note=f"v4 crisp auto-demote: lap/ring {cr['lap'] / max(cr['ring'], 1e-6):.2f} < {THR} on {cr['n']} native-L2 views "
                        f"(calibrated on the 43 eye-confirmed / 287 eye-demoted clusters of overrides_v3); support x{FAC}"))
    n += 1
json.dump(ov, open(OUT, 'w'), indent=0)
print('overrides_v4', len(ov), 'entries, of which', n, 'new crisp demotions (thr', THR, 'fac', FAC, ')')
