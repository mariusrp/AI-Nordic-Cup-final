"""Answer-stream variants for cycle 5 (drone understanding lane; offline, no model calls) from route_probe5.py outputs.

  R_FB      routed + shadow floor band (R_N06 extras at x 0.0029)                      = cycle 4's package part 1
  R_FBH3    R_FB + tiered floor hedges (condor->jet x K, medium->small x K*K)            = cycle 4's full package
  R_B3      routed + FL-r5-B B3 per-class birth threshold (mine_roller/ta-ta/tank .12)
  R_B3FB    R_B3 + the same shadow floor band (extras not covered by an R_B3 box of the same class)
  R_B3FBH3  R_B3FB + tiered hedges
  R_FBLL    routed + floor band from a shadow that also takes r11's large_launcher (RL_N06), open question 5
  P_B3      production (v2-only) + B3, against P_base: cross-check of the fast lane's own B3 numbers
Usage: python variants5.py SRC_DIR OUT_DIR rec [rec ...]
"""
import sys, os, json
K = 0.0029
src, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1])); i = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0


def rd(p):
    return {r['frame']: r for r in (json.loads(l) for l in open(p)) if r['frame'] <= 150}


def covered(c, b, ann):
    return any(c == c2 and iou(b, b2) >= .5 for c2, b2, _ in ann)


def band(base, shadow):
    o = {}
    for f, r in base.items():
        ann = list(r['ann'])
        for c, b, s in shadow.get(f, dict(ann=[]))['ann']:
            if not covered(c, b, r['ann']):
                ann.append([c, b, round(s * K, 10)])
        o[f] = dict(frame=f, level=r['level'], ann=ann)
    return o


def hedge3(st):
    o = {}
    for f, r in st.items():
        ann = list(r['ann'])
        for s_, d_, k in (('condor', 'jet_plane', K), ('medium_plane', 'small_plane', K * K)):
            for c, b, s in r['ann']:
                if c == s_ and not covered(d_, b, ann):
                    ann.append([d_, b, round(s * k, 12)])
        o[f] = dict(frame=f, level=r['level'], ann=ann)
    return o


for rec in sys.argv[3:]:
    S = {k: rd(os.path.join(src, f'{rec}_{k}.jsonl')) for k in ('P_base', 'R_base', 'R_N06', 'RL_N06', 'R_B3', 'P_B3')}
    V = dict(P_base=S['P_base'], P_B3=S['P_B3'], R_base=S['R_base'], R_B3=S['R_B3'])
    V['R_FB'] = band(S['R_base'], S['R_N06']); V['R_FBH3'] = hedge3(V['R_FB'])
    V['R_B3FB'] = band(S['R_B3'], S['R_N06']); V['R_B3FBH3'] = hedge3(V['R_B3FB'])
    V['R_FBLL'] = band(S['R_base'], S['RL_N06'])
    for k, st in V.items():
        with open(os.path.join(out, f'{rec}_{k}.jsonl'), 'w') as w:
            for f in sorted(st):
                w.write(json.dumps(st[f]) + '\n')
    print(rec, 'max boxes/frame R_B3FBH3', max(len(r['ann']) for r in V['R_B3FBH3'].values()))
