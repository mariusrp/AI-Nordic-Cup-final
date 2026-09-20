"""Build answer-stream variants from route_probe.py outputs (drone understanding lane, cycle 4; offline, no model calls).

For each recording prefix:
  P_FB      production + shadow-tracker floor band (P_N06 extras x 0.0029, cycle 3's package part 1)
  P_FBH     P_FB + floor hedges (condor->jet_plane, medium_plane->small_plane copies x 0.0029)
  P_r5B     production + FL-r5-B's STATELESS floor band as written in drone-fast-r5-2 drone/floor_band.py:
            this frame's post-verifier detections not covered by a same-class reported box (IoU >= .5), at
            min(conf x 0.01, 0.99 x lowest same-class conf in THIS frame), or min(conf x 0.01, 1e-4) if the class has none
  P_r5Bfix  the same stateless band at conf x 0.0029 (strictly below the 0.003 output floor)
  R_FB, R_FBH, R_H, R_FBU (routed + union-of-both-detectors shadow band), R_FBUH, R_r5Bfix: the same on the routed stack
  P_FBH3, R_FBH3  TIERED: hedge copies at conf x K*K, one tier below every floor-band box (x K), so hedges never
            outrank a floor-band box of the same class (the floor band and hedges compete inside the tail otherwise)
Usage: python route_variants.py ROUTE_DIR OUT_DIR rec [rec ...]
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


def hedge(st, k=K):
    o = {}
    for f, r in st.items():
        ann = list(r['ann'])
        for s_, d_ in (('condor', 'jet_plane'), ('medium_plane', 'small_plane')):
            for c, b, s in r['ann']:
                if c == s_ and not covered(d_, b, ann):
                    ann.append([d_, b, round(s * k, 12)])
        o[f] = dict(frame=f, level=r['level'], ann=ann)
    return o


def stateless(base, dets, key, fixed):
    o = {}
    for f, r in base.items():
        ann = list(r['ann']); low = {}
        for c, b, s in r['ann']:
            low[c] = min(low.get(c, 9), s)
        for c, b, s in dets.get(f, {}).get(key, []):
            if covered(c, b, r['ann']):
                continue
            conf = s * K if fixed else (min(s * 0.01, 0.99 * low[c]) if c in low else min(s * 0.01, 1e-4))
            ann.append([c, b, round(conf, 10)])
        ann.sort(key=lambda x: -x[2])
        o[f] = dict(frame=f, level=r['level'], ann=ann[:100])
    return o


def wr(rec, name, st):
    with open(os.path.join(out, f'{rec}_{name}.jsonl'), 'w') as w:
        for f in sorted(st):
            w.write(json.dumps(st[f]) + '\n')


for rec in sys.argv[3:]:
    S = {k: rd(os.path.join(src, f'{rec}_{k}.jsonl')) for k in ('P_base', 'P_N06', 'P_DEM', 'R_base', 'R_N06', 'RL_base', 'U_N06')}
    D = rd(os.path.join(src, f'{rec}_dets.jsonl'))
    V = dict(P_base=S['P_base'], P_DEM=S['P_DEM'], R_base=S['R_base'], RL_base=S['RL_base'])
    V['P_FB'] = band(S['P_base'], S['P_N06']); V['P_FBH'] = hedge(V['P_FB']); V['P_H'] = hedge(S['P_base'])
    V['P_r5B'] = stateless(S['P_base'], D, 'P', False); V['P_r5Bfix'] = stateless(S['P_base'], D, 'P', True)
    V['R_FB'] = band(S['R_base'], S['R_N06']); V['R_FBH'] = hedge(V['R_FB']); V['R_H'] = hedge(S['R_base'])
    V['R_FBU'] = band(S['R_base'], S['U_N06']); V['R_FBUH'] = hedge(V['R_FBU'])
    V['R_r5Bfix'] = stateless(S['R_base'], D, 'R', True)
    # tiered: hedge copies one tier BELOW the floor band (x K*K <= 8.41e-6 < K x 0.003, the lowest floor-band box)
    V['P_FBH3'] = hedge(V['P_FB'], K * K); V['R_FBH3'] = hedge(V['R_FB'], K * K)
    for k, st in V.items():
        wr(rec, k, st)
    mx = max(len(r['ann']) for r in V['R_FBUH'].values())
    print(rec, 'variants', len(V), 'max boxes/frame R_FBUH', mx)
