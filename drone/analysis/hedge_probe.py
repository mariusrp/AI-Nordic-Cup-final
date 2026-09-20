"""Class hedges measured on recorded answers (drone understanding lane, cycle 2).

Raw-detector probe (det_probe.py) shows systematic confusions at L1 on the validation city: v2 and r11 call the
swept-wing jets #0/#56 'condor' in ~95% of views, v2 calls the red-nosed small_planes 'medium_plane' in ~90%.
A hedge appends a copy of every box of class A as class B (same box, conf x k). By score_semantics.py T2/T4 a copy can
only cost within class B's own ranking. Scored with the unmodified upstream scorer on a valcity scene (frames <= 150).
Usage: python hedge_probe.py UPSTREAM_DIR SCENE name=meta.jsonl [...]
"""
import sys, os, json
up, scene = sys.argv[1], sys.argv[2]
sys.path.insert(0, up); os.chdir(up)
import local_evaluator as le  # noqa: E402


def load(mp):
    P = {}
    for line in open(mp):
        r = json.loads(line)
        if r['frame'] > 150:
            continue
        P[r['frame']] = [dict(object_id=c, bbox=tuple(v * (3840 if i % 2 == 0 else 2160) for i, v in enumerate(b)), confidence=s)
                         for c, b, s in r['ann']]
    return P


def hedge(P, pairs, k):
    out = {}
    for f, ds in P.items():
        extra = [dict(d, object_id=b, confidence=d['confidence'] * k) for a, b in pairs for d in ds if d['object_id'] == a]
        out[f] = ds + extra
    return out


V = [('none', [], 1.0), ('condor->jet x1', [('condor', 'jet_plane')], 1.0), ('condor->jet x.3', [('condor', 'jet_plane')], 0.3),
     ('medium->small x1', [('medium_plane', 'small_plane')], 1.0), ('medium->small x.3', [('medium_plane', 'small_plane')], 0.3),
     ('both x1', [('condor', 'jet_plane'), ('medium_plane', 'small_plane')], 1.0),
     ('both x.3', [('condor', 'jet_plane'), ('medium_plane', 'small_plane')], 0.3)]
for a in sys.argv[3:]:
    name, mp = a.split('=', 1)
    P = load(mp)
    ncond = sum(d['object_id'] == 'condor' for v in P.values() for d in v)
    nmed = sum(d['object_id'] == 'medium_plane' for v in P.values() for d in v)
    base = None
    for vn, pairs, k in V:
        m, per = le.score(scene, hedge(P, pairs, k))
        base = m if base is None else base
        print(f'{name:6s} {vn:18s} mAP {m:.4f} ({m - base:+.4f})  jet {per.get("jet_plane", 0):.3f} small_plane {per.get("small_plane", 0):.3f}'
              + (f'   [condor boxes {ncond}, medium_plane boxes {nmed}]' if vn == 'none' else ''))
