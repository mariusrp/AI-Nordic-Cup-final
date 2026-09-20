"""Does an offline replay reproduce what the live server answered? (drone understanding lane, cycle 5)

Compares, frame by frame (frames <= 150), the recorder's live answers of a platform validation run with the replay of the
SAME recorded views through the same production stack (route_probe5.py P_base). Box agreement: a live box is matched by a
replay box of the same class with IoU >= .9; reports the matched share, the median |conf diff| of matches, and both
answer streams' valcity_plus mAP.
Usage: python fidelity.py UPSTREAM_DIR name=live.jsonl,replay.jsonl [...]
"""
import sys, os, json
import numpy as np
up = sys.argv[1]
sys.path.insert(0, up); os.chdir(up)
import local_evaluator as le  # noqa: E402


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1])); i = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0


def rd(p):
    return {r['frame']: r['ann'] for r in (json.loads(l) for l in open(p)) if r['frame'] <= 150}


def pred(S):
    return {f: [dict(object_id=c, bbox=tuple(v * (3840 if i % 2 == 0 else 2160) for i, v in enumerate(b)), confidence=s) for c, b, s in A]
            for f, A in S.items()}


for a in sys.argv[2:]:
    name, ps = a.split('=', 1); lp, rp = ps.split(',')
    L, R = rd(lp), rd(rp)
    n = m = 0; dc = []; first_bad = None
    for f in sorted(L):
        for c, b, s in L[f]:
            n += 1
            best = max(((iou(b, b2), s2) for c2, b2, s2 in R.get(f, []) if c2 == c), default=(0, 0))
            if best[0] >= .9:
                m += 1; dc.append(abs(best[1] - s))
            elif first_bad is None:
                first_bad = f
    ml, mr = le.score('valcity_v1_plus', pred(L))[0], le.score('valcity_v1_plus', pred(R))[0]
    nr = sum(len(v) for v in R.values())
    print(f'{name}: live boxes {n}, replay boxes {nr}, live matched {m / max(1, n):.3f}, median |dconf| {np.median(dc) if dc else -1:.4f}, '
          f'first unmatched frame {first_bad}; valcity_plus live {ml:.4f} replay {mr:.4f}')
