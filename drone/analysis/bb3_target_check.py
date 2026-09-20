"""Are the 2 real objects that valcity_v1_core misses taught as BACKGROUND by drone-bigbet-3's fine-tune targets?
(drone understanding lane, cycle 4; analysis only.)

bigbet-3 (drone/bb3/make_targets.py) trains v2 on recorded validation-city views with positives = valcity core labels
and grey 'ignore' regions = unsure labels + answer clusters that are not within 40 px of a core object or clutter.
valcity_plus.py found 2 real objects v1_core misses: a third red-nosed small_plane between #9 and #29, and a second
black hangar that NEITHER detector ever answers. A real object that is neither positive nor greyed is a training
NEGATIVE. This propagates both objects to frames 1-150 exactly as valcity_plus.py does and reports, per object, how much
of its box is greyed (ignore minus positive boxes, as build_views.py paints it) in each frame.
Usage: python bb3_target_check.py Happrox.npy targets_v1.json
"""
import sys, json
import numpy as np
H = np.load(sys.argv[1]); Hi = np.linalg.inv(H); TG = json.load(open(sys.argv[2]))
EXTRA = [('small_plane#3', 126, (2750, 1002, 2792, 1033)), ('hangar#2', 140, (2700, 1880, 2825, 2080))]


def pt(M, x, y):
    q = M @ [x, y, 1]; return q[0] / q[2], q[1] / q[2]


P = {0: np.eye(3)}; F = {0: np.eye(3)}
for t in range(1, 151):
    P[t] = P[t - 1] @ Hi; F[t] = F[t - 1] @ H
for name, t0, (x1, y1, x2, y2) in EXTRA:
    a = pt(P[t0], x1, y1); b = pt(P[t0], x2, y2)
    cov, vis = [], 0
    for t in range(1, 151):
        cs = np.array([pt(F[t], x, y) for x, y in ((a[0], a[1]), (b[0], a[1]), (a[0], b[1]), (b[0], b[1]))])
        lo, hi = np.clip(cs.min(0), 0, [3840, 2160]), np.clip(cs.max(0), 0, [3840, 2160])
        if hi[0] - lo[0] < 4 or hi[1] - lo[1] < 4:
            continue
        vis += 1
        X0, Y0 = int(lo[0]), int(lo[1]); w, h = int(np.ceil(hi[0])) - X0, int(np.ceil(hi[1])) - Y0
        m = np.zeros((h, w), bool)
        for q in TG[str(t)]['ign']:
            qx1, qy1, qx2, qy2 = [int(round(v)) for v in q]
            m[max(0, qy1 - Y0):max(0, min(h, qy2 - Y0)), max(0, qx1 - X0):max(0, min(w, qx2 - X0))] = True
        for c, qx1, qy1, qx2, qy2 in TG[str(t)]['pos']:   # build_views never greys over a positive box
            m[max(0, int(qy1) - 1 - Y0):max(0, min(h, int(np.ceil(qy2)) + 1 - Y0)), max(0, int(qx1) - 1 - X0):max(0, min(w, int(np.ceil(qx2)) + 1 - X0))] = False
        cov.append((t, m.mean()))
    c = np.array([v for _, v in cov])
    print(f'{name}: in frame {vis}/150 frames; greyed share of its box: median {np.median(c):.2f}, '
          f'frames >= 50% greyed {int((c >= .5).sum())}, frames < 10% greyed (taught as background) {int((c < .1).sum())}; '
          f'first/last frame {cov[0][0]}/{cov[-1][0]}')
