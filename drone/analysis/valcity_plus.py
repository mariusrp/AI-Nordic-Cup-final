"""valcity_v1_core + objects found by eye that the labels miss (drone understanding lane, cycle 2).

By-eye check (frames <= 150): the L2 view of frame 126 (run 11b4714e, region [1950,870,2910,1410]) shows THREE identical
red-nosed small_planes where valcity labels two (#9, #29), and a second black arched hangar next to jet #171; the render.py
reconstruction of frame 140 shows the same. This writes a scene copy with those objects added (boxes found by eye, then
propagated to frames 1-150 with the same per-step homography as build_labels.py) and rescores recorded runs on it.
Usage: python valcity_plus.py UPSTREAM_DIR Happrox.npy name=meta.jsonl@platform [...]
"""
import sys, os, json, shutil, glob
import numpy as np
up, Hp = sys.argv[1], sys.argv[2]
H = np.load(Hp); Hi = np.linalg.inv(H)
EXTRA = [  # (class, frame t, box in source px of frame t)
    ('small_plane', 126, (2750, 1002, 2792, 1033)),   # the unlabelled middle plane (L2 view, frame 126)
    ('hangar', 140, (2700, 1880, 2825, 2080)),        # second hangar (reconstructed frame 140, bottom-right quadrant)
]
src = os.path.join(up, 'src', 'valcity_v1_core'); dst = os.path.join(up, 'src', 'valcity_v1_plus')
if os.path.exists(dst):
    shutil.rmtree(dst)
shutil.copytree(src, dst)


def pt(M, x, y):
    q = M @ [x, y, 1]; return q[0] / q[2], q[1] / q[2]


P = {0: np.eye(3)}; Hf = {0: np.eye(3)}
for t in range(1, 151):
    P[t] = P[t - 1] @ Hi; Hf[t] = Hf[t - 1] @ H
added = 0
for k, (cls, t0, (x1, y1, x2, y2)) in enumerate(EXTRA):
    a = pt(P[t0], x1, y1); b = pt(P[t0], x2, y2)
    for t in range(1, 151):
        cs = np.array([pt(Hf[t], x, y) for x, y in ((a[0], a[1]), (b[0], a[1]), (a[0], b[1]), (b[0], b[1]))])
        lo, hi = np.clip(cs.min(0), 0, [3840, 2160]), np.clip(cs.max(0), 0, [3840, 2160])
        if hi[0] - lo[0] >= 4 and hi[1] - lo[1] >= 4:
            p = os.path.join(dst, 'annotations', f'frame_{t:06d}.json')
            d = json.load(open(p))
            d['annotations'].append(dict(object_id=cls, bbox=[int(round(lo[0])), int(round(lo[1])), int(round(hi[0])), int(round(hi[1]))],
                                         valcity_cluster=f'eye{k}', unsure=False))
            json.dump(d, open(p, 'w')); added += 1
print(f'added {added} GT boxes for {len(EXTRA)} objects')
sys.path.insert(0, up); os.chdir(up)
import local_evaluator as le  # noqa: E402


def load(mp):
    Pd = {}
    for line in open(mp):
        r = json.loads(line)
        if r['frame'] > 150:
            continue
        Pd[r['frame']] = [dict(object_id=c, bbox=tuple(v * (3840 if i % 2 == 0 else 2160) for i, v in enumerate(b)), confidence=s)
                          for c, b, s in r['ann']]
    return Pd


rows = []
for a in sys.argv[3:]:
    name, spec = a.split('=', 1)
    mp, _, ps = spec.partition('@')
    Pd = load(mp)
    m0, per0 = le.score('valcity_v1_core', Pd)
    m1, per1 = le.score('valcity_v1_plus', Pd)
    rows.append((name, float(ps), m0, m1))
    print(f'{name:8s} platform {float(ps):.4f}  core {m0:.4f} -> plus {m1:.4f} ({m1 - m0:+.4f})  small_plane {per0["small_plane"]:.3f}->{per1["small_plane"]:.3f}  hangar {per0["hangar"]:.3f}->{per1["hangar"]:.3f}')
p = np.array([r[1] for r in rows])
for j, lab in ((2, 'core'), (3, 'plus')):
    v = np.array([r[j] for r in rows])
    rho = np.corrcoef(v.argsort().argsort(), p.argsort().argsort())[0, 1]
    conc = sum(1 for i in range(len(rows)) for k in range(i + 1, len(rows)) if (v[i] - v[k]) * (p[i] - p[k]) > 0)
    print(f'{lab}: Spearman vs platform {rho:.2f}, concordant pairs {conc}/{len(rows) * (len(rows) - 1) // 2}, Pearson {np.corrcoef(v, p)[0, 1]:.2f}')
