"""B3 step 1: per-frame training targets for the validation city (frames 1-150 ONLY).
pos = sure core objects from a built valcity scene (build_labels.py output, H-propagated boxes).
ign = everything that might be a real object but is not a sure label: unsure labels plus every answer cluster
(conf >= CONF over all recorded runs, frames <= 150) that is neither a core object nor an eyeballed clutter cluster.
Ignore regions are greyed out in training views so unlabelled real assets are never taught as background.
Usage: python make_targets.py H.npy CORE_SCENE_DIR CANDS.json LABELS.json OUT.json meta1.jsonl [...]"""
import sys, json, os, numpy as np
H = np.load(sys.argv[1]); SCENE = sys.argv[2]; CANDS = json.load(open(sys.argv[3])); LAB = json.load(open(sys.argv[4]))
OUT = sys.argv[5]; METAS = sys.argv[6:]
CONF, R, MAXF = float(os.environ.get('IGN_CONF', 0.1)), 30.0, int(os.environ.get('MAXF', 150))
Hi = np.linalg.inv(H)
P = {0: np.eye(3)}; F = {0: np.eye(3)}
for t in range(1, MAXF + 1): P[t] = P[t - 1] @ Hi; F[t] = F[t - 1] @ H


def pt(M, x, y):
    q = M @ [x, y, 1.0]; return q[0] / q[2], q[1] / q[2]


pos = {}; known = []  # known = frame-0 centres of core objects and clutter clusters
for t in range(1, MAXF + 1):
    a = json.load(open(f'{SCENE}/annotations/frame_{t:06d}.json'))['annotations']
    pos[t] = [[o['object_id'], *o['bbox'], bool(o.get('unsure'))] for o in a]
    for o in a:
        x1, y1, x2, y2 = o['bbox']; known.append(pt(P[t], (x1 + x2) / 2, (y1 + y2) / 2))
cmap = {c['id']: c for c in CANDS}
for cid in LAB['clutter']:
    c = cmap.get(cid)
    if c: known.append((c['gx'], c['gy']))
known = np.array(known)

dets = []
for mp in METAS:
    for line in open(mp):
        r = json.loads(line); t = r['frame']
        if t > MAXF or t < 1: continue  # frames > 150 are holdout: never read their answers
        for c, b, s in r['ann']:
            if s < CONF: continue
            x1, y1, x2, y2 = b[0] * 3840, b[1] * 2160, b[2] * 3840, b[3] * 2160
            g = pt(P[t], (x1 + x2) / 2, (y1 + y2) / 2)
            dets.append((g[0], g[1], s, x2 - x1, y2 - y1))
dets.sort(key=lambda d: -d[2])
cl = []
for d in dets:
    for c in cl:
        if (c['g'][0] - d[0]) ** 2 + (c['g'][1] - d[1]) ** 2 < R * R: c['m'].append(d); break
    else: cl.append(dict(g=(d[0], d[1]), m=[d]))
ign0 = []
for c in cl:
    if len(known) and np.min(np.hypot(known[:, 0] - c['g'][0], known[:, 1] - c['g'][1])) < 40: continue
    w = float(np.median([d[3] for d in c['m']])); h = float(np.median([d[4] for d in c['m']]))
    ign0.append((c['g'][0], c['g'][1], max(w, 16) * 1.4, max(h, 16) * 1.4))
out = {}
for t in range(1, MAXF + 1):
    ign = []
    for gx, gy, w, h in ign0:
        x, y = pt(F[t], gx, gy)
        if -w < x < 3840 + w and -h < y < 2160 + h: ign.append([x - w / 2, y - h / 2, x + w / 2, y + h / 2])
    p = []
    for c, x1, y1, x2, y2, uns in pos[t]:
        if uns: ign.append([x1 - 4, y1 - 4, x2 + 4, y2 + 4])
        else: p.append([c, x1, y1, x2, y2])
    out[t] = dict(pos=p, ign=ign)
json.dump(out, open(OUT, 'w'))
print('clusters', len(cl), 'ignore objects', len(ign0), 'pos boxes', sum(len(v['pos']) for v in out.values()),
      'ign boxes', sum(len(v['ign']) for v in out.values()))
