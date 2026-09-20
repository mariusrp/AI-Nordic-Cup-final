"""M2 step 1: candidate physical objects from the union of recorded platform answers (frames 1-150 only).
Each answer box centre is mapped to frame-0 ground coords g = H^-t x; greedy clustering (radius R px) gives one
identity per physical object. Per cluster: class votes (conf-weighted), #frames, #runs, box size, first/last frame.
Usage: python cands.py H.npy OUT.json meta1.jsonl [...]"""
import sys, json, os, numpy as np
H = np.load(sys.argv[1]); OUT = sys.argv[2]
Hi = np.linalg.inv(H)
P = {0: np.eye(3)}
MAXF = int(os.environ.get('MAXF', 150))
for t in range(1, MAXF + 1): P[t] = P[t - 1] @ Hi   # frame t -> frame 0
CONF, R = 0.25, 30.0
dets = []
for ri, mp in enumerate(sys.argv[3:]):
    for line in open(mp):
        r = json.loads(line); t = r['frame']
        if t > MAXF: continue
        for c, b, s in r['ann']:
            if s < CONF: continue
            x1, y1, x2, y2 = b[0] * 3840, b[1] * 2160, b[2] * 3840, b[3] * 2160
            q = P[t] @ [(x1 + x2) / 2, (y1 + y2) / 2, 1]
            dets.append((q[0] / q[2], q[1] / q[2], c, s, t, ri, x2 - x1, y2 - y1, x1, y1, x2, y2))
dets.sort(key=lambda d: -d[3])
cl = []
for d in dets:
    for c in cl:
        if (c['gx'] - d[0]) ** 2 + (c['gy'] - d[1]) ** 2 < R * R:
            c['m'].append(d); break
    else:
        cl.append(dict(gx=d[0], gy=d[1], m=[d]))
out = []
for i, c in enumerate(cl):
    m = c['m']; votes = {}
    for d in m: votes[d[2]] = votes.get(d[2], 0) + d[3]
    frames = sorted({d[4] for d in m}); runs = sorted({d[5] for d in m})
    g = np.array([(d[0], d[1]) for d in m])
    out.append(dict(id=i, gx=float(np.median(g[:, 0])), gy=float(np.median(g[:, 1])), votes=dict(sorted(votes.items(), key=lambda kv: -kv[1])),
                    nframes=len(frames), nruns=len(runs), first=frames[0], last=frames[-1], maxconf=max(d[3] for d in m),
                    w=float(np.median([d[6] for d in m])), h=float(np.median([d[7] for d in m])), n=len(m),
                    best=max(m, key=lambda d: d[3])[4:12]))
out.sort(key=lambda o: -(o['nframes'] * o['nruns']))
json.dump(out, open(OUT, 'w'), indent=0)
print('dets', len(dets), 'clusters', len(out))
for o in out[:60]:
    v = list(o['votes'].items())[:3]
    print(o['id'], 'g=(%.0f,%.0f)' % (o['gx'], o['gy']), 'fr %d-%d n%d runs%d' % (o['first'], o['last'], o['nframes'], o['nruns']),
          'maxc %.2f' % o['maxconf'], 'wh %.0fx%.0f' % (o['w'], o['h']), ' '.join(f'{k}:{s:.1f}' for k, s in v))
