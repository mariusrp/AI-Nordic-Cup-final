"""M2 step 2: write src/valcity (frames 1-150) from eyeball-labelled candidate clusters.
Re-runs the cands.py clustering (same H / CONF / R / meta order -> same ids) to get every member detection, maps member
box corners to frame-0 ground coords (H^-t), takes the conf-weighted median box as the object's frame-0 box (or a
manual size override about that centre), then propagates it to every frame 1..150 with H^t, clips to 3840x2160 and
keeps it if the clipped box is >= 4 px on both sides. Images: tiny placeholders unless --mosaic (replay scoring only
needs annotations + the frame list).
Usage: python build_labels.py H.npy LABELS.json OUT_SCENE_DIR meta1.jsonl [...]   (meta order must match cands.py run)"""
import sys, json, os, numpy as np
H = np.load(sys.argv[1]); LAB = json.load(open(sys.argv[2])); OUT = sys.argv[3]; METAS = sys.argv[4:]
Hi = np.linalg.inv(H)
P = {0: np.eye(3)}
MAXF = int(os.environ.get('MAXF', 150))
for t in range(1, MAXF + 1): P[t] = P[t - 1] @ Hi
CONF, R = 0.25, 30.0
dets = []
for ri, mp in enumerate(METAS):
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


def pt(M, x, y):
    q = M @ [x, y, 1]; return q[0] / q[2], q[1] / q[2]


def wmed(v, w):
    o = np.argsort(v); c = np.cumsum(w[o]); return float(v[o][np.searchsorted(c, c[-1] / 2)])


objs = []
for key, lab in LAB['objects'].items():
    if key.startswith('m'):  # manual object: box in source coords of frame lab['t'] (found by eye on render.py frames)
        t = lab['t']; x1, y1, x2, y2 = lab['box']; a = pt(P[t], x1, y1); b = pt(P[t], x2, y2)
        objs.append(dict(cid=key, cls=lab['cls'], box0=[a[0], a[1], b[0], b[1]], unsure=lab.get('unsure', False), n=0))
        continue
    cid = int(key)
    m = cl[cid]['m']
    B = []
    for d in m:
        t = d[4]; x1, y1 = pt(P[t], d[8], d[9]); x2, y2 = pt(P[t], d[10], d[11]); B.append((x1, y1, x2, y2, d[3]))
    B = np.array(B); w = B[:, 4]
    box = [wmed(B[:, i], w) for i in range(4)]
    if 'size' in lab:  # manual size at mid-life frame, converted to frame 0 by the local scale
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        tm = int(np.median([d[4] for d in m])); sx, sy = np.diag(np.linalg.matrix_power(Hi, tm))[:2]
        box = [cx - lab['size'][0] * sx / 2, cy - lab['size'][1] * sy / 2, cx + lab['size'][0] * sx / 2, cy + lab['size'][1] * sy / 2]
    objs.append(dict(cid=cid, cls=lab['cls'], box0=box, unsure=lab.get('unsure', False), n=len(m)))

os.makedirs(f'{OUT}/annotations', exist_ok=True); os.makedirs(f'{OUT}/images', exist_ok=True)
Hf = {0: np.eye(3)}
for t in range(1, 151): Hf[t] = Hf[t - 1] @ H
nbox = 0
for t in range(1, 151):
    anns = []
    for o in objs:
        x1, y1, x2, y2 = o['box0']
        cs = np.array([pt(Hf[t], x, y) for x, y in ((x1, y1), (x2, y1), (x1, y2), (x2, y2))])
        a, b = cs.min(0), cs.max(0)
        a = np.clip(a, 0, [3840, 2160]); b = np.clip(b, 0, [3840, 2160])
        if b[0] - a[0] >= 4 and b[1] - a[1] >= 4:
            anns.append(dict(object_id=o['cls'], bbox=[int(round(a[0])), int(round(a[1])), int(round(b[0])), int(round(b[1]))],
                             valcity_cluster=o['cid'], unsure=o['unsure']))
    nbox += len(anns)
    json.dump(dict(frame=t, annotations=anns), open(f'{OUT}/annotations/frame_{t:06d}.json', 'w'))
    ip = f'{OUT}/images/frame_{t:06d}.png'
    if not os.path.exists(ip):
        import cv2; cv2.imwrite(ip, np.zeros((8, 8, 3), np.uint8))
json.dump(dict(note='valcity v0: frames 1-150 of the validation flight, labels from eyeballed clusters + H propagation',
               objects=objs, H=H.tolist()), open(f'{OUT}/run_metadata.json', 'w'), indent=1)
print('objects', len(objs), 'boxes', nbox)
