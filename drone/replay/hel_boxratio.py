"""Per-class box-convention ratio: run the table's detectors on native-L2 crops of the 20 annotated Helsinki frames
(same view geometry as the flown city's survey views) and compare every matched prediction with the GT box.
Out: json {model: {cls: {n, rw(med pred_w/gt_w), rh, rarea}}} + a printed table."""
import glob, json, os, sys, collections
import numpy as np, cv2
SCENE = '/workspace/upstream/drone-flyby/src/helsinki'
OUT = sys.argv[1]
models = [a.split('=', 1) for a in sys.argv[2:] if '=' in a]
from ultralytics import YOLO
res = collections.defaultdict(lambda: collections.defaultdict(list))
ann = sorted(glob.glob(f'{SCENE}/annotations/*.json'))
for name, w in models:
    m = YOLO(w, task='detect'); nm = m.names
    for ap in ann:
        d = json.load(open(ap)); f = d['frame']
        ip = f'{SCENE}/images/frame_{f:06d}.png'
        if not os.path.exists(ip): ip = f'{SCENE}/images/frame_{f:06d}.jpg'
        if not os.path.exists(ip): print('no image', f); continue
        img = cv2.imread(ip)
        for a in d['annotations']:
            x1, y1, x2, y2 = a['bbox']; cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            if x2 - x1 < 6 or y2 - y1 < 6: continue          # edge slivers: no usable extent
            if x1 < 4 or y1 < 4 or x2 > 3836 or y2 > 2156: continue  # truncated
            rx, ry = int(np.clip(cx - 480, 0, 3840 - 960)), int(np.clip(cy - 270, 0, 2160 - 540))
            tile = img[ry:ry + 540, rx:rx + 960]
            r = m.predict(tile, imgsz=960, conf=0.05, verbose=False)[0]
            best, bi = 0.0, None
            gx1, gy1, gx2, gy2 = x1 - rx, y1 - ry, x2 - rx, y2 - ry
            for b, c, cf in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy(), r.boxes.conf.cpu().numpy()):
                ix1, iy1 = max(gx1, b[0]), max(gy1, b[1]); ix2, iy2 = min(gx2, b[2]), min(gy2, b[3])
                inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                iou = inter / ((gx2 - gx1) * (gy2 - gy1) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)
                if iou > best: best, bi = iou, (b, nm[int(c)], float(cf))
            if bi is None or best < 0.20: 
                res[name][a['object_id']].append((None, None, None, best)); continue
            b, pc, cf = bi
            res[name][a['object_id']].append(((b[2] - b[0]) / (x2 - x1), (b[3] - b[1]) / (y2 - y1), pc, best))
out = {}
for name in res:
    out[name] = {}
    print('=== ', name)
    for c in sorted(res[name]):
        v = res[name][c]; hit = [e for e in v if e[0] is not None]
        if not hit:
            print(f'  {c:17s} n{len(v):3d} NO MATCH'); out[name][c] = dict(n=0, miss=len(v)); continue
        rw = float(np.median([e[0] for e in hit])); rh = float(np.median([e[1] for e in hit]))
        cls_ok = sum(1 for e in hit if e[2] == c)
        out[name][c] = dict(n=len(hit), miss=len(v) - len(hit), rw=round(rw, 3), rh=round(rh, 3),
                            rarea=round(float(np.median([e[0] * e[1] for e in hit])), 3), cls_ok=cls_ok,
                            iou=round(float(np.median([e[3] for e in hit])), 3))
        print(f'  {c:17s} n{len(hit):3d} miss{len(v)-len(hit):3d}  pred/GT  w {rw:5.2f}  h {rh:5.2f}  '
              f'sqrt(area) {np.sqrt(np.median([e[0]*e[1] for e in hit])):5.2f}  IoU {np.median([e[3] for e in hit]):.2f}  cls_ok {cls_ok}/{len(hit)}')
json.dump(out, open(OUT, 'w'), indent=1)
print('BOXRATIO_DONE')
