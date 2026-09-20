"""Same as hel_boxratio but per VIEW LEVEL: the evaluator renders a level-l view by cropping 960*s x 540*s source px
and INTER_AREA-downscaling by s (s = 4/2/1 for L0/L1/L2). Measures pred/GT box ratio and IoU per level."""
import glob, json, os, sys, collections
import numpy as np, cv2
SCENE = '/workspace/upstream/drone-flyby/src/helsinki'
OUT = sys.argv[1]; models = [a.split('=', 1) for a in sys.argv[2:] if '=' in a]
from ultralytics import YOLO
SC = {0: 4, 1: 2, 2: 1}
res = collections.defaultdict(list)
for name, w in models:
    m = YOLO(w, task='detect'); nm = m.names
    for ap in sorted(glob.glob(f'{SCENE}/annotations/*.json')):
        d = json.load(open(ap)); f = d['frame']
        img = cv2.imread(f'{SCENE}/images/frame_{f:06d}.png')
        for a in d['annotations']:
            x1, y1, x2, y2 = a['bbox']; cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            if x2 - x1 < 6 or y2 - y1 < 6 or x1 < 4 or y1 < 4 or x2 > 3836 or y2 > 2156: continue
            for lev in (0, 1, 2):
                s = SC[lev]
                rx = int(np.clip(cx - 480 * s, 0, 3840 - 960 * s)); ry = int(np.clip(cy - 270 * s, 0, 2160 - 540 * s))
                tile = img[ry:ry + 540 * s, rx:rx + 960 * s]
                if s > 1: tile = cv2.resize(tile, (960, 540), interpolation=cv2.INTER_AREA)
                for isz in (960, 1280, 1920):
                    r = m.predict(tile, imgsz=isz, conf=0.05, verbose=False)[0]
                    gx1, gy1, gx2, gy2 = (x1 - rx) / s, (y1 - ry) / s, (x2 - rx) / s, (y2 - ry) / s
                    best, bi = 0.0, None
                    for b, c, cf in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy(), r.boxes.conf.cpu().numpy()):
                        ix1, iy1 = max(gx1, b[0]), max(gy1, b[1]); ix2, iy2 = min(gx2, b[2]), min(gy2, b[3])
                        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                        iou = inter / ((gx2 - gx1) * (gy2 - gy1) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)
                        if iou > best: best, bi = iou, (b, nm[int(c)])
                    if bi is None or best < 0.15:
                        res[(name, lev, isz)].append((None, None, 0.0, a['object_id'], 0)); continue
                    b, pc = bi
                    res[(name, lev, isz)].append(((b[2] - b[0]) / max(1e-9, gx2 - gx1), (b[3] - b[1]) / max(1e-9, gy2 - gy1),
                                                  best, a['object_id'], int(pc == a['object_id'])))
out = {}
for k in sorted(res):
    v = res[k]; hit = [e for e in v if e[0] is not None]
    if not hit: continue
    rw = float(np.median([e[0] for e in hit])); rh = float(np.median([e[1] for e in hit]))
    iou = float(np.median([e[2] for e in hit])); lo = float(np.mean([e[2] < 0.5 for e in v]))
    out[str(k)] = dict(n=len(v), hit=len(hit), rw=round(rw, 3), rh=round(rh, 3), iou=round(iou, 3),
                       frac_iou_lt50=round(lo, 3), cls_ok=round(float(np.mean([e[4] for e in hit])), 3))
    print(f'{k[0]:8s} L{k[1]} imgsz{k[2]:5d}  n{len(v):3d} hit{len(hit):3d}  pred/GT w {rw:5.2f} h {rh:5.2f}  '
          f'medIoU {iou:.3f}  frac(IoU<0.5) {lo:.3f}  cls_ok {out[str(k)]["cls_ok"]:.2f}')
json.dump(out, open(OUT, 'w'), indent=1)
print('BOXLEVEL_DONE')
