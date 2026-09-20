"""Why is hangar #2 invisible to v2? Orientation or appearance? (drone understanding lane, cycle 4; analysis only)

Hangar #1 is detected; hangar #2 (vertical, near-black, bottom-right, frames 109-143) is never answered by v2 or r11.
Take every recorded L1/L2 view (frames <= 150) that fully contains hangar #1 (valcity_v1_core box), run v2 on the view
and on transformed copies (rot90/180/270, h-flip, darkened x0.35), and report v2's best 'hangar' conf on the hangar #1
box (IoU >= .3 after mapping the box through the same transform) plus the best conf of ANY class there.
Usage (gpu3): python hangar_probe.py --drone DRONE_DIR --scene SCENE_DIR --run DIR [...] [--cluster eye1 --weights r11.pt]
(--cluster eye1 with the valcity_v1_plus scene probes hangar #2 itself; a box fully inside the view is required.)
"""
import argparse, json, os, sys, glob
import numpy as np
import cv2

ap = argparse.ArgumentParser()
ap.add_argument('--drone', required=True); ap.add_argument('--scene', required=True); ap.add_argument('--run', action='append', required=True)
ap.add_argument('--weights', default='/workspace/drone_weights/v2.pt'); ap.add_argument('--max', type=int, default=12)
ap.add_argument('--cluster', default='')  # e.g. eye1 = hangar #2 in valcity_v1_plus
a = ap.parse_args()
sys.path.insert(0, a.drone)
from common import CLASSES, LEVEL_SCALE  # noqa: E402
from detector import YoloDetector  # noqa: E402
GT = {}
for f in glob.glob(os.path.join(a.scene, 'annotations', '*.json')):
    d = json.load(open(f)); GT[d['frame']] = [g['bbox'] for g in d['annotations'] if g['object_id'] == 'hangar' and (not a.cluster or g.get('valcity_cluster') == a.cluster)]


def iou(p, q):
    ix = max(0, min(p[2], q[2]) - max(p[0], q[0])); iy = max(0, min(p[3], q[3]) - max(p[1], q[1])); i = ix * iy
    u = (p[2] - p[0]) * (p[3] - p[1]) + (q[2] - q[0]) * (q[3] - q[1]) - i
    return i / u if u > 0 else 0


def tf_box(b, k, W, H):  # box after the image transform k (on a W x H view)
    x1, y1, x2, y2 = b
    if k == 'rot90':   # cv2.ROTATE_90_CLOCKWISE: (x, y) -> (H - y, x)
        return [H - y2, x1, H - y1, x2]
    if k == 'rot180':
        return [W - x2, H - y2, W - x1, H - y1]
    if k == 'rot270':  # counterclockwise: (x, y) -> (y, W - x)
        return [y1, W - x2, y2, W - x1]
    if k == 'flip':
        return [W - x2, y1, W - x1, y2]
    return list(b)


TF = {'orig': lambda im: im, 'rot90': lambda im: cv2.rotate(im, cv2.ROTATE_90_CLOCKWISE), 'rot180': lambda im: cv2.rotate(im, cv2.ROTATE_180),
      'rot270': lambda im: cv2.rotate(im, cv2.ROTATE_90_COUNTERCLOCKWISE), 'flip': lambda im: cv2.flip(im, 1),
      'dark': lambda im: (im.astype(np.float32) * 0.35).astype(np.uint8)}
yolo = YoloDetector(a.weights)
res = {k: [] for k in TF}; resany = {k: [] for k in TF}; n = 0
for run in a.run:
    for m in sorted((json.loads(l) for l in open(os.path.join(run, 'meta.jsonl'))), key=lambda m: m['frame']):
        f, lv = m['frame'], m['level']
        if f > 150 or lv == 0 or not GT.get(f) or n >= a.max * len(a.run):
            continue
        rg = m['region']; s = LEVEL_SCALE[lv]; g = GT[f][0]
        if not (g[0] >= rg[0] and g[1] >= rg[1] and g[2] <= rg[2] and g[3] <= rg[3]):
            continue
        img = cv2.imread(os.path.join(run, m['file']))
        if img is None:
            continue
        n += 1
        vb = [(g[0] - rg[0]) / s, (g[1] - rg[1]) / s, (g[2] - rg[0]) / s, (g[3] - rg[1]) / s]
        for k, fn in TF.items():
            im = np.ascontiguousarray(fn(img)); H, W = img.shape[:2]
            tb = tf_box(vb, k, W, H)
            best, bestany = 0.0, (0.0, '-')
            for d in yolo(im, lv):
                if iou(d['box'], tb) >= .3:
                    best = max(best, float(d['probs'][CLASSES.index('hangar')]))
                    c = int(np.argmax(d['probs']))
                    if d['conf'] > bestany[0]:
                        bestany = (float(d['conf']), CLASSES[c])
            res[k].append(best); resany[k].append(bestany)
        print(f, f'L{lv}', 'view box', [round(v) for v in vb], ' '.join(f'{k}={res[k][-1]:.2f}({resany[k][-1][1][:6]})' for k in TF), flush=True)
print('views', n)
for k in TF:
    v = np.array(res[k])
    print(f'{k:6s} hangar conf median {np.median(v):.2f}  detected(>=.25) {int((v >= .25).sum())}/{len(v)}')
