#!/usr/bin/env python3
"""FROZEN drone detector scorer (read-only after the first commit; write new scripts instead of editing).

Measures a single-view detector on real Helsinki worker frames, rendered exactly like the
evaluator (local_evaluator.render_view: crop the source region, INTER_AREA downscale by 4/2/1
to 960x540, PNG round trip is lossless so it is skipped).

Detector contract (module given by path, or the built-in names 'oracle' / 'none'):

    detect(image_bgr_960x540: np.ndarray, view_meta: dict) -> list of (class_name, [x1,y1,x2,y2], conf)

    boxes in VIEW pixels (0..960, 0..540); view_meta = {'resolution_level', 'center_x', 'center_y',
    'source_region_xyxy', 'width': 960, 'height': 540}. No frame number is passed.
    Optional: detect_batch(list_of_images, list_of_meta) -> list of lists (used if present).

Views per frame: level 0 = the full frame (1 view); level 1 = 2x2 non-overlapping tiles;
level 2 = 4x4 non-overlapping tiles (the tiles are exact legal camera positions).

Ground truth per view: every annotated box clipped to the view's source region. A clipped
object counts as GT if >= 50% of its area is inside the view, otherwise it is an IGNORE region
(detections matching it at IoU >= 0.5 are dropped, not counted as false positives).

Metric: AP@0.5 with COCO 101-point interpolation, pooled over all views of one level in the
evaluated block, macro-averaged over classes that have >= 1 GT at that level. Matching is
greedy by descending confidence to the best-IoU unmatched GT of the same class in the same view.
Detections are evaluated in view pixel space (IoU is scale invariant within a view).

2-block split (--fold):
    fold 0  -> evaluate frames 0-9   (the detector must be trained ONLY on frames 10-19)
    fold 1  -> evaluate frames 10-19 (the detector must be trained ONLY on frames 0-9)
The detector module receives os.environ['EVAL_DET_FOLD'] = fold, so it can load
the weights trained on the other block. Report both folds.

Usage:
    python drone/eval_det.py --det drone/detectors/yolo_synth.py --fold 0
    python drone/eval_det.py --det oracle --fold 1          # sanity: prints 1.000
Output: one line
    RESULT eval_det det=<name> fold=<f> L0=<ap> L1=<ap> L2=<ap> mean=<ap> n_views=<n> ms_per_view=<t>
"""
import argparse
import glob
import importlib.util
import json
import os
import sys
import time

import cv2
import numpy as np

W, H = 3840, 2160
VIEW = (960, 540)
REGION = {0: (3840, 2160), 1: (1920, 1080), 2: (960, 540)}
CLASSES = ('hangar', 'helicopter', 'jet_plane', 'mine_roller', 'small_launcher', 'medium_launcher',
           'large_launcher', 'small_plane', 'medium_plane', 'condor', 'spacecraft', 'small_tower',
           'large_tower', 'jammer', 'ta-ta', 'tank')
BLOCKS = {0: list(range(0, 10)), 1: list(range(10, 20))}
CANDIDATE_SCENES = [
    os.environ.get('DRONE_SCENE', ''),
    '/home/claude/upstream-work/drone-flyby/src/helsinki',
    '/home/claude/Nordic-AI-Cup-2026/drone-flyby/src/helsinki',
    '/workspace/upstream/drone-flyby/src/helsinki',
]


def find_scene(explicit=None):
    for s in ([explicit] if explicit else []) + CANDIDATE_SCENES:
        if s and os.path.isdir(os.path.join(s, 'images')):
            return s
    raise SystemExit('helsinki scene not found; pass --scene')


def load_gt(scene, frame):
    with open(os.path.join(scene, 'annotations', f'frame_{frame:06d}.json')) as f:
        d = json.load(f)
    return [(a['object_id'], [float(c) for c in a['bbox']]) for a in d['annotations']]


def view_regions(level):
    rw, rh = REGION[level]
    out = []
    for y in range(0, H, rh):
        for x in range(0, W, rw):
            out.append((x, y, x + rw, y + rh))
    return out


def render(frame_img, region):
    x1, y1, x2, y2 = region
    crop = frame_img[y1:y2, x1:x2]
    if (crop.shape[1], crop.shape[0]) != VIEW:
        crop = cv2.resize(crop, VIEW, interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(crop)


def view_gt(gt, region, level):
    """Returns list of (cls, box_in_view_px, ignore)."""
    rx1, ry1, rx2, ry2 = region
    s = REGION[level][0] / VIEW[0]
    out = []
    for cls, (x1, y1, x2, y2) in gt:
        cx1, cy1, cx2, cy2 = max(x1, rx1), max(y1, ry1), min(x2, rx2), min(y2, ry2)
        if cx2 <= cx1 or cy2 <= cy1:
            continue
        full = max(1e-9, (x2 - x1) * (y2 - y1))
        vis = (cx2 - cx1) * (cy2 - cy1) / full
        box = [(cx1 - rx1) / s, (cy1 - ry1) / s, (cx2 - rx1) / s, (cy2 - ry1) / s]
        out.append((cls, box, vis < 0.5))
    return out


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def ap101(tp_flags, n_gt):
    if n_gt == 0:
        return None
    if len(tp_flags) == 0:
        return 0.0
    tp = np.cumsum(tp_flags)
    fp = np.cumsum(1 - np.asarray(tp_flags))
    rec = tp / n_gt
    prec = tp / np.maximum(tp + fp, 1e-9)
    for i in range(len(prec) - 2, -1, -1):
        prec[i] = max(prec[i], prec[i + 1])
    total = 0.0
    for r in np.linspace(0, 1, 101):
        k = np.searchsorted(rec, r, side='left')
        total += prec[k] if k < len(prec) else 0.0
    return total / 101


def score_level(records):
    """records: list of (gts, dets) per view. gts [(cls, box, ignore)], dets [(cls, box, conf)]."""
    per_class = {}
    for c in CLASSES:
        n_gt = 0
        scored = []  # (conf, tp)
        for gts, dets in records:
            g = [(b, ign) for cls, b, ign in gts if cls == c]
            n_gt += sum(1 for _, ign in g if not ign)
            used = [False] * len(g)
            for cls, b, conf in sorted([d for d in dets if d[0] == c], key=lambda d: -d[2]):
                best, bj = 0.5, -1
                # prefer non-ignored GT (COCO semantics)
                for j, (gb, ign) in enumerate(g):
                    if used[j] or ign:
                        continue
                    v = iou(b, gb)
                    if v >= best:
                        best, bj = v, j
                if bj >= 0:
                    used[bj] = True
                    scored.append((conf, 1))
                    continue
                matched_ignore = any(ign and iou(b, gb) >= 0.5 for gb, ign in g)
                if not matched_ignore:
                    scored.append((conf, 0))
        scored.sort(key=lambda t: -t[0])
        ap = ap101([t for _, t in scored], n_gt)
        if ap is not None:
            per_class[c] = ap
    m = float(np.mean(list(per_class.values()))) if per_class else 0.0
    return m, per_class


def load_detector(spec):
    if spec == 'oracle':
        return None, 'oracle'
    if spec == 'none':
        return (lambda img, meta: []), 'none'
    path = os.path.abspath(spec)
    sys.path.insert(0, os.path.dirname(path))
    mod_spec = importlib.util.spec_from_file_location('eval_det_detector', path)
    mod = importlib.util.module_from_spec(mod_spec)
    mod_spec.loader.exec_module(mod)
    return mod, os.path.splitext(os.path.basename(path))[0]


def sanitize(dets):
    out = []
    for d in dets or []:
        try:
            cls, b, conf = d[0], [float(v) for v in d[1]], float(d[2])
        except Exception:
            continue
        if cls not in CLASSES or not (b[2] > b[0] and b[3] > b[1]):
            continue
        out.append((cls, b, conf))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--det', required=True, help="detector module path, or 'oracle' / 'none'")
    ap.add_argument('--fold', type=int, choices=[0, 1], required=True)
    ap.add_argument('--scene', default=None)
    ap.add_argument('--levels', type=int, nargs='+', default=[0, 1, 2])
    ap.add_argument('--per-class', action='store_true')
    ap.add_argument('--json', default=None, help='optional path to dump per-class results')
    a = ap.parse_args()

    os.environ['EVAL_DET_FOLD'] = str(a.fold)
    scene = find_scene(a.scene)
    mod, name = load_detector(a.det)
    frames = [f for f in BLOCKS[a.fold]
              if os.path.exists(os.path.join(scene, 'images', f'frame_{f:06d}.png'))]
    if not frames:
        raise SystemExit('no frames for this fold')

    records = {lv: [] for lv in a.levels}
    n_views, t_total = 0, 0.0
    for f in frames:
        img = cv2.imread(os.path.join(scene, 'images', f'frame_{f:06d}.png'), cv2.IMREAD_COLOR)
        gt = load_gt(scene, f)
        for lv in a.levels:
            regions = view_regions(lv)
            imgs, metas, gts = [], [], []
            for r in regions:
                imgs.append(render(img, r))
                cx, cy = (r[0] + r[2]) // 2, (r[1] + r[3]) // 2
                metas.append({'resolution_level': lv, 'center_x': cx, 'center_y': cy,
                              'source_region_xyxy': list(r), 'width': VIEW[0], 'height': VIEW[1]})
                gts.append(view_gt(gt, r, lv))
            t0 = time.perf_counter()
            if mod is None:  # oracle
                outs = [[(c, b, 1.0) for c, b, ign in g if not ign] for g in gts]
            elif hasattr(mod, 'detect_batch'):
                outs = mod.detect_batch(imgs, metas)
            elif hasattr(mod, 'detect'):
                outs = [mod.detect(im, m) for im, m in zip(imgs, metas)]
            else:
                outs = [mod(im, m) for im, m in zip(imgs, metas)]
            t_total += time.perf_counter() - t0
            n_views += len(imgs)
            for g, o in zip(gts, outs):
                records[lv].append((g, sanitize(o)))

    res, detail = {}, {}
    for lv in a.levels:
        res[lv], detail[lv] = score_level(records[lv])
    mean = float(np.mean(list(res.values())))
    if a.per_class:
        for lv in a.levels:
            print(f'L{lv}: ' + ' '.join(f'{c}={v:.2f}' for c, v in detail[lv].items()))
    if a.json:
        with open(a.json, 'w') as fh:
            json.dump({'det': name, 'fold': a.fold, 'levels': {str(k): v for k, v in res.items()},
                       'per_class': {str(k): v for k, v in detail.items()}}, fh, indent=1)
    lv_str = ' '.join(f'L{lv}={res[lv]:.4f}' for lv in a.levels)
    print(f'RESULT eval_det det={name} fold={a.fold} {lv_str} mean={mean:.4f} '
          f'n_views={n_views} ms_per_view={1000 * t_total / max(1, n_views):.1f}')


if __name__ == '__main__':
    main()
