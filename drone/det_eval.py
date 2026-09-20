"""FROZEN detector metric (do not edit after first commit; add new scripts instead).

Leave-frames-out detector quality on real Helsinki renders. For every held-out
frame (make_synth.HOLDOUT_FRAMES, never used for cut-outs/backgrounds) we tile the
whole 4K frame with camera views at one resolution level, rendered exactly like
local_evaluator.render_view (crop + INTER_AREA), run the detector, lift boxes to
source pixels, class-wise NMS across tiles, and score COCO mAP@0.5 (macro over
the classes present in those frames), like the upstream scorer.

    python det_eval.py [--weights w.pt] [--frames 4 12 20] [--levels 0 1 2]

Caveat: the same 16 instances appear in the training frames (adjacent frames),
so this is optimistic about new instances; it is mainly a relative measure.
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CLASSES, DRONE_UPSTREAM, LEVEL_SCALE  # noqa: E402

SCENE = os.path.join(DRONE_UPSTREAM, "src", "helsinki")
REG = {0: (3840, 2160), 1: (1920, 1080), 2: (960, 540)}


def tiles(level):
    rw, rh = REG[level]
    xs = np.linspace(0, 3840 - rw, max(1, int(np.ceil(3840 / rw * 1.3)) if level else 1)).astype(int)
    ys = np.linspace(0, 2160 - rh, max(1, int(np.ceil(2160 / rh * 1.3)) if level else 1)).astype(int)
    return [(x, y, x + rw, y + rh) for y in ys for x in xs]


def nms(dets, thr=0.5):
    out = []
    for d in sorted(dets, key=lambda d: -d[2]):
        ok = True
        for o in out:
            if o[1] != d[1]:
                continue
            a, b = o[0], d[0]
            ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
            u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy
            if u > 0 and ix * iy / u > thr:
                ok = False
                break
        if ok:
            out.append(d)
    return out


def coco_map(gt, pred):
    from faster_coco_eval import COCO, COCOeval_faster
    frames = sorted(gt)
    fid = {f: i + 1 for i, f in enumerate(frames)}
    cid = {c: i + 1 for i, c in enumerate(CLASSES)}
    present = sorted({a["object_id"] for f in frames for a in gt[f]}, key=CLASSES.index)
    anns, k = [], 1
    for f in frames:
        for a in gt[f]:
            x1, y1, x2, y2 = a["bbox"]
            anns.append(dict(id=k, image_id=fid[f], category_id=cid[a["object_id"]], bbox=[x1, y1, x2 - x1, y2 - y1], area=(x2 - x1) * (y2 - y1), iscrowd=0))
            k += 1
    g = dict(images=[dict(id=fid[f], width=3840, height=2160, file_name=str(f)) for f in frames],
             categories=[dict(id=cid[c], name=c) for c in CLASSES], annotations=anns)
    dts = [dict(image_id=fid[f], category_id=cid[CLASSES[c]], bbox=[b[0], b[1], b[2] - b[0], b[3] - b[1]], score=s)
           for f in frames for b, c, s in pred.get(f, [])]
    if not dts:
        return 0.0, {}
    cg = COCO(g)
    ev = COCOeval_faster(cg, cg.loadRes(dts), "bbox")
    ev.params.catIds = [cid[c] for c in present]
    ev.params.iouThrs = np.array([0.5])
    ev.evaluate(); ev.accumulate()
    P = ev.eval["precision"]
    ap = {}
    for i, c in enumerate(present):
        v = P[0, :, i, 0, -1]
        v = v[v > -1]
        ap[c] = float(v.mean()) if v.size else 0.0
    return float(np.mean(list(ap.values()))), ap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights", "best.pt"))
    ap.add_argument("--frames", type=int, nargs="*", default=[4, 12, 20])
    ap.add_argument("--levels", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--imgsz", type=int, default=int(os.environ.get("DRONE_IMGSZ", "1280")))
    a = ap.parse_args()
    from detector import YoloDetector
    det = YoloDetector(a.weights, imgsz=a.imgsz)
    gt = {f: json.load(open(os.path.join(SCENE, "annotations", f"frame_{f:06d}.json")))["annotations"] for f in a.frames}
    res = {}
    for L in a.levels:
        pred = {}
        for f in a.frames:
            img = cv2.imread(os.path.join(SCENE, "images", f"frame_{f:06d}.png"))
            s = LEVEL_SCALE[L]
            dets = []
            for (x1, y1, x2, y2) in tiles(L):
                v = img[y1:y2, x1:x2]
                if L < 2:
                    v = cv2.resize(v, (960, 540), interpolation=cv2.INTER_AREA)
                for d in det(v, L):
                    b = d["box"]
                    c = int(np.argmax(d["probs"]))
                    dets.append(((x1 + b[0] * s, y1 + b[1] * s, x1 + b[2] * s, y1 + b[3] * s), c, d["conf"] * d["probs"][c]))
            pred[f] = nms(dets)
        m, per = coco_map(gt, pred)
        res[L] = m
        print(f"L{L} mAP50 {m:.3f}  " + " ".join(f"{c[:6]}={v:.2f}" for c, v in per.items()), flush=True)
    print("DET_EVAL", json.dumps({"weights": a.weights, **{f"L{k}": round(v, 4) for k, v in res.items()}}))


if __name__ == "__main__":
    main()
