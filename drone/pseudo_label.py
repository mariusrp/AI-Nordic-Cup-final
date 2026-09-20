"""Turn recorded validation runs into extra YOLO training data (self-training).

Replays each recorded sequence (<record_dir>/<seq>/meta.jsonl + PNG views) through
the offline pipeline (motion + detector + tracker). A track's class and
confidence are decided with hindsight over the whole sequence. Then, for every
view, the detections matched to a confident track become labels (box from that
frame's detection, class from the track). Views holding any ambiguous
detection (confidence between LOW and the track threshold) are skipped, so
unlabeled objects never become background.

    python pseudo_label.py --rec /workspace/drone_seen --out data/pseudo --weights weights/best.pt
Then train with EXTRA_ARGS or merge: the output is a YOLO split (images/, labels/) that
make_synth.py --merge can append to a dataset (see train.sh).
"""
import argparse
import glob
import json
import os
import shutil

import cv2
import numpy as np

import common  # noqa
from common import CLASSES, LEVEL_SCALE, VIEW_H, VIEW_W
from detector import YoloDetector
from tracker import MotionModel, Tracker


def run_sequence(seq_dir, det, track_thr, low):
    metas = [json.loads(l) for l in open(os.path.join(seq_dir, "meta.jsonl"))]
    metas = sorted({m["frame"]: m for m in metas}.values(), key=lambda m: m["frame"])
    motion, tracker = MotionModel(), Tracker()
    per_view = []  # (meta, img_path, [(track, det_box_view, conf)], ambiguous)
    for m in metas:
        p = os.path.join(seq_dir, m["file"])
        img = cv2.imread(p)
        if img is None:
            continue
        level, region = m["level"], m["region"]
        motion.observe(m["frame"], img, level, region)
        tracker.predict(m["frame"], motion.H)
        raw = det(img, level)
        s = LEVEL_SCALE[level]
        dets = []
        for d in raw:
            x1, y1, x2, y2 = d["box"]
            box = np.array([region[0] + x1 * s, region[1] + y1 * s, region[0] + x2 * s, region[1] + y2 * s])
            dets.append((box, d["probs"], float(d["conf"]), False))
        before = {id(t) for t in tracker.tracks}
        tracker.update(dets, level, region, m["frame"])
        # associate each detection to the track whose box it now defines (nearest centre)
        links = []
        ambiguous = False
        for d, (box, probs, conf, _) in zip(raw, dets):
            best, bd = None, 1e9
            c = (box[:2] + box[2:]) / 2
            for t in tracker.tracks:
                tc = (t.box[:2] + t.box[2:]) / 2
                dd = np.linalg.norm(tc - c)
                if dd < bd:
                    best, bd = t, dd
            size = max(box[2] - box[0], box[3] - box[1])
            if best is not None and bd < 0.5 * size + 4 * s:
                links.append((best, d["box"], conf))
            elif conf >= low:
                ambiguous = True
        per_view.append((m, p, links, ambiguous))
    return per_view, tracker


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rec", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--weights", default="weights/best.pt")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--track-thr", type=float, default=0.6, help="min hindsight track score to be a label")
    ap.add_argument("--low", type=float, default=0.15)
    a = ap.parse_args()
    det = YoloDetector(a.weights, imgsz=a.imgsz)
    os.makedirs(os.path.join(a.out, "images"), exist_ok=True)
    os.makedirs(os.path.join(a.out, "labels"), exist_ok=True)
    n_img = n_lab = 0
    for seq_dir in sorted(glob.glob(os.path.join(a.rec, "*"))):
        if not os.path.isfile(os.path.join(seq_dir, "meta.jsonl")) or "warmup" in seq_dir:
            continue
        per_view, _ = run_sequence(seq_dir, det, a.track_thr, a.low)
        # hindsight: tracks that ever reached a good score with a confident class
        for m, p, links, ambiguous in per_view:
            lines = []
            bad = ambiguous
            for t, vb, conf in links:
                cd = t.class_dist()
                good = t.score() >= a.track_thr and cd.max() >= 0.6 and t.hits >= 3
                if good:
                    x1, y1, x2, y2 = [float(v) for v in vb]
                    lines.append(f"{int(np.argmax(cd))} {(x1 + x2) / 2 / VIEW_W:.6f} {(y1 + y2) / 2 / VIEW_H:.6f} "
                                 f"{(x2 - x1) / VIEW_W:.6f} {(y2 - y1) / VIEW_H:.6f}")
                elif conf >= a.low:
                    bad = True
            if bad:
                continue
            name = f"{os.path.basename(seq_dir)[:12]}_{os.path.basename(p)[:-4]}"
            shutil.copy(p, os.path.join(a.out, "images", name + ".png"))
            with open(os.path.join(a.out, "labels", name + ".txt"), "w") as f:
                f.write("\n".join(lines) + ("\n" if lines else ""))
            n_img += 1
            n_lab += len(lines)
        print(seq_dir, "views", len(per_view), "kept", n_img, "labels", n_lab, flush=True)


if __name__ == "__main__":
    main()
