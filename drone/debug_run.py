"""Per-object/per-frame diagnosis of a recorded local run (recorder meta.jsonl vs local GT).

    python debug_run.py <record_dir>/<sequence_id>
Prints, per GT object, one char per frame:  '#' matched with right class (IoU>=.5),
'c' box ok but wrong top class, 'b' right class but IoU<.5, '.' missing, ' ' not in frame.
"""
import json
import os
import sys

import numpy as np

import common  # noqa
from common import DRONE_UPSTREAM


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i + 1e-9)


def main(d, scene="helsinki"):
    metas = {}
    for line in open(os.path.join(d, "meta.jsonl")):
        m = json.loads(line)
        metas[m["frame"]] = m
    adir = os.path.join(DRONE_UPSTREAM, "src", scene, "annotations")
    gt = {}
    for f in sorted(os.listdir(adir)):
        g = json.load(open(os.path.join(adir, f)))
        gt[g["frame"]] = g["annotations"]
    frames = sorted(gt)
    objs = sorted({a["object_id"] for f in frames for a in gt[f]})
    print("frame      " + "".join(str(f % 10) for f in frames))
    print("level      " + "".join(str(metas[f]["level"]) if f in metas else "-" for f in frames))
    print("views", [(metas[f]["level"], metas[f]["cx"], metas[f]["cy"]) for f in frames if f in metas])
    fps = []
    for o in objs:
        row = ""
        for f in frames:
            g = [a for a in gt[f] if a["object_id"] == o]
            if not g:
                row += " "
                continue
            gb = g[0]["bbox"]
            m = metas.get(f)
            if m is None:
                row += "x"
                continue
            preds = [(c, [b[0] * 3840, b[1] * 2160, b[2] * 3840, b[3] * 2160], s) for c, b, s in m["ann"]]
            same = [(iou(gb, b), s) for c, b, s in preds if c == o]
            anyc = [iou(gb, b) for c, b, s in preds]
            if same and max(same)[0] >= 0.5:
                row += "#"
            elif anyc and max(anyc) >= 0.5:
                row += "c"
            elif same and max(same)[0] > 0:
                row += "b"
            else:
                row += "."
        print(f"{o[:10]:10s} {row}")
    # false positives with high confidence
    for f in frames:
        m = metas.get(f)
        if not m:
            continue
        for c, b, s in m["ann"]:
            bb = [b[0] * 3840, b[1] * 2160, b[2] * 3840, b[3] * 2160]
            if s > 0.3 and not any(iou(bb, a["bbox"]) >= 0.5 and a["object_id"] == c for a in gt[f]):
                fps.append((f, c, round(s, 2), [int(v) for v in bb]))
    print("high-conf FPs:", len(fps))
    for x in fps[:15]:
        print("  ", x)


if __name__ == "__main__":
    main(sys.argv[1])
