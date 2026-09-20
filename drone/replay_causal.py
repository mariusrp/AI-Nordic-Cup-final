"""Offline causal replay of a recorded flight (DRONE_RECORD_DIR layout: <seq>/meta.jsonl + PNG views) through the
full server pipeline (register -> track -> detect(+verifier) -> SAFER answer), open loop: the recorded view sequence
is replayed as-is, the planner's requests are ignored. Registration is synchronous for determinism.
Writes one jsonl line per frame {frame, ann} in the recorder's normalised format, readable by valcity/replay_boot.py.

  DRONE_RECORD_DIR= DRONE_WEIGHTS=... DRONE_VERIFY=... python replay_causal.py /workspace/drone_seen/<seq> out.jsonl [max_frame]
Also prints per-level FP proxies: detector boxes per view (after the verifier) with conf >= 0.25.
"""
import base64
import json
import os
import sys
import time

os.environ["DRONE_RECORD_DIR"] = ""            # never append to the recordings
os.environ.setdefault("DRONE_ASYNC_AFTER_FITS", "1000000")

import numpy as np  # noqa: E402

import server  # noqa: E402  (loads detector + verifier, warms up)
from dtos import DroneFlybyPredictRequestDto  # noqa: E402

src, out = sys.argv[1], sys.argv[2]
max_frame = int(sys.argv[3]) if len(sys.argv) > 3 else 150
P = server.PREDICTOR
metas = [json.loads(l) for l in open(os.path.join(src, "meta.jsonl"))]
metas = sorted([m for m in metas if m["frame"] <= max_frame], key=lambda m: m["frame"])
cons = {"maximum_center_delta": 2203.0, "allowed_resolution_levels": [0, 1, 2], "full_view_reset_exempt_from_delta": True,
        "center_bounds": [{"resolution_level": 0, "width": 960, "height": 540, "minimum_center_x": 1920, "maximum_center_x": 1920,
                           "minimum_center_y": 1080, "maximum_center_y": 1080}]}
# count detector outputs per view (post verifier) by wrapping the detector
stats = {0: [], 1: [], 2: []}
det = P.detector


class Wrap:
    def __init__(self, d):
        self.d = d

    def __call__(self, img, level):
        r = self.d(img, level)
        stats[level].append(sum(1 for x in r if x["conf"] >= 0.25))
        return r

    def __getattr__(self, k):
        return getattr(self.d, k)


P.detector = Wrap(det)
seq = os.path.basename(src.rstrip("/"))
ts = []
with open(out, "w") as fo:
    for m in metas:
        png = open(os.path.join(src, m["file"]), "rb").read()
        req = DroneFlybyPredictRequestDto.model_validate({
            "sequence_id": seq, "frame": m["frame"], "frame_index": m["frame_index"], "request_id": f"r{m['frame']}",
            "frame_interval_ms": 333, "response_timeout_ms": 3333, "original_width": 3840, "original_height": 2160,
            "view": {"resolution_level": m["level"], "center_x": m["cx"], "center_y": m["cy"], "view_id": f"v{m['frame']}",
                     "image": base64.b64encode(png).decode(), "image_media_type": "image/png", "width": 960, "height": 540,
                     "source_region_xyxy": m["region"]},
            "camera_constraints": cons, "camera_command_feedback": None})
        t = time.perf_counter()
        resp, _ = P._predict(req)
        ts.append(1000 * (time.perf_counter() - t))
        fo.write(json.dumps(dict(frame=m["frame"], level=m["level"],
                                 ann=[[a.object_id, [round(float(c), 6) for c in a.bbox], round(float(a.confidence), 4)]
                                      for a in resp.annotations])) + "\n")
print(f"{seq} frames={len(metas)} ms p50={np.median(ts):.1f} p90={np.percentile(ts, 90):.1f} " +
      " ".join(f"L{k}: views={len(v)} det@0.25/view={np.mean(v) if v else float('nan'):.2f}" for k, v in stats.items()))
