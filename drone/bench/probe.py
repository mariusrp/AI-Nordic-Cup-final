"""External health/latency probe of a drone endpoint (run from the Mac): posts N real Helsinki views (frame 0.. at L0,
then the L1 view the server asks for, rendered like local_evaluator.render_view) and prints the round-trip time,
the number of annotations and the requested view. A fresh sequence_id per run, so the live server's session state for
any concurrent flight is untouched (never run this while the FINAL flight is in progress).

    python3 drone/bench/probe.py http://194.68.245.26:22174/predict [N=8]
"""
import base64
import json
import os
import sys
import time
import urllib.request
import uuid

import cv2
import numpy as np

URL = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 8
HERE = os.path.dirname(os.path.abspath(__file__))
SCENE = os.environ.get("HEL", os.path.join(HERE, "..", "..", "..", "nac-n3-tmp", "upstream", "drone-flyby", "src", "helsinki", "images"))
REG = {0: (3840, 2160), 1: (1920, 1080), 2: (960, 540)}
CONS = {"maximum_center_delta": 2203.0, "allowed_resolution_levels": [0, 1], "full_view_reset_exempt_from_delta": True,
        "center_bounds": [{"resolution_level": 0, "width": 960, "height": 540, "minimum_center_x": 1920, "maximum_center_x": 1920,
                           "minimum_center_y": 1080, "maximum_center_y": 1080},
                          {"resolution_level": 1, "width": 960, "height": 540, "minimum_center_x": 960, "maximum_center_x": 2880,
                           "minimum_center_y": 540, "maximum_center_y": 1620},
                          {"resolution_level": 2, "width": 960, "height": 540, "minimum_center_x": 480, "maximum_center_x": 3360,
                           "minimum_center_y": 270, "maximum_center_y": 1890}]}


def view(img4k, level, cx, cy):
    rw, rh = REG[level]
    x1, y1 = int(cx - rw // 2), int(cy - rh // 2)
    crop = img4k[y1:y1 + rh, x1:x1 + rw]
    if level < 2:
        crop = cv2.resize(crop, (960, 540), interpolation=cv2.INTER_AREA)
    ok, png = cv2.imencode(".png", crop)
    return base64.b64encode(png.tobytes()).decode(), [x1, y1, x1 + rw, y1 + rh]


seq = "probe-" + uuid.uuid4().hex[:8]
level, cx, cy = 0, 1920, 1080
for i in range(N):
    img = cv2.imread(os.path.join(SCENE, f"frame_{i:06d}.png"))
    if img is None:
        print("no frame", i); break
    b64, region = view(img, level, cx, cy)
    body = {"sequence_id": seq, "frame": i, "frame_index": i, "request_id": f"{seq}:{i}", "frame_interval_ms": 333,
            "response_timeout_ms": 3333, "original_width": 3840, "original_height": 2160,
            "view": {"resolution_level": level, "center_x": cx, "center_y": cy, "view_id": f"{seq}:{i}", "image": b64,
                     "image_media_type": "image/png", "width": 960, "height": 540, "source_region_xyxy": region},
            "camera_constraints": CONS, "camera_command_feedback": None}
    data = json.dumps(body).encode()
    t = time.perf_counter()
    try:
        r = urllib.request.Request(URL, data=data, headers={"Content-Type": "application/json"})
        resp = json.loads(urllib.request.urlopen(r, timeout=10).read())
    except Exception as e:
        print(f"frame {i}: ERROR {e}"); break
    ms = 1000 * (time.perf_counter() - t)
    rv = resp.get("requested_view")
    top = sorted(resp["annotations"], key=lambda a: -a["confidence"])[:3]
    print(f"frame {i} L{level}({cx},{cy}) {len(data) / 1e6:.2f} MB -> {ms:6.0f} ms  {len(resp['annotations']):3d} ann  "
          f"next={None if rv is None else (rv['resolution_level'], rv['center_x'], rv['center_y'])}  "
          f"top={[(a['object_id'], round(a['confidence'], 2)) for a in top]}")
    if rv is not None:
        level, cx, cy = rv["resolution_level"], rv["center_x"], rv["center_y"]
