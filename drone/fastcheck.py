"""Compare DRONE_FAST path vs ultralytics predict on recorded views (boxes + timing)."""
import glob, sys, time
import cv2, numpy as np
import common  # noqa
from detector import YoloDetector
w, seq = sys.argv[1], sys.argv[2]
files = sorted(glob.glob(seq + "/*.png"))[::20]
d = YoloDetector(w)
imgs = [cv2.imread(f) for f in files]
for mode in (True, False):
    d.fast = mode
    t = time.perf_counter(); res = [d(im, 1) for im in imgs]; dt = (time.perf_counter() - t) / len(imgs)
    print("fast" if mode else "slow", f"{1000*dt:.1f} ms", [len(r) for r in res][:10], [round(max([x['conf'] for x in r] or [0]), 3) for r in res][:10])
