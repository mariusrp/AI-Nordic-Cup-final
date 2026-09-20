"""Draw raw detector output on recorded views.  python viz_views.py <seq_dir> <out.jpg> [weights] [n] [conf]"""
import glob, os, sys
import cv2, numpy as np
import common  # noqa
from common import CLASSES
from detector import YoloDetector
seq, out = sys.argv[1], sys.argv[2]
w = sys.argv[3] if len(sys.argv) > 3 else "weights/best.pt"
n = int(sys.argv[4]) if len(sys.argv) > 4 else 6
thr = float(sys.argv[5]) if len(sys.argv) > 5 else 0.2
files = sorted(glob.glob(os.path.join(seq, "*.png")))
files = [files[i] for i in np.linspace(0, len(files) - 1, n).astype(int)]
det = YoloDetector(w)
tiles = []
for f in files:
    img = cv2.imread(f)
    L = int(os.path.basename(f).split("_L")[1][0])
    for d in det(img, L):
        if d["conf"] < thr:
            continue
        x1, y1, x2, y2 = map(int, d["box"])
        c = int(np.argmax(d["probs"]))
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 1)
        cv2.putText(img, f"{CLASSES[c][:8]} {d['conf']:.2f}", (x1, max(10, y1 - 2)), 0, 0.4, (0, 255, 255), 1)
    cv2.putText(img, os.path.basename(f), (5, 20), 0, 0.6, (255, 255, 255), 2)
    tiles.append(img)
while len(tiles) % 2:
    tiles.append(np.zeros_like(tiles[0]))
cv2.imwrite(out, np.vstack([np.hstack(tiles[i:i + 2]) for i in range(0, len(tiles), 2)]), [cv2.IMWRITE_JPEG_QUALITY, 85])
