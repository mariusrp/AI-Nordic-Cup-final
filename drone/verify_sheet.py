"""By-eye check: v2 boxes (conf >= .25) on recorded L1/L2 views, with the verifier's verdict.
Green = kept (new score), red = dropped as background.   python verify_sheet.py RUN_DIR bank.pt out.jpg"""
import glob, os, sys
import cv2, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CLASSES, NC  # noqa: E402
from detector import YoloDetector  # noqa: E402
import proto_verify as PV  # noqa: E402

run, bank, out = sys.argv[1:4]
det = YoloDetector('/workspace/drone_weights/v2.pt', conf=0.05)
V = PV.Verifier(bank)
tiles = []
for p in sorted(glob.glob(os.path.join(run, '*_L[12]_*.png')))[::3]:
    L = int(os.path.basename(p).split('_L')[1][0]); img = cv2.imread(p)
    ds = [d for d in det(img, L) if d['conf'] * d['probs'].max() >= 0.25]
    if not ds:
        continue
    P = V.probs(V.emb.embed([PV.crop_box(img, d['box']) for d in ds])).cpu().numpy()
    for d, pr in zip(ds, P):
        yc = int(np.argmax(d['probs'])); old = d['conf'] * d['probs'][yc]
        new = 0.0 if pr[-1] > V.drop else d['conf'] ** V.a * pr[yc] ** (1 - V.a) * d['probs'][yc]
        c = cv2.resize(PV.crop_box(img, d['box'], ctx=2.0), (128, 128), interpolation=cv2.INTER_CUBIC)
        col = (0, 0, 255) if pr[-1] > V.drop else (0, 255, 0)
        cv2.rectangle(c, (0, 0), (127, 127), col, 2)
        cv2.putText(c, f"L{L} {CLASSES[yc][:9]}", (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 255), 1)
        cv2.putText(c, f"{old:.2f}>{new:.2f} bg{pr[-1]:.2f}", (3, 124), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 255), 1)
        tiles.append((old, c))
tiles = [t for _, t in sorted(tiles, key=lambda x: -x[0])][:160]
while len(tiles) % 16:
    tiles.append(np.zeros_like(tiles[0]))
cv2.imwrite(out, cv2.vconcat([cv2.hconcat(tiles[i:i + 16]) for i in range(0, len(tiles), 16)]), [cv2.IMWRITE_JPEG_QUALITY, 85])
print('tiles', len(tiles))
