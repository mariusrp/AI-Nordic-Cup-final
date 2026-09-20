"""Exact 90-degree rotated copies of real recorded views (image + YOLO labels) so that, together with ultralytics'
fliplr/flipud, training covers all 8 dihedral orientations of every real render (a new city places the same 3D models
at new yaws; the synthetic cut-outs are rotated but do not transfer to real renders).

    python mk_rot90.py LIST.txt OUT_DIR          # LIST = image paths; writes OUT_DIR/images/train, OUT_DIR/labels/train
np.rot90 (k=1, counter-clockwise): (x, y) normalised -> (y, 1 - x); box (cx, cy, w, h) -> (cy, 1 - cx, h, w).
"""
import os
import sys

import cv2

lst, out = sys.argv[1], sys.argv[2]
os.makedirs(os.path.join(out, 'images', 'train'), exist_ok=True)
os.makedirs(os.path.join(out, 'labels', 'train'), exist_ok=True)
n = 0
for p in open(lst):
    p = p.strip()
    if not p:
        continue
    img = cv2.imread(p)
    if img is None:
        continue
    name = os.path.splitext(os.path.basename(p))[0]
    lp = p.replace('/images/', '/labels/').rsplit('.', 1)[0] + '.txt'
    rows = []
    if os.path.exists(lp):
        for line in open(lp):
            v = line.split()
            if len(v) < 5:
                continue
            c, cx, cy, w, h = v[0], float(v[1]), float(v[2]), float(v[3]), float(v[4])
            rows.append(f"{c} {cy:.6f} {1 - cx:.6f} {h:.6f} {w:.6f}")
    cv2.imwrite(os.path.join(out, 'images', 'train', name + '_r90.png'), cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE),
                [cv2.IMWRITE_PNG_COMPRESSION, 1])
    with open(os.path.join(out, 'labels', 'train', name + '_r90.txt'), 'w') as f:
        f.write('\n'.join(rows) + ('\n' if rows else ''))
    n += 1
print('rot90 copies', n, '->', out)
