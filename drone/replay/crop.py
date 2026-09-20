"""Native-resolution crops around (frame, source_x, source_y) from the recorded views. Usage:
python crop.py OUTDIR "name:frame:x:y:half" ... ; picks the finest level view containing the point, upscales L1/L0 back."""
import glob, os, re, sys
import cv2, numpy as np
OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True)
SC = {0: 4, 1: 2, 2: 1}
V = {}
for root in ('/workspace/drone_seen_full', '/workspace/drone_seen', '/workspace/.holdout/drone_seen_heldout'):
    for p in glob.glob(f'{root}/*/*.png'):
        b = os.path.basename(p)
        if 'local' in p or b in V: continue
        m = re.match(r'(\d+)_L(\d)_(\d+)_(\d+)\.png', b)
        if m: V[b] = (int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4)), p)
byf = {}
for f, l, cx, cy, p in V.values(): byf.setdefault(f, []).append((l, cx, cy, p))
for spec in sys.argv[2:]:
    name, fr, x, y, half = spec.split(':'); fr, x, y, half = int(fr), float(x), float(y), int(half)
    best = None
    for l, cx, cy, p in sorted(byf.get(fr, [])):
        s = SC[l]; rx, ry = cx - 480 * s, cy - 270 * s
        u, v = (x - rx) / s, (y - ry) / s
        if u - half / s < 0 or v - half / s < 0 or u + half / s > 959 or v + half / s > 539: continue
        if best is None or l > best[0]: best = (l, p, u, v, s, rx, ry)
    if best is None:
        print('NOVIEW', spec); continue
    l, p, u, v, s, rx, ry = best
    img = cv2.imread(p)
    h2 = half / s
    c = cv2.getRectSubPix(img, (int(2 * h2), int(2 * h2)), (u, v))
    z = 4 if s == 1 else (2 if s == 2 else 1)
    c = cv2.resize(c, None, fx=z * s, fy=z * s, interpolation=cv2.INTER_NEAREST)
    cv2.putText(c, f'{name} f{fr} L{l} src({int(x)},{int(y)}) half{half} zoom{z*s}x', (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
    cv2.putText(c, f'{name} f{fr} L{l} src({int(x)},{int(y)}) half{half} zoom{z*s}x', (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    fn = f'{OUT}/{name}.jpg'; cv2.imwrite(fn, c, [cv2.IMWRITE_JPEG_QUALITY, 95]); print('wrote', fn, 'L', l, c.shape)
print('CROP_DONE')
