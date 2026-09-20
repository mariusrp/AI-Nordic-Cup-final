"""By-eye check of valcity labels against Helsinki references (drone understanding lane, cycle 2).

For every labelled valcity object, crop it from recorded validation views (frames <= 150 ONLY; L2 preferred, else L1)
and place it next to Helsinki reference crops of the class it is labelled as, shown at the SAME scale (view px).
Writes one JPEG per object group into OUT. Crops are upscaled x4 (nearest) so small objects are visible.
Usage: python eye_sheet.py SEEN_DIR VALCITY_SCENE_DIR HELSINKI_DIR OUT [run_id ...]
"""
import sys, os, json, glob, collections
import cv2
import numpy as np

seen, vc, hel, out = sys.argv[1:5]
runs = sys.argv[5:]
os.makedirs(out, exist_ok=True)
UP = 4
WIN = 40  # half window in VIEW px around the object centre


def views():
    for r in runs:
        for line in open(os.path.join(seen, r, 'meta.jsonl')):
            m = json.loads(line)
            if m['frame'] > 150:
                continue  # never touch frames >= 151
            p = os.path.join(seen, r, m['file'])
            if os.path.exists(p):
                yield r, m, p


GT = {}
for f in sorted(glob.glob(os.path.join(vc, 'annotations', '*.json'))):
    d = json.load(open(f))
    if d['frame'] <= 150:
        GT[d['frame']] = d['annotations']

cands = collections.defaultdict(list)  # cluster -> [(level, frame, run, path, region, bbox)]
for r, m, p in views():
    reg = m['region']; lv = m['level']
    for g in GT.get(m['frame'], []):
        b = g['bbox']
        if b[0] >= reg[0] and b[1] >= reg[1] and b[2] <= reg[2] and b[3] <= reg[3]:
            cands[(g['valcity_cluster'], g['object_id'])].append((lv, m['frame'], r, p, reg, b))


def crop_view(img, reg, b, lv):
    s = 960.0 / (reg[2] - reg[0])
    cx, cy = ((b[0] + b[2]) / 2 - reg[0]) * s, ((b[1] + b[3]) / 2 - reg[1]) * s
    x1, y1 = int(cx - WIN), int(cy - WIN)
    pad = cv2.copyMakeBorder(img, WIN, WIN, WIN, WIN, cv2.BORDER_CONSTANT, value=(40, 40, 40))
    c = pad[y1 + WIN:y1 + 3 * WIN, x1 + WIN:x1 + 3 * WIN].copy()
    bx1, by1 = int((b[0] - reg[0]) * s - x1), int((b[1] - reg[1]) * s - y1)
    bx2, by2 = int((b[2] - reg[0]) * s - x1), int((b[3] - reg[1]) * s - y1)
    c = cv2.resize(c, None, fx=UP, fy=UP, interpolation=cv2.INTER_NEAREST)
    cv2.rectangle(c, (bx1 * UP, by1 * UP), (bx2 * UP, by2 * UP), (0, 255, 255), 1)
    return c


def label(c, t):
    c = c.copy()
    cv2.putText(c, t, (3, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3)
    cv2.putText(c, t, (3, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    return c


# Helsinki references per class, at L1 scale (source/2) and L2 scale (source/1)
href = collections.defaultdict(list)
for f in sorted(glob.glob(os.path.join(hel, 'annotations', '*.json')))[::3]:
    fr = int(os.path.basename(f).split('_')[-1][:6])
    if fr >= 20:
        continue  # Helsinki frames 20-24 are the hidden holdout (some pod copies still hold them): never use
    img = cv2.imread(os.path.join(hel, 'images', f'frame_{fr:06d}.png'))
    for a in json.load(open(f))['annotations']:
        b = a['bbox']
        for lv, sc in ((1, 2), (2, 1)):
            small = img if sc == 1 else None
            cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
            reg = [cx - 480 * sc, cy - 270 * sc, cx + 480 * sc, cy + 270 * sc]
            x1, y1 = int(max(0, cx - WIN * sc)), int(max(0, cy - WIN * sc))
            x2, y2 = int(min(3840, cx + WIN * sc)), int(min(2160, cy + WIN * sc))
            c = img[y1:y2, x1:x2]
            c = cv2.resize(c, ((x2 - x1) // sc, (y2 - y1) // sc), interpolation=cv2.INTER_AREA)
            c = cv2.copyMakeBorder(c, 0, max(0, 2 * WIN - c.shape[0]), 0, max(0, 2 * WIN - c.shape[1]), cv2.BORDER_CONSTANT, value=(40, 40, 40))[:2 * WIN, :2 * WIN]
            bb = [int((b[0] - x1) / sc), int((b[1] - y1) / sc), int((b[2] - x1) / sc), int((b[3] - y1) / sc)]
            c = cv2.resize(c, None, fx=UP, fy=UP, interpolation=cv2.INTER_NEAREST)
            cv2.rectangle(c, (bb[0] * UP, bb[1] * UP), (bb[2] * UP, bb[3] * UP), (0, 255, 0), 1)
            href[(a['object_id'], lv)].append(label(c, f'HEL {a["object_id"][:10]} L{lv} f{fr}'))

cache = {}
rows_by_cls = collections.defaultdict(list)
for (cid, cls), lst in sorted(cands.items(), key=lambda x: x[0][1]):
    lst.sort(key=lambda x: (-x[0], x[1]))
    pick = [x for x in lst if x[0] == 2][:3]
    l1 = [x for x in lst if x[0] == 1]
    if l1:
        idx = np.linspace(0, len(l1) - 1, min(4, len(l1))).astype(int)
        pick += [l1[i] for i in idx]
    tiles = []
    for lv, fr, r, p, reg, b in pick[:6]:
        if p not in cache:
            cache[p] = cv2.imread(p)
        tiles.append(label(crop_view(cache[p], reg, b, lv), f'#{cid} {cls[:10]} L{lv} f{fr} {r[:4]}'))
    tiles += (href.get((cls, 2), [])[:1] + href.get((cls, 1), [])[:1])
    if tiles:
        rows_by_cls[cls].append(np.hstack(tiles + [np.zeros_like(tiles[0])] * (8 - len(tiles))))
    print(f'#{cid} {cls}: {len(lst)} in-view crops (L2 {sum(1 for x in lst if x[0] == 2)}, L1 {len(l1)})')

for cls, rows in rows_by_cls.items():
    cv2.imwrite(os.path.join(out, f'eye_{cls}.jpg'), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88])
# reference sheet of all 16 Helsinki classes at L1 and L2 scale
allref = []
for cls in sorted({k[0] for k in href}):
    t = href[(cls, 2)][:2] + href[(cls, 1)][:2]
    allref.append(np.hstack(t + [np.zeros_like(t[0])] * (4 - len(t))))
for i in range(0, len(allref), 8):
    cv2.imwrite(os.path.join(out, f'helsinki_ref_{i // 8}.jpg'), np.vstack(allref[i:i + 8]), [cv2.IMWRITE_JPEG_QUALITY, 88])
