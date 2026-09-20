"""Re-mask the Helsinki cut-outs with SAM (box prompt) so pasted objects carry no disc of Helsinki ground.
    python sam_masks.py data/synth_src data/synth_src_sam /root/dd/sam2.1_b.pt
Keeps the SAM mask when plausible (fills 12-95% of the box, spans >= 70% of it); else keeps the old
(grabcut) mask if it was grabcut; else a tight box-shaped soft mask (ellipse discs were a shortcut)."""
import json, os, shutil, sys
import cv2, numpy as np
from ultralytics import SAM

src, out, w = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(os.path.join(out, "cutouts"), exist_ok=True)
if not os.path.exists(os.path.join(out, "bg")):
    shutil.copytree(os.path.join(src, "bg"), os.path.join(out, "bg"))
meta = json.load(open(os.path.join(src, "cutouts.json")))
sam = SAM(w)


def check(mk, x1, y1, x2, y2):
    """Plausible object mask? Returns a quality score or None."""
    bw, bh = x2 - x1, y2 - y1
    inb = mk[y1:y2, x1:x2] > 0
    frac = inb.mean()
    if not (0.08 < frac < 0.92):
        return None
    ys, xs = np.where(inb)
    if (xs.max() - xs.min() + 1) < 0.75 * bw or (ys.max() - ys.min() + 1) < 0.75 * bh:
        return None
    outside = (mk > 0).sum() - inb.sum()
    if outside > 0.25 * bw * bh:
        return None
    c = inb[int(bh * 0.35):int(np.ceil(bh * 0.65)), int(bw * 0.35):int(np.ceil(bw * 0.65))]
    if c.size and c.mean() < 0.5:   # inverted (ground around the object)
        return None
    border = np.concatenate([inb[0], inb[-1], inb[:, 0], inb[:, -1]]).mean()
    if border > 0.5:                # ground patch filling the box
        return None
    return float(c.mean() if c.size else 0) - border - outside / (bw * bh)
mont, stats = [], {}
for m in meta:
    rgba = cv2.imread(os.path.join(src, "cutouts", m["name"] + ".png"), cv2.IMREAD_UNCHANGED)
    img = np.ascontiguousarray(rgba[..., :3])
    x1, y1, x2, y2 = m["box"]
    bw, bh = x2 - x1, y2 - y1
    f = 512.0 / max(img.shape[:2])
    big = cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC)
    mode = None
    cx, cy = (x1 + x2) / 2 * f, (y1 + y2) / 2 * f
    prompts = [dict(bboxes=[[x1 * f, y1 * f, x2 * f, y2 * f]]),
               dict(bboxes=[[x1 * f, y1 * f, x2 * f, y2 * f]], points=[[cx, cy]], labels=[1]),
               dict(points=[[cx, cy]], labels=[1])]
    best = None
    for pr in prompts:
        try:
            r = sam(big, verbose=False, **pr)[0]
        except Exception as e:
            continue
        if r.masks is None:
            continue
        for mk in r.masks.data.cpu().numpy():
            mk = cv2.resize(mk.astype(np.uint8) * 255, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
            mk = (mk > 127).astype(np.uint8) * 255
            q = check(mk, x1, y1, x2, y2)
            if q is not None and (best is None or q > best[0]):
                best = (q, mk)
    if best is not None:
        mk = best[1]
        mk[:y1] = 0; mk[y2:] = 0; mk[:, :x1] = 0; mk[:, x2:] = 0
        mode = "sam"
    if mode is None:
        if m["mode"] == "grabcut":
            mode = "grabcut"
            mk = None
        else:
            mode = "box"
            mk = np.zeros(img.shape[:2], np.uint8)
            mk[y1:y2, x1:x2] = 255
    if mk is not None:
        mk = cv2.dilate(mk, np.ones((3, 3), np.uint8)) if mode == "sam" else mk
        rgba = np.dstack([img, cv2.GaussianBlur(mk, (3, 3), 0)])
    cv2.imwrite(os.path.join(out, "cutouts", m["name"] + ".png"), rgba)
    m["mode"] = mode
    stats[(m["cls"], mode)] = stats.get((m["cls"], mode), 0) + 1
    if len(mont) < 48 and m["name"].endswith(("_00_" + m["cls"],)) or (len(mont) < 48 and np.random.rand() < 0.2):
        a = rgba[..., 3:4].astype(np.float32) / 255
        t = (rgba[..., :3] * a + np.array([255, 0, 255]) * (1 - a)).astype(np.uint8)
        t = cv2.resize(t, (128, 128), interpolation=cv2.INTER_NEAREST)
        cv2.putText(t, f"{m['cls'][:10]} {mode}", (2, 10), 0, 0.33, (0, 0, 0), 1)
        mont.append(t)
json.dump(meta, open(os.path.join(out, "cutouts.json"), "w"), indent=0)
print(sorted(stats.items()))
while len(mont) % 8:
    mont.append(np.zeros((128, 128, 3), np.uint8))
cv2.imwrite(os.path.join(out, "montage.jpg"), np.vstack([np.hstack(mont[i:i + 8]) for i in range(0, len(mont), 8)]))
