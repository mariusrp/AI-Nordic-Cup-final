"""Offline A/B for the I4 DINOv2 prototype verifier (proto_verify.py) on top of the same v2 proposals.

Test fold X uses a bank built ONLY from the other block's frames (cut-outs, Helsinki backgrounds, Helsinki FPs).
Eval views (L1 and L2 only; the verifier passes L0 through):
  real : the 10 real Helsinki frames of the test block, tiled 2x2 (L1) / 4x4 (L2) like the evaluator.
         (v2 was trained on all 20 frames, so the proposals are optimistic here; only the re-scoring is A/B'd.)
  comp : N LoveDA tiles never used for the bank (g1b/loveda, train_0..699) with test-block cut-outs pasted at
         native source scale (rotation, flip, scale 0.7-1.4, colour jitter), rendered per level.
  seen : a recorded validation-city run NOT used in any bank (label-free): confident boxes per view.
Metric: AP@0.5, 101-pt, pooled over the views of one level, macro over classes with GT (objects < 50% inside
a view are ignore regions).  Proposals + crop embeddings are cached once, variants are re-scored from them.
    python eval_verify.py --fold B --bank /workspace/i4/bankA.pt
"""
import argparse
import glob
import json
import math
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CLASSES, CLS_INDEX, DRONE_UPSTREAM, LEVEL_SCALE, NC  # noqa: E402
import proto_verify as PV  # noqa: E402

SCENE = os.path.join(DRONE_UPSTREAM, "src", "helsinki")
BLOCKS = {"A": range(0, 10), "B": range(10, 20)}


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0.0


def view_gt(anns, t, s):
    """anns [(cls, box src)] -> (gt [(c, box view)], ignore [box view])."""
    gt, ign = [], []
    for c, b in anns:
        cb = (max(b[0], t[0]), max(b[1], t[1]), min(b[2], t[2]), min(b[3], t[3]))
        if cb[2] <= cb[0] or cb[3] <= cb[1]:
            continue
        frac = (cb[2] - cb[0]) * (cb[3] - cb[1]) / max(1e-6, (b[2] - b[0]) * (b[3] - b[1]))
        vb = ((cb[0] - t[0]) / s, (cb[1] - t[1]) / s, (cb[2] - t[0]) / s, (cb[3] - t[1]) / s)
        (gt if frac >= 0.5 else ign).append((CLS_INDEX[c], vb) if frac >= 0.5 else vb)
    return gt, ign


def ap_level(views, preds):
    """views: list of (gt, ign); preds: list of [(box, cls, score)] per view -> (macro AP, per class)."""
    per = {}
    npos = {}
    for (gt, ign), P in zip(views, preds):
        for c, _ in gt:
            npos[c] = npos.get(c, 0) + 1
        used = set()
        for b, c, s in sorted(P, key=lambda x: -x[2]):
            best, bj = 0.5, -1
            for j, (gc, gb) in enumerate(gt):
                if gc == c and j not in used:
                    v = iou(b, gb)
                    if v >= best:
                        best, bj = v, j
            if bj >= 0:
                used.add(bj); per.setdefault(c, []).append((s, 1))
            elif any(iou(b, g) >= 0.5 for g in ign):
                continue
            else:
                per.setdefault(c, []).append((s, 0))
    aps = {}
    for c, n in npos.items():
        lst = sorted(per.get(c, []), key=lambda x: -x[0])
        if not lst:
            aps[c] = 0.0; continue
        tp = np.cumsum([t for _, t in lst]); fp = np.cumsum([1 - t for _, t in lst])
        rec = tp / n; prec = tp / (tp + fp)
        prec = np.maximum.accumulate(prec[::-1])[::-1]
        aps[c] = float(np.mean([prec[rec >= r].max() if (rec >= r).any() else 0.0 for r in np.linspace(0, 1, 101)]))
    return (float(np.mean(list(aps.values()))) if aps else 0.0), aps


def build_views(fold, n_comp, seed):
    import make_synth as MS
    rng = np.random.default_rng(seed)
    views = []  # (kind, level, img, gt, ign)
    for f in BLOCKS[fold]:
        img = cv2.imread(os.path.join(SCENE, "images", f"frame_{f:06d}.png"))
        anns = [(x["object_id"], x["bbox"]) for x in json.load(open(os.path.join(SCENE, "annotations", f"frame_{f:06d}.json")))["annotations"]]
        for L in (1, 2):
            for t in PV.helsinki_tiles(L):
                g, ig = view_gt(anns, t, LEVEL_SCALE[L])
                views.append(("real", L, PV.render(img[t[1]:t[3], t[0]:t[2]], L), g, ig))
    src = "/workspace/g1b/synth_src_sam"
    meta = [m for m in json.load(open(os.path.join(src, "cutouts.json"))) if m["frame"] in BLOCKS[fold]]
    cuts = [(m["cls"], cv2.imread(os.path.join(src, "cutouts", m["name"] + ".png"), cv2.IMREAD_UNCHANGED), m["box"]) for m in meta]
    lov = sorted(glob.glob("/workspace/g1b/loveda/*.png"))
    for i in range(n_comp):
        L = 1 + i % 2
        canvas = PV.loveda_region(lov[int(rng.integers(len(lov)))], L, rng)
        anns, placed = [], []
        for _ in range(int(rng.integers(4, 11)) if L == 1 else int(rng.integers(1, 4))):
            c, rgba, box = cuts[int(rng.integers(len(cuts)))]
            o, b, _rot = MS.transform_cutout(rgba, box, rng, float(np.exp(rng.uniform(np.log(0.7), np.log(1.4)))))
            if o is None:
                continue
            o = np.dstack([MS.jitter_color(o[..., :3], rng, 0.7), o[..., 3]])
            oh, ow = o.shape[:2]
            if ow >= canvas.shape[1] or oh >= canvas.shape[0]:
                continue
            for _t in range(30):
                x = int(rng.integers(0, canvas.shape[1] - ow)); y = int(rng.integers(0, canvas.shape[0] - oh))
                gb = (x + b[0], y + b[1], x + b[2], y + b[3])
                if all(not (gb[0] < q[2] + 10 and q[0] < gb[2] + 10 and gb[1] < q[3] + 10 and q[1] < gb[3] + 10) for q in placed):
                    break
            else:
                continue
            MS.paste(canvas, o, x, y)
            placed.append(gb); anns.append((c, gb))
        H, W = canvas.shape[:2]
        g, ig = view_gt(anns, (0, 0, W, H), LEVEL_SCALE[L])
        views.append(("comp", L, PV.render(canvas, L), g, ig))
    return views


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", choices=["A", "B"], required=True)
    ap.add_argument("--bank", required=True)
    ap.add_argument("--weights", default="/workspace/drone_weights/v2.pt")
    ap.add_argument("--n-comp", type=int, default=240)
    ap.add_argument("--seen", default="/workspace/g13seen/5fd5807bfa504f53948ed293a2a3b4a5")
    ap.add_argument("--seed", type=int, default=4242)
    a = ap.parse_args()
    import torch
    from detector import YoloDetector
    det = YoloDetector(a.weights, conf=0.05)
    emb = PV.Embedder()
    bank = torch.load(a.bank, map_location="cuda")
    t0 = time.time()
    views = build_views(a.fold, a.n_comp, a.seed + (0 if a.fold == "A" else 1))
    seen = []
    for p in sorted(glob.glob(os.path.join(a.seen, "*_L[12]_*.png"))):
        seen.append(("seen", int(os.path.basename(p).split("_L")[1][0]), cv2.imread(p), [], []))
    print("views", len(views), "seen", len(seen), f"{time.time() - t0:.0f}s", flush=True)
    cache, ms = [], []
    for kind, L, img, g, ig in views + seen:
        dets = det(img, L)
        t1 = time.perf_counter()
        F = emb.embed([PV.crop_box(img, d["box"]) for d in dets]) if dets else None
        torch.cuda.synchronize(); ms.append(1000 * (time.perf_counter() - t1))
        cache.append((kind, L, dets, F, g, ig))
    print(f"embed ms/view p50 {np.median(ms):.1f} p90 {np.percentile(ms, 90):.1f} max {np.max(ms):.1f} "
          f"dets/view {np.mean([len(c[2]) for c in cache]):.1f}", flush=True)

    def diag(cfg):
        """On proposals that match a GT box (IoU >= .5, any class): YOLO vs prototype class accuracy, p_bg."""
        V = PV.Verifier(bank, emb=emb, **cfg)
        has = V.has.cpu().numpy()
        for kind in ("real", "comp"):
            st = {"n": 0, "yolo_ok": 0, "proto_ok": 0, "bg_drop_tp": 0, "fp": 0, "bg_drop_fp": 0}
            conf = {}
            for k, L, dets, F, g, ig in cache:
                if k != kind or not dets:
                    continue
                pb = V.probs(F).cpu().numpy()
                for d, p in zip(dets, pb):
                    m = [gc for gc, gb in g if iou(d["box"], gb) >= 0.5]
                    if not m:
                        st["fp"] += 1; st["bg_drop_fp"] += int(p[-1] > V.drop); continue
                    gc = m[0]
                    if not has[gc]:
                        continue
                    st["n"] += 1; st["bg_drop_tp"] += int(p[-1] > V.drop)
                    st["yolo_ok"] += int(np.argmax(d["probs"]) == gc)
                    pc = int(np.argmax(p[:NC] * has)); st["proto_ok"] += int(pc == gc)
                    if pc != gc:
                        conf[(CLASSES[gc][:8], CLASSES[pc][:8])] = conf.get((CLASSES[gc][:8], CLASSES[pc][:8]), 0) + 1
            print("DIAG", kind, cfg, st, "top confusions", sorted(conf.items(), key=lambda x: -x[1])[:6], flush=True)

    diag(dict(a=0.5, T=0.03, bg_key="bg"))

    def run(name, cfg, fuse=0.0):
        V = None
        if cfg is not None:
            V = PV.Verifier(bank, emb=emb, **cfg)
            has = V.has.cpu().numpy()
        res = {}
        allpred = []
        for kind, L, dets, F, g, ig in cache:
            P = []
            if dets:
                pb = V.probs(F).cpu().numpy() if V is not None else None
                for i, d in enumerate(dets):
                    yc = int(np.argmax(d["probs"])); ys = d["conf"] * d["probs"][yc]
                    if V is None:
                        P.append((d["box"], yc, ys)); continue
                    p = pb[i]
                    if p[-1] > V.drop:
                        continue
                    if not has[yc] and d["probs"][yc] >= 0.5:
                        P.append((d["box"], yc, ys)); continue
                    if V.mode == "keep":
                        P.append((d["box"], yc, d["conf"] ** V.a * p[yc] ** (1 - V.a) * d["probs"][yc])); continue
                    pc = p[:NC] * has
                    if fuse:  # class from YOLO probs x prototype probs (renormalised over classes)
                        q = pc / max(pc.sum(), 1e-9)
                        f = np.where(has, np.asarray(d["probs"]) ** fuse * np.maximum(q, 1e-6) ** (1 - fuse), np.asarray(d["probs"]))
                        c = int(np.argmax(f))
                        P.append((d["box"], c, d["conf"] ** V.a * (f[c] / f.sum() * (1 - p[-1])) ** (1 - V.a)))
                        continue
                    c = int(np.argmax(pc))
                    P.append((d["box"], c, d["conf"] ** V.a * p[c] ** (1 - V.a)))
            allpred.append(P)
        for kind in ("real", "comp"):
            for L in (1, 2):
                idx = [i for i, c in enumerate(cache) if c[0] == kind and c[1] == L]
                m, per = ap_level([(cache[i][4], cache[i][5]) for i in idx], [allpred[i] for i in idx])
                res[f"{kind}_L{L}"] = round(m, 4)
        sidx = [i for i, c in enumerate(cache) if c[0] == "seen"]
        for thr in (0.25, 0.5):
            res[f"seen@{thr}"] = round(float(np.mean([sum(s >= thr for _, _, s in allpred[i]) for i in sidx])), 2) if sidx else None
        res["real"] = round(np.mean([res["real_L1"], res["real_L2"]]), 4)
        res["comp"] = round(np.mean([res["comp_L1"], res["comp_L2"]]), 4)
        print("EVAL_VERIFY", json.dumps({"fold": a.fold, "variant": name, **res}), flush=True)
        return res

    run("v2_baseline", None)
    for A in (0.3, 0.5):
        for T in (0.02, 0.03, 0.05):
            run(f"keep a={A} T={T} bg", dict(a=A, T=T, drop=0.5))
    run("keep a=0.5 T=0.03 bg_with_seen", dict(a=0.5, T=0.03, drop=0.5, bg_key="bg_with_seen"))
    run("argmax a=0.5 T=0.03 bg", dict(a=0.5, T=0.03, drop=0.5, mode="argmax"))
    run("argmax+fuse.5 a=0.5 T=0.05 bg", dict(a=0.5, T=0.05, drop=0.5, mode="argmax"), fuse=0.5)

if __name__ == "__main__":
    main()
