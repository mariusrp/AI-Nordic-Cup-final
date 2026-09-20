"""Per-view box fusion shared by the FINALS scorers (eval_scene.py, fp_negatives.py).

Weighted box fusion (Solovyev et al., the recipe every MaCVi-2026 thermal-detection podium entry used) over the
output of several detectors on the SAME view. Kept next to detector.py's UnionDetector, which is the `union` mode:
pool both models' boxes and let the downstream class-wise NMS / merge_classes drop the duplicates.
"""
import numpy as np


def wbf(per_model, weights, iou_thr=0.55, mode='wbf'):
    """Weighted box fusion of the detections of ONE view. per_model: list (one entry per model) of
    [(bbox xyxy, class name, score)]; weights: per-model weight. Returns the fused list.

    Clusters are grown greedily in score order (the ZFTurbo WBF recipe), per class, against the running
    fused box. `wbf` applies the consensus rescale (a cluster missing a model keeps only that model's share
    of the weight), `avg` keeps the best score and only averages the box, `union` pools unchanged."""
    u = np.asarray(weights, float)
    u = u / u.sum()
    if mode == 'union':
        return [d for dets in per_model for d in dets]
    pool = [(b, c, s, mi) for mi, dets in enumerate(per_model) for b, c, s in dets]
    pool.sort(key=lambda d: -d[2] * u[d[3]])
    cl = []  # [fused box, class, {model: best score}, [member boxes/scores]]
    for b, c, s, mi in pool:
        hit = None
        for k in cl:
            if k[1] != c:
                continue
            a, q = k[0], b
            ix = max(0.0, min(a[2], q[2]) - max(a[0], q[0]))
            iy = max(0.0, min(a[3], q[3]) - max(a[1], q[1]))
            un = (a[2] - a[0]) * (a[3] - a[1]) + (q[2] - q[0]) * (q[3] - q[1]) - ix * iy
            if un > 0 and ix * iy / un > iou_thr:
                hit = k
                break
        if hit is None:
            cl.append([tuple(b), c, {mi: s}, [(np.asarray(b, float), s * u[mi])]])
        else:
            hit[2][mi] = max(hit[2].get(mi, 0.0), s)
            hit[3].append((np.asarray(b, float), s * u[mi]))
            W = np.array([w for _, w in hit[3]])
            B = np.stack([bb for bb, _ in hit[3]])
            hit[0] = tuple((B * W[:, None]).sum(0) / W.sum())
    out = []
    for box, c, best, _ in cl:
        if mode == 'avg':
            sc = max(best.values())
        else:
            sc = float(sum(u[m] * s for m, s in best.items()))
        out.append((box, c, float(min(1.0, sc))))
    return out
