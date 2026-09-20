"""Fuse offline multi-model detections of every recorded view into one per-frame answer table for the deterministic
validation flight (frames 1-249). Each detection (source px, frame t) is mapped to CANONICAL coordinates (t*, x*) =
the fractional frame at which its centre crosses the image centre line y=1080 and its x there, by iterating the flight
homography H (<= ~20 steps, so H errors never chain over the whole flight; frame-0 coordinates squash frames > 150 onto
the horizon). Greedy clustering in (x*, v*t*) with a size-dependent radius gives one cluster per physical object.
Per cluster: support score from per-view evidence, class vote (model x level weights), box trajectory = weighted median
box in a reference frame + linear residual (parallax of tall objects), propagated with H to every frame where it is
inside the image. Output: {"frame": [[cls, [x1,y1,x2,y2] as fractions of 3840x2160, conf], ...]} ranked by conf, with the
runner-up class appended at lower conf (COCO AP is per class; a wrong-class TP costs nothing at lower rank).
Usage: python fuse.py H.npy DETS_DIR[:DETS_DIR2...] OUT_TABLE.json OUT_CLUSTERS.json [--models ft_all,ft2,...] [--overrides ov.json]
       [--levelw 0.5,0.85,1.0] (class-vote/trajectory weight per view level) [--box med|l2] (box extent: weighted median over
       all members, or over the non-truncated level-2 = native-resolution members when a cluster has them)
       [--runnerup 3] (classes emitted per cluster: 1 = top class only) [--maxbox 300] (boxes kept per frame)
       [--imgszw 960=1,1280=1,1920=1,2560=1] (weight per inference size) [--l2support 0] (support bonus for L2 evidence)
       [--framemap 239=238] (stalled/duplicate flight frames: treat detections of that view as the earlier frame)
       [--unjudged 1.0] (support multiplier for clusters that no override touched, i.e. not eye-checked; {"t","x","keep": true}
       marks a checked cluster without changing it)
v2 (19 Sep): several dets dirs (the survey runs' native L2 views live in their own dirs), overrides may ADD manual
clusters ({"add": true, "cls": c, "t": frame, "box": [x1,y1,x2,y2] source px, "support": s}).
"""
import glob
import json
import os
import sys

import numpy as np

H = np.load(sys.argv[1]); DETS = sys.argv[2].split(':'); OUT = sys.argv[3]; OUTC = sys.argv[4]
argv = sys.argv[5:]


def opt(name, default):
    return argv[argv.index(name) + 1] if name in argv else default


MODELS = opt('--models', '').split(',') if opt('--models', '') else None
OVR = json.load(open(opt('--overrides', ''))) if opt('--overrides', '') else {}
MINCONF = float(opt('--minconf', 0.01))
SIZES = json.load(open(opt('--sizes', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'class_sizes.json'))))  # per-class (w, h) at the centre line, from the eye-verified 1-150 clusters (+ Helsinki for absent classes)
RATIO = float(opt('--ratio', 0.5))  # weight of the detection-ratio factor in the support (0 = off)
YC = 1080.0
W, FH = 3840.0, 2160.0
MODEL_W = dict(ft_all=1.0, ft_m=1.0, ft2=0.8, dft=0.8, wise085=0.9, v2=0.6, r11=0.35)      # class-vote weights
SUPPORT_W = dict(ft_all=1.0, ft_m=1.0, ft2=0.9, dft=0.9, wise085=1.0, v2=0.25, r11=0.25)  # existence-evidence weights (v2/r11 fire high-conf junk on this city)
LEVEL_W = dict(zip((0, 1, 2), map(float, opt('--levelw', '0.5,0.85,1.0').split(','))))
BOXRULE = opt('--box', 'med')
NRUN = int(opt('--runnerup', 3))
MAXBOX = int(opt('--maxbox', 300))
IMGSZ_W = {int(k): float(v) for k, v in (kv.split('=') for kv in opt('--imgszw', '').split(',') if kv)}
L2SUP = float(opt('--l2support', 0.0))
# stalled frames: the flight emitted the same image twice (frame_steps.py; 239 = a duplicate of 238 in this flight), so
# detections made on that view belong to the earlier frame's pose. --framemap 239=238 remaps them before clustering.
FRAMEMAP = {int(a): int(b) for a, b in (kv.split('=') for kv in opt('--framemap', '').split(',') if kv)}
Hi = np.linalg.inv(H)
_pw = {}


def mpow(k):
    if k not in _pw:
        _pw[k] = np.linalg.matrix_power(H, k) if k >= 0 else np.linalg.matrix_power(Hi, -k)
    return _pw[k]


def warp_pts(M, pts):
    ph = np.hstack([pts, np.ones((len(pts), 1))]) @ M.T
    return ph[:, :2] / ph[:, 2:3]


def warp_box(M, box):
    x1, y1, x2, y2 = box
    cx, cy, w, h = (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1
    p = warp_pts(M, np.array([[cx, cy], [cx - w / 2, cy], [cx + w / 2, cy], [cx, cy - h / 2], [cx, cy + h / 2]]))
    nw = np.linalg.norm(p[2] - p[1]); nh = np.linalg.norm(p[4] - p[3])
    return np.array([p[0][0] - nw / 2, p[0][1] - nh / 2, p[0][0] + nw / 2, p[0][1] + nh / 2])


# ---------------------------------------------------------------- load detections
D = []  # dicts
for fp in sorted(f for dd in DETS for f in glob.glob(f'{dd}/*.jsonl')):
    tag = os.path.basename(fp)[:-6]
    parts = tag.split('_')
    flip = parts[-1] == 'flip'
    if flip: parts = parts[:-1]
    imgsz = int(parts[-1]); model = '_'.join(parts[:-1])
    if MODELS and model not in MODELS: continue
    if model not in MODEL_W: print('unknown model weight', model); MODEL_W[model] = 0.7; SUPPORT_W[model] = 0.7
    for line in open(fp):
        r = json.loads(line)
        rx1, ry1, rx2, ry2 = r['region']; s = {0: 4, 1: 2, 2: 1}[r['level']]
        fr = FRAMEMAP.get(r['frame'], r['frame'])
        for c, x1, y1, x2, y2, cf in r['dets']:
            if cf < MINCONF: continue
            trunc = x1 < rx1 + 2 * s or y1 < ry1 + 2 * s or x2 > rx2 - 2 * s or y2 > ry2 - 2 * s
            D.append(dict(t=fr, view=r['file'], level=r['level'], model=model, imgsz=imgsz, cls=c, box=[x1, y1, x2, y2],
                          conf=cf * IMGSZ_W.get(imgsz, 1.0), trunc=trunc))
print('dets', len(D), 'models', sorted({d['model'] for d in D}), flush=True)
n = len(D)
T = np.array([d['t'] for d in D], float)
B = np.array([d['box'] for d in D], float)
C = np.stack([(B[:, 0] + B[:, 2]) / 2, (B[:, 1] + B[:, 3]) / 2], 1)

# ---------------------------------------------------------------- canonical coordinates (centre-line crossing)
ts = np.full(n, np.nan); xs = np.full(n, np.nan); ws = np.full(n, np.nan); hs = np.full(n, np.nan)
for direction in (+1, -1):
    M = H if direction > 0 else Hi
    active = (C[:, 1] < YC) if direction > 0 else (C[:, 1] > YC)
    idx = np.where(active)[0]
    P = np.hstack([C[idx], B[idx, :2], B[idx, 2:]])  # centre, corner1, corner2
    k = 0
    while len(idx) and k < 80:
        Q = np.hstack([warp_pts(M, P[:, 0:2]), warp_pts(M, P[:, 2:4]), warp_pts(M, P[:, 4:6])])
        crossed = (Q[:, 1] >= YC) if direction > 0 else (Q[:, 1] <= YC)
        if crossed.any():
            j = np.where(crossed)[0]
            fr = (YC - P[j, 1]) / (Q[j, 1] - P[j, 1])
            R = P[j] + fr[:, None] * (Q[j] - P[j])
            ts[idx[j]] = T[idx[j]] + direction * (k + fr)
            xs[idx[j]] = R[:, 0]; ws[idx[j]] = R[:, 4] - R[:, 2]; hs[idx[j]] = R[:, 5] - R[:, 3]
        keep = ~crossed
        idx = idx[keep]; P = Q[keep]; k += 1
same = np.isclose(C[:, 1], YC)
ts[same] = T[same]; xs[same] = C[same, 0]; ws[same] = B[same, 2] - B[same, 0]; hs[same] = B[same, 3] - B[same, 1]
ok = np.isfinite(ts)
print('canonical ok', ok.sum(), 'of', n, flush=True)
VC = float(np.linalg.norm(warp_pts(H, np.array([[1920.0, YC]]))[0] - [1920.0, YC]))  # px per frame at the centre line
print('centre-line flow px/frame', round(VC, 2))

# ---------------------------------------------------------------- greedy clustering in (x*, VC*t*)
W_I = np.array([d['conf'] * MODEL_W[d['model']] * LEVEL_W[d['level']] for d in D])
order = np.argsort(-W_I)
cl_seed = []  # (x*, y*=VC t*, radius)
assign = np.full(n, -1)
SX = np.empty((0,)); SY = np.empty((0,)); SR = np.empty((0,))
members = []
for i in order:
    if not ok[i]: continue
    x, y = xs[i], VC * ts[i]
    if len(SX):
        d2 = (SX - x) ** 2 + (SY - y) ** 2
        j = int(np.argmin(d2 / SR ** 2))
        if d2[j] < SR[j] ** 2:
            assign[i] = j; members[j].append(i); continue
    r = float(np.clip(0.5 * np.sqrt(max(ws[i] * hs[i], 1.0)), 22.0, 80.0))
    SX = np.append(SX, x); SY = np.append(SY, y); SR = np.append(SR, r)
    assign[i] = len(members); members.append([i])
print('clusters', len(members), flush=True)
# post-merge in canonical space: box_i = (x* +- w*/2, VC t* +- h*/2) per cluster (median over members); IoU > 0.4 -> merge into the stronger
cb = []
for mem in members:
    m = np.array(mem); x, y, w, h = np.median(xs[m]), VC * np.median(ts[m]), np.median(ws[m]), np.median(hs[m])
    cb.append([x - w / 2, y - h / 2, x + w / 2, y + h / 2])
cb = np.array(cb); merged = 0; alive = np.ones(len(members), bool)
for i in range(len(members)):  # members are in creation (= weighted-conf) order: i is stronger than j > i
    if not alive[i]: continue
    ix1 = np.maximum(cb[i, 0], cb[i + 1:, 0]); iy1 = np.maximum(cb[i, 1], cb[i + 1:, 1])
    ix2 = np.minimum(cb[i, 2], cb[i + 1:, 2]); iy2 = np.minimum(cb[i, 3], cb[i + 1:, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    ai = (cb[i, 2] - cb[i, 0]) * (cb[i, 3] - cb[i, 1]); aj = (cb[i + 1:, 2] - cb[i + 1:, 0]) * (cb[i + 1:, 3] - cb[i + 1:, 1])
    iou = inter / np.maximum(ai + aj - inter, 1e-9)
    for j in np.where((iou > 0.4) & alive[i + 1:])[0] + i + 1:
        members[i].extend(members[j]); members[j] = []; alive[j] = False; merged += 1
members = [m for m in members if m]
print('post-merge', merged, '-> clusters', len(members), flush=True)


def wmed(v, w):
    o = np.argsort(v); c = np.cumsum(w[o]); return float(v[o][np.searchsorted(c, c[-1] / 2)])


# ---------------------------------------------------------------- per cluster: support, class vote, trajectory model
VIEWS = {}  # frame -> list of (file, level, region)
_vseen = set()
for fp in [sorted(glob.glob(f'{dd}/*.jsonl'))[0] for dd in DETS]:
    for line in open(fp):
        r = json.loads(line)
        if r['level'] >= 1 and r['file'] not in _vseen:
            _vseen.add(r['file']); VIEWS.setdefault(r['frame'], []).append((r['file'], r['level'], r['region']))
clusters = []
for ci, mem in enumerate(members):
    mem = np.array(mem)
    ds = [D[i] for i in mem]
    # per-view evidence: best conf over models (support), model-level weighted class votes
    ev = {}
    vote = {}
    fam = {'ft': {}, 'v2': {}}  # class votes of the fine-tuned family vs the synthetic-trained family (v2, r11)
    for d in ds:
        ev[d['view']] = max(ev.get(d['view'], 0.0), d['conf'] * SUPPORT_W[d['model']])
        w = d['conf'] * MODEL_W[d['model']] * LEVEL_W[d['level']]
        vote[d['cls']] = vote.get(d['cls'], 0.0) + w
        fv = fam['v2' if d['model'] in ('v2', 'r11') else 'ft']
        fv[d['cls']] = fv.get(d['cls'], 0.0) + w
    e = sorted(ev.values(), reverse=True)
    nv = len(e); nf = len({d['t'] for d in ds})
    top3 = float(np.mean(e[:3]))
    support = top3 * (1 - 0.5 ** nv) * (1 if nf >= 2 else 0.6)
    vs = sum(vote.values()); pdist = {k: v / vs for k, v in sorted(vote.items(), key=lambda kv: -kv[1])}
    # trajectory: reference frame = weighted median of member frames
    wts = np.array([d['conf'] * MODEL_W[d['model']] * LEVEL_W[d['level']] * (0.15 if d['trunc'] else 1.0) for d in ds])
    tt = np.array([d['t'] for d in ds], float)
    tref = int(round(wmed(tt, wts)))
    Bref = np.array([warp_box(mpow(tref - d['t']), d['box']) for d in ds])
    box_ref = np.array([wmed(Bref[:, k], wts) for k in range(4)])
    l2 = np.array([d['level'] == 2 and not d['trunc'] and d['conf'] * SUPPORT_W[d['model']] >= 0.1 for d in ds])
    n_l2 = int(l2.sum())
    if BOXRULE == 'l2' and n_l2 >= 2:
        wl, hl = wmed(Bref[l2, 2] - Bref[l2, 0], wts[l2]), wmed(Bref[l2, 3] - Bref[l2, 1], wts[l2])
        cx0, cy0 = (box_ref[0] + box_ref[2]) / 2, (box_ref[1] + box_ref[3]) / 2
        box_ref = np.array([cx0 - wl / 2, cy0 - hl / 2, cx0 + wl / 2, cy0 + hl / 2])
    cen = (Bref[:, :2] + Bref[:, 2:]) / 2 - (box_ref[:2] + box_ref[2:]) / 2
    a = np.zeros(2); b = np.zeros(2)
    span = tt.max() - tt.min()
    if span >= 6 and len(set(tt)) >= 4:
        dt = tt - tref
        for k in range(2):
            r = cen[:, k]; mad = np.median(np.abs(r - np.median(r))) + 1e-6
            good = np.abs(r - np.median(r)) < 4 * mad + 2
            if good.sum() >= 4:
                A = np.stack([np.ones(good.sum()), dt[good]], 1); sw = np.sqrt(wts[good])
                sol = np.linalg.lstsq(A * sw[:, None], r[good] * sw, rcond=None)[0]
                a[k], b[k] = sol[0], float(np.clip(sol[1], -3, 3))
    # detection ratio over the views that contained the (propagated) box: real objects are re-detected, clutter is sporadic
    # (level-aware: only views at least as fine as the coarsest level that detected it count as coverage)
    cover = hit = 0
    lv_det = [d['level'] for d in ds if d['conf'] * SUPPORT_W[d['model']] >= 0.1] or [d['level'] for d in ds]
    min_level = min(lv_det)
    for t in range(max(1, int(tt.min()) - 2), min(249, int(tt.max()) + 2) + 1):
        dt = t - tref; sh = a + b * dt
        bx = warp_box(mpow(dt), box_ref + np.array([sh[0], sh[1], sh[0], sh[1]]))
        for vf, vl, (rx1, ry1, rx2, ry2) in VIEWS.get(t, []):
            if vl < min_level: continue
            m = 3 * (2 if vl == 1 else 1)
            if bx[0] >= rx1 + m and bx[1] >= ry1 + m and bx[2] <= rx2 - m and bx[3] <= ry2 - m:
                cover += 1; hit += ev.get(vf, 0.0) >= 0.1
    ratio = hit / cover if cover else 0.5
    if RATIO > 0: support *= (1 - RATIO) + RATIO * ratio
    if L2SUP > 0 and n_l2:
        e2 = max(ev.get(d['view'], 0.0) for d in ds if d['level'] == 2)
        support = min(1.0, support + L2SUP * e2 * (1 - support))
    clusters.append(dict(id=ci, n=len(ds), nviews=nv, nframes=nf, cover=cover, hit=hit, t_star=float(np.median(ts[mem])), x_star=float(np.median(xs[mem])),
                         w_star=float(np.median(ws[mem])), h_star=float(np.median(hs[mem])), tmin=int(tt.min()), tmax=int(tt.max()),
                         tref=tref, box_ref=box_ref.tolist(), resid=[a.tolist(), b.tolist()], support=support, top3=top3,
                         vote={k: round(v, 3) for k, v in pdist.items()}, cls=next(iter(pdist)),
                         models=sorted({d['model'] for d in ds}), levels=sorted({d['level'] for d in ds}),
                         fam={k: (max(v, key=v.get), round(max(v.values()) / sum(v.values()), 2), round(sum(v.values()), 2)) for k, v in fam.items() if v},
                         best=max(ds, key=lambda d: d['conf'])['view'], n_l2=n_l2))
# manual overrides (eyeballed crop sheets), keyed by canonical position {"t": t_star, "x": x_star, ...} so they survive
# a re-fusion with another H or detection set (matched to the nearest cluster within 45 px of (x*, VC t*))
CT = np.array([[c['x_star'], VC * c['t_star']] for c in clusters]) if clusters else np.zeros((0, 2))
n_ovr = 0
for o in (OVR if isinstance(OVR, list) else [dict(v, id=int(k)) for k, v in OVR.items()]):
    if o.get('add'):  # manual cluster: an eye-found object the detectors never fired on (box at frame t, source px)
        t0 = int(o['t']); bx = np.array(o['box'], float)
        clusters.append(dict(id=len(clusters), n=0, nviews=0, nframes=0, cover=0, hit=0, t_star=float(t0), x_star=float((bx[0] + bx[2]) / 2),
                             w_star=float(bx[2] - bx[0]), h_star=float(bx[3] - bx[1]), tmin=int(o.get('tmin', 1)), tmax=int(o.get('tmax', 249)),
                             tref=t0, box_ref=bx.tolist(), resid=[[0, 0], [0, 0]], support=float(o.get('support', 0.5)), top3=0,
                             vote=o.get('vote', {o['cls']: 1.0}), cls=o['cls'], eye='vote' not in o, manual=True, judged=True, fixed=True,
                             note=o.get('note', '')))
        n_ovr += 1; continue
    if 'id' in o: c = clusters[o['id']]
    else:
        d = np.hypot(CT[:, 0] - o['x'], CT[:, 1] - VC * o['t']); j = int(np.argmin(d))
        if d[j] > 45: print('override unmatched', o); continue
        c = clusters[j]
    n_ovr += 1; c['judged'] = True
    if o.get('drop'): c['drop'] = True
    if 'vote' in o: c['vote'] = dict(sorted(o['vote'].items(), key=lambda kv: -kv[1])); c['cls'] = next(iter(c['vote'])); c['eye'] = False
    elif 'cls' in o: c['cls'] = o['cls']; c['vote'] = {o['cls']: 1.0}; c['eye'] = True
    if 'support' in o: c['support'] = o['support']; c['fixed'] = True  # an eye-set support is final (no size prior)
print('overrides applied', n_ovr)
# class-size prior: the eye-verified objects of a class have a known size at the centre line (source px); a cluster whose
# canonical box area is < 0.3x (skylights, 7-13 px dots voted ta-ta/small_launcher) or > 4x that is demoted, never dropped
n_small = n_big = 0
for c in clusters:
    if c.get('eye') or c.get('drop') or c.get('fixed') or c['cls'] not in SIZES: continue
    rw, rh = SIZES[c['cls']]; ratio = (c['w_star'] * c['h_star']) / (rw * rh)
    c['size_ratio'] = round(ratio, 2)
    if ratio < 0.3: c['support'] *= 0.35; n_small += 1
    elif ratio > 4: c['support'] *= 0.5; n_big += 1
print('size prior demoted small', n_small, 'big', n_big)
UNJ = float(opt('--unjudged', 1.0))  # support multiplier for clusters no override touched (eye-unverified); 1 = off
if UNJ != 1.0:
    for c in clusters:
        if not c.get('judged'): c['support'] *= UNJ
    print('unjudged clusters x', UNJ, sum(1 for c in clusters if not c.get('judged')))

# ---------------------------------------------------------------- emit per-frame table
table = {str(t): [] for t in range(1, 250)}
nbox = 0
for c in clusters:
    if c.get('drop'): continue
    tref = c['tref']; a, b = np.array(c['resid'][0]), np.array(c['resid'][1])
    pd = list(c['vote'].items())
    for t in range(1, 250):
        dt = t - tref
        if abs(dt) > 60: continue
        sh = a + b * dt
        br = np.array(c['box_ref']) + np.array([sh[0], sh[1], sh[0], sh[1]])
        bx = warp_box(mpow(dt), br)
        cx1, cy1, cx2, cy2 = max(0.0, bx[0]), max(0.0, bx[1]), min(W, bx[2]), min(FH, bx[3])
        if cx2 - cx1 < 4 or cy2 - cy1 < 4 or bx[2] - bx[0] > 1200: continue
        vis = (cx2 - cx1) * (cy2 - cy1) / max(1.0, (bx[2] - bx[0]) * (bx[3] - bx[1]))
        dist = 0 if c['tmin'] <= t <= c['tmax'] else min(abs(t - c['tmin']), abs(t - c['tmax']))
        conf = c['support'] * (1.0 / (1.0 + dist / 30.0)) * (1.0 if vis >= 0.6 else vis / 0.6)
        nb = [round(cx1 / W, 6), round(cy1 / FH, 6), round(cx2 / W, 6), round(cy2 / FH, 6)]
        for rank, (cls, p) in enumerate(pd[:NRUN]):
            cc = conf * (p if rank == 0 else 0.7 * p) if not c.get('eye') else conf * (1.0 if rank == 0 else 0.0)
            if cc < 0.002: continue
            table[str(t)].append([cls, nb, round(float(cc), 5), c['id']]); nbox += 1
for t in table:
    table[t].sort(key=lambda r: -r[2]); table[t] = table[t][:MAXBOX]
if opt('--ids', ''):  # debug: cluster id of every table row (same order)
    json.dump({t: [r[3] for r in v] for t, v in table.items()}, open(opt('--ids', ''), 'w'))
table = {t: [r[:3] for r in v] for t, v in table.items()}
json.dump(table, open(OUT, 'w'))
json.dump(clusters, open(OUTC, 'w'), indent=0)
print('table boxes', sum(len(v) for v in table.values()), 'max/frame', max(len(v) for v in table.values()),
      'clusters emitted', sum(1 for c in clusters if not c.get('drop')))
