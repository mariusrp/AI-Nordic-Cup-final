"""Table-to-table post-processor: apply the organisers' annotation conventions (read off the 25 Helsinki reference frames,
same generator as the validation flight) to a fuse.py answer table. Independent of fuse.py: it re-derives each row's
unclipped box from the cluster file + flight homography, so it works on any table that fuse.py wrote with --ids.

Helsinki conventions (upstream drone-flyby/src/helsinki, 16 objects x 25 frames, notes in the session scratchpad
results/drone_conventions.md):
  (a) GT box = the object's FULL (projected 3D) box CLIPPED to the image; an object is annotated in every frame where that
      box overlaps the image by >= 1 px (observed slivers 6, 8, 9, 13 px; condor f10 181x39 and medium_launcher f23 9x46
      contain almost no object pixels). Clipping is to the last pixel index: x2 <= 3839, y2 <= 2159.
  (b) boxes are LOOSER than the rendered pixels (GT/tight 1.1-1.2 for jammer, mine_roller, spacecraft, ta-ta; 1.25-1.6
      for tank, large_launcher, planes; 2.2 x 2.7 for small_launcher); no cast shadows are rendered.
  (c)/(d) no occluded object in Helsinki; the only 'invisible' annotations are the edge slivers of (a).

Transforms (each can be switched off):
  clip    x2 <= 3839/3840, y2 <= 2159/2160 (fuse clips at 3840/2160).
  edge    fuse.py ranks a border-crossing box by conf *= vis/0.6 (vis = visible AREA FRACTION). Under (a) the GT box is the
          clipped full box, so whether our box is a TP depends on the visible EXTENT in the clipped dimension (v px) versus
          the box-edge error (propagation p90 1.2 px near the top edge; box-convention error a few px), not on the fraction:
          conf = conf_unpenalised * erf(v / (2 sqrt2 sigma)) / erf(h / (2 sqrt2 sigma)) (h = full extent), sigma = --edge-sigma px (default 4). A 22 px sliver of a
          133 px hangar goes from x0.28 to x1.0; a 4 px sliver from x0.11 to x0.38.
  slivers add the 1-3 px border slivers fuse.py drops (< 4 px) for clusters with support >= --sliver-min-support, top class
          only, same erf factor (replaces the frame's lowest rows only when it outranks them; --maxbox cap kept).
  mbox    (off by default) scale the boxes of MANUAL clusters (eye-measured TIGHT boxes on native L2 views) about their
          centre by per-class factors, e.g. --mbox large_launcher=1.12x1.19 (= sqrt of Helsinki GT/tight 1.25 x 1.42:
          IoU >= 0.75 whether the GT is tight or Helsinki-loose).
Usage: python conventions.py TABLE_IN IDS.json CLUSTERS.json H.npy TABLE_OUT [--no-clip] [--no-edge] [--no-slivers]
       [--edge-sigma 4] [--fuse-vis 0.6] [--sliver-min-support 0.3] [--mbox cls=SXxSY,...] [--maxbox N (default: input max)]
       [--report report.json]
"""
import json
import math
import sys

import numpy as np

W, FH = 3840.0, 2160.0
XMAX, YMAX = 3839.0 / W, 2159.0 / FH

argv = sys.argv[1:]
TIN, IDS, CLS_P, HP, TOUT = argv[:5]
opts = argv[5:]


def opt(name, default):
    return opts[opts.index(name) + 1] if name in opts else default


DO_CLIP = '--no-clip' not in opts
DO_EDGE = '--no-edge' not in opts
DO_SLIV = '--no-slivers' not in opts
SIGMA = float(opt('--edge-sigma', 4.0))
FUSE_VIS = float(opt('--fuse-vis', 0.6))  # fuse.py emit: conf *= 1 if vis >= 0.6 else vis / 0.6
SLIV_MIN = float(opt('--sliver-min-support', 0.3))
MBOX = {}
for kv in [s for s in opt('--mbox', '').split(',') if s]:
    k, v = kv.split('='); sx, sy = v.split('x'); MBOX[k] = (float(sx), float(sy))

table = json.load(open(TIN)); ids = json.load(open(IDS)); clusters = json.load(open(CLS_P))
C = {c['id']: c for c in clusters}
H = np.load(HP); Hi = np.linalg.inv(H)
MAXBOX = int(opt('--maxbox', max(len(v) for v in table.values())))
_pw = {}


def mpow(k):
    if k not in _pw:
        _pw[k] = np.linalg.matrix_power(H, k) if k >= 0 else np.linalg.matrix_power(Hi, -k)
    return _pw[k]


def warp_box(M, box):  # identical to fuse.py
    x1, y1, x2, y2 = box
    cx, cy, w, h = (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1
    p = np.array([[cx, cy], [cx - w / 2, cy], [cx + w / 2, cy], [cx, cy - h / 2], [cx, cy + h / 2]])
    ph = np.hstack([p, np.ones((5, 1))]) @ M.T; p = ph[:, :2] / ph[:, 2:3]
    nw = np.linalg.norm(p[2] - p[1]); nh = np.linalg.norm(p[4] - p[3])
    return np.array([p[0][0] - nw / 2, p[0][1] - nh / 2, p[0][0] + nw / 2, p[0][1] + nh / 2])


def full_box(c, t):  # fuse.py emit geometry: box_ref + linear residual, propagated with H^(t - tref)
    dt = t - c['tref']; a, b = np.array(c['resid'][0]), np.array(c['resid'][1]); sh = a + b * dt
    return warp_box(mpow(dt), np.array(c['box_ref']) + np.array([sh[0], sh[1], sh[0], sh[1]]))


def clipped(bx):
    return max(0.0, bx[0]), max(0.0, bx[1]), min(W, bx[2]), min(FH, bx[3])


def _p(v):  # P(|edge error| < v/2) for a N(0, SIGMA) edge error
    return math.erf(max(0.0, v) / (2 * math.sqrt(2) * SIGMA))


def edge_factor(bx):
    """Per clipped dimension: P(hit | visible extent v) / P(hit | full extent) (1.0 for an interior box; the full-extent
    denominator keeps small objects from paying twice: their interior boxes carry the same edge error)."""
    x1, y1, x2, y2 = clipped(bx); f = 1.0
    if bx[0] < 0 or bx[2] > W: f *= _p(x2 - x1) / max(1e-6, _p(bx[2] - bx[0]))
    if bx[1] < 0 or bx[3] > FH: f *= _p(y2 - y1) / max(1e-6, _p(bx[3] - bx[1]))
    return f


def fuse_pen(bx):
    x1, y1, x2, y2 = clipped(bx)
    vis = (x2 - x1) * (y2 - y1) / max(1.0, (bx[2] - bx[0]) * (bx[3] - bx[1]))
    return 1.0 if vis >= FUSE_VIS else vis / FUSE_VIS


def scaled(bx, s):
    cx, cy, w, h = (bx[0] + bx[2]) / 2, (bx[1] + bx[3]) / 2, (bx[2] - bx[0]) * s[0], (bx[3] - bx[1]) * s[1]
    return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])


def norm(x1, y1, x2, y2):
    b = [x1 / W, y1 / FH, x2 / W, y2 / FH]
    if DO_CLIP: b = [max(0.0, b[0]), max(0.0, b[1]), min(XMAX, b[2]), min(YMAX, b[3])]
    return [round(v, 6) for v in b]


rep = dict(rows_in=0, edge_rows=0, edge_up=0, edge_down=0, clip_changed=0, geom_mismatch=0, mbox_rows=0, slivers_added=0,
           slivers_rejected_cap=0, rows_cut_by_cap=0, edge_by_class={}, examples=[])
out = {}
present = {}  # (cid, t) pairs already in the table
for t, rows in table.items():
    rid = ids[t]; T = int(t); new = []
    for r, cid in zip(rows, rid):
        rep['rows_in'] += 1
        cls, box, conf = r[0], list(r[1]), float(r[2])
        c = C.get(cid); present[(cid, T)] = True
        if c is None: new.append([cls, box, conf]); continue
        bx = full_box(c, T)
        cb = clipped(bx)
        if max(abs(a - b) for a, b in zip([cb[0] / W, cb[1] / FH, cb[2] / W, cb[3] / FH], box)) > 2e-5:
            rep['geom_mismatch'] += 1; new.append([cls, box, conf]); continue  # not fuse geometry: leave untouched
        is_edge = bx[0] < 0 or bx[1] < 0 or bx[2] > W or bx[3] > FH
        if DO_EDGE and is_edge:
            rep['edge_rows'] += 1
            nc = conf / fuse_pen(bx) * edge_factor(bx)
            rep['edge_up' if nc > conf + 1e-9 else 'edge_down'] += 1
            ec = rep['edge_by_class'].setdefault(cls, [0, 0]); ec[0] += 1; ec[1] += int(nc > conf + 1e-9)
            if len(rep['examples']) < 40 and c['support'] >= 0.5:
                x1, y1, x2, y2 = cb
                rep['examples'].append(f"t{T} {cls} cid{cid} vis {x2 - x1:.0f}x{y2 - y1:.0f} of {bx[2] - bx[0]:.0f}x{bx[3] - bx[1]:.0f}: conf {conf:.3f} -> {nc:.3f}")
            conf = nc
        if MBOX and c.get('manual') and c['cls'] in MBOX:
            bx = scaled(bx, MBOX[c['cls']]); rep['mbox_rows'] += 1
            if DO_EDGE:  # re-rank by the new visible extent (the scaled box may now cross the border)
                conf = conf / edge_factor(full_box(c, T)) * edge_factor(bx) if edge_factor(full_box(c, T)) > 0 else conf
            cb = clipped(bx)
            if cb[2] - cb[0] < 1 or cb[3] - cb[1] < 1: continue
        nb = norm(*cb)
        if nb != [round(v, 6) for v in box]: rep['clip_changed'] += 1
        new.append([cls, nb, round(float(conf), 5)])
    out[t] = new

if DO_SLIV:  # border slivers < 4 px that fuse.py dropped
    for c in clusters:
        if c.get('drop') or c['support'] < SLIV_MIN: continue
        for T in range(1, 250):
            dt = T - c['tref']
            if abs(dt) > 60 or (c['id'], T) in present: continue
            bx = full_box(c, T)
            if MBOX and c.get('manual') and c['cls'] in MBOX: bx = scaled(bx, MBOX[c['cls']])
            x1, y1, x2, y2 = clipped(bx)
            if x2 - x1 < 1 or y2 - y1 < 1 or bx[2] - bx[0] > 1200: continue
            if x2 - x1 >= 4 and y2 - y1 >= 4: continue  # fuse emitted (or cut) it: not a sliver
            dist = 0 if c['tmin'] <= T <= c['tmax'] else min(abs(T - c['tmin']), abs(T - c['tmax']))
            conf = c['support'] / (1.0 + dist / 30.0) * edge_factor(bx)
            rows = out[str(T)]
            if len(rows) >= MAXBOX and conf <= min(r[2] for r in rows):
                rep['slivers_rejected_cap'] += 1; continue
            rows.append([c["cls"], norm(x1, y1, x2, y2), round(float(conf), 5)]); rep['slivers_added'] += 1
            if len(rep['examples']) < 60:
                rep['examples'].append(f"+sliver t{T} {c['cls']} cid{c['id']} vis {x2 - x1:.1f}x{y2 - y1:.1f} conf {conf:.3f}")

for t in out:
    out[t].sort(key=lambda r: -r[2])
    rep['rows_cut_by_cap'] += max(0, len(out[t]) - MAXBOX)
    out[t] = out[t][:MAXBOX]
json.dump(out, open(TOUT, 'w'))
rep['rows_out'] = sum(len(v) for v in out.values())
print(json.dumps({k: v for k, v in rep.items() if k != 'examples'}))
for e in rep['examples']: print('  ', e)
if opt('--report', ''): json.dump(rep, open(opt('--report', ''), 'w'), indent=1)
