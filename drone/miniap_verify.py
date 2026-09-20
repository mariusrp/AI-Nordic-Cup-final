"""Verifier on I3's eyeball-labelled validation tiles (run 5fd5807b; labels in miniap_tiles.py on drone-evolve-g1-2).
Replays the run exactly like crops_sheet.py (I3 code in /workspace/g13/drone), and for every labelled tile embeds
the track's box in that view (proto_verify.crop_box) and asks the verifier for p(track class) and p(background).
Scores: baseline = tile conf; verified = conf^a * p_cls^(1-a), 0 if p_bg > drop.  Mini-AP as in miniap_tiles.py.
Bank must not contain run 5fd5807b views (bank 'bg_with_seen' uses runs 1062106c/a2e63304: same city = optimistic).
    python miniap_verify.py RUN_DIR tiles.json bank.pt
"""
import json, os, sys
sys.path.insert(0, '/workspace/g13/drone')
os.environ.setdefault('DRONE_WEIGHTS', '/workspace/drone_weights/v2.pt')
import cv2, numpy as np
import replay_rescore as R
from common import LEVEL_SCALE, CLASSES, NC
from detector import build_detector
from tracker import MotionModel, Tracker
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import proto_verify as PV  # noqa: E402

TP = {12, 25, 29, 39, 45, 209, 66, 83, 90, 91, 11, 16, 24, 343, 404}
UNSURE = {42, 308, 384, 390, 365, 226, 19, 22, 44}
run, tiles_json, bank = sys.argv[1:4]
lab = {(r['id'], r['frame'], r['level']): r for r in json.load(open(tiles_json)) if r['id'] not in UNSURE}
det = build_detector()
metas = sorted([json.loads(l) for l in open(os.path.join(run, 'meta.jsonl'))], key=lambda m: (m['frame'], m['t']))
motion, tracker = MotionModel(), Tracker()
crops, keys, seen = [], [], {}
for m in metas:
    img = cv2.imread(os.path.join(run, m['file']))
    if img is None:
        continue
    level, region, frame = m['level'], tuple(int(v) for v in m['region']), m['frame']
    motion.observe(frame, img, level, region); tracker.predict(frame, motion.H)
    tracker.update(R.to_source(det(img, level), level, region), level, region, frame)
    if level == 0:
        continue
    s = LEVEL_SCALE[level]
    for t in tracker.tracks:
        b = t.box; cd = t.class_dist(); c = int(np.argmax(cd))
        if t.score() * cd[c] < 0.4 or t.last_seen != frame:
            continue
        x1, y1, x2, y2 = [(b[0] - region[0]) / s, (b[1] - region[1]) / s, (b[2] - region[0]) / s, (b[3] - region[1]) / s]
        if x1 < 0 or y1 < 0 or x2 > 960 or y2 > 540:
            continue
        if t.id in seen and seen[t.id] >= level:
            continue
        seen[t.id] = level
        if (t.id, frame, level) in lab:
            crops.append(PV.crop_box(img, (x1, y1, x2, y2))); keys.append((t.id, frame, level))
rows = [lab[k] for k in keys]
print('matched tiles', len(rows), 'of', len(lab), 'TP', sum(r['id'] in TP for r in rows), flush=True)
emb = PV.Embedder()
F = emb.embed(crops)


def ap(scored):
    by = {}
    for r, s in scored:
        by.setdefault(r['cls'], []).append((s, r['id'] in TP))
    aps = {}
    for c, lst in by.items():
        npos = sum(t for _, t in lst)
        if npos == 0:
            continue
        lst.sort(key=lambda x: -x[0])
        tp = np.cumsum([t for _, t in lst]); fp = np.cumsum([not t for _, t in lst])
        prec = tp / (tp + fp); rec = tp / npos
        aps[c] = np.mean([prec[rec >= t].max() if (rec >= t).any() else 0 for t in np.linspace(0, 1, 101)])
    return np.mean(list(aps.values())), aps


m, aps = ap([(r, r['conf']) for r in rows])
print(f"{'baseline':34s} miniAP {m:.3f} kept {len(rows)} TPkept {sum(r['id'] in TP for r in rows)}")
for bg_key in ('bg',):
    for A in (0.3, 0.5, 0.7):
        for T in (0.01, 0.02, 0.03, 0.05):
            V = PV.Verifier(bank, emb=emb, a=A, T=T, bg_key=bg_key)
            P = V.probs(F).cpu().numpy()
            has = V.has.cpu().numpy()
            sc, kept, tpk, agree = [], 0, 0, 0
            for r, p in zip(rows, P):
                c = CLASSES.index(r['cls'])
                if p[-1] > V.drop:
                    sc.append((r, 0.0)); continue
                kept += 1; tpk += r['id'] in TP
                if not has[c]:
                    sc.append((r, r['conf'])); continue
                agree += int(np.argmax(p[:NC] * has)) == c
                sc.append((r, r['conf'] ** A * p[c] ** (1 - A)))
            m, aps = ap(sc)
            m2, _ = ap([(r, 0.0 if p[-1] > V.drop else r['conf']) for r, p in zip(rows, P)])
            m3, _ = ap([(r, r['conf'] * (1 - p[-1])) for r, p in zip(rows, P)])
            print(f"  bg-drop only miniAP {m2:.3f}   conf*(1-p_bg) miniAP {m3:.3f}")
            print(f"verify a={A} T={T} {bg_key:13s} miniAP {m:.3f} kept {kept} TPkept {tpk} class_agree {agree}  "
                  + " ".join(f"{c[:8]}:{v:.2f}" for c, v in sorted(aps.items())), flush=True)
