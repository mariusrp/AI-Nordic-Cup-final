"""Paired block-bootstrap of valcity replay scores with the UNMODIFIED upstream local_evaluator.score (frames are
resampled in blocks of B frames; resampled frames get virtual ids by wrapping frame_numbers/load_annotations in this
process only; scoring code is untouched). Prints mean/se per run and paired differences vs the first run.
Usage: python replay_boot.py UPSTREAM_DIR SCENE NBOOT B name1=meta1.jsonl name2=meta2.jsonl ..."""
import sys, json, os, numpy as np
up, scene, NB, B = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
runs = [a.split('=', 1) for a in sys.argv[5:]]
sys.path.insert(0, up); os.chdir(up)
import local_evaluator as le  # noqa: E402
real_frames = le.frame_numbers(scene); real_load = le.load_annotations
GT = {f: real_load(f, scene) for f in real_frames}
P = {}
for name, mp in runs:
    d = {}
    for line in open(mp):
        r = json.loads(line)
        d[r['frame']] = [dict(object_id=c, bbox=tuple(v * (3840 if i % 2 == 0 else 2160) for i, v in enumerate(b)), confidence=s)
                         for c, b, s in r['ann']]
    P[name] = d


def score_on(fr):
    vids = list(range(1, len(fr) + 1))
    le.frame_numbers = lambda s: vids
    le.load_annotations = lambda v, s: GT[fr[v - 1]]
    out = {}
    for name, _ in runs:
        preds = {v: P[name][f] for v, f in zip(vids, fr) if f in P[name]}
        out[name] = le.score(scene, preds)[0]
    return out


full = score_on(list(real_frames))
blocks = [real_frames[i:i + B] for i in range(0, len(real_frames), B)]
rng = np.random.default_rng(0); S = {n: [] for n, _ in runs}
for _ in range(NB):
    fr = [f for k in rng.integers(0, len(blocks), len(blocks)) for f in blocks[k]]
    try: o = score_on(fr)
    except ValueError: continue
    for n in o: S[n].append(o[n])
base = runs[0][0]
for n, _ in runs:
    s = np.array(S[n]); d = s - np.array(S[base])
    print(f'{n:10s} full {full[n]:.4f}  boot se {s.std():.4f}  diff vs {base} {full[n] - full[base]:+.4f} +- {d.std():.4f}')
