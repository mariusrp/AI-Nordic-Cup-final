"""Replay the clause trim (pipeline.clause_trim, MED_SPAN_TRIM) on cached predictions - no LLM calls.

The trim is deterministic post-processing of a finished span, so replaying it on the base arm's saved
predictions is an EXACTLY paired A/B (same citations, same ASR, zero vLLM noise) and costs nothing.
Used to pick the knobs; the adoption number comes from the real paired harness (rules_arms.py).

    UPSTREAM=/workspace/.holdout/upstream-full MED_TX_DIR=/workspace/medical_val/tx \
        python3 analysis/trim_probe.py --mode 1 --margin 2 out/b1/base.rep*.json

Prints, per preds file and pooled: score before/after, the per-conversation paired diff with a
bootstrap CI, the exact/under/over/shift/wrong/none breakdown, and how often the trim fired.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import offline_eval as OE  # noqa: E402
import pipeline as P  # noqa: E402
import spans as S  # noqa: E402
from span_modes import classify  # noqa: E402

sys.path.insert(0, os.path.join(os.environ.get('UPSTREAM', '/workspace/.holdout/upstream-full'),
                                'medical-appointment'))
from utils import evidence_interval, gold_evidence, temporal_iou  # noqa: E402

TX_DIRS = [d for d in os.environ.get('MED_TX_DIR', '/workspace/medical_val/tx').split(',') if d]
UPSTREAM = os.environ.get('UPSTREAM', '/workspace/.holdout/upstream-full')
AUDIO_DIRS = [d for d in os.environ.get(
    'MED_AUDIO_DIR',
    os.path.join(UPSTREAM, 'medical-appointment/data/audio') + ',/workspace/.holdout/med_holdout_audio'
).split(',') if d]


def load_ctx():
    """{fn: {'words': [...], 'onsets': [...], 'questions': [...]}} for the 39 training conversations."""
    from faster_whisper.audio import decode_audio
    qs = {fn: [r['question'] for r in rows] for fn, rows in OE.group_questions_by_conversation()}
    ctx = {}
    for fn in sorted(qs):
        base = fn.replace('.mp3', '')
        txp = next((os.path.join(d, base + '.json') for d in TX_DIRS
                    if os.path.exists(os.path.join(d, base + '.json'))), None)
        if txp is None:
            print(f'WARN no transcript for {fn}', flush=True)
            continue
        tx = dict(S.load_transcript(txp))
        ap = next((os.path.join(d, fn) for d in AUDIO_DIRS if os.path.exists(os.path.join(d, fn))), None)
        if ap:
            tx['_pcm'] = decode_audio(ap, sampling_rate=16000)
        units = S.build_units(tx, 'island')
        words = [w for u in units for w in u['words']]
        ons = P._request_onsets(tx, tx.get('duration'))
        ctx[fn] = {'words': words, 'onsets': ons, 'questions': qs[fn]}
    return ctx


def apply_trim(preds, ctx, **kw):
    out, fired, moved = {}, 0, []
    for fn, p in preds.items():
        c = ctx.get(fn)
        q = dict(p)
        if c is None:
            out[fn] = q
            continue
        st, en = list(p['evidence_start']), list(p['evidence_end'])
        for k in range(len(st)):
            if st[k] is None or en[k] is None or k >= len(c['questions']):
                continue
            span = (float(st[k]), float(en[k]))
            t, note = P.clause_trim(span, c['words'], c['questions'][k], **kw)
            if not note:
                continue
            if t[0] > span[0] + 0.01 and P.CFG['span_onset_sentstart']:
                t = S.onset_sentstart(t, c['words'], c['onsets'], 0, P.CFG['span_onset_off'])
            fired += 1
            moved.append((fn, k, span, (round(t[0], 2), round(t[1], 2)), note))
            st[k], en[k] = round(float(t[0]), 2), round(float(t[1]), 2)
        q['evidence_start'], q['evidence_end'] = st, en
        out[fn] = q
    return out, fired, moved


def modes(preds):
    counts = {k: 0 for k in ('exact', 'under', 'over', 'shift', 'wrong', 'none')}
    tot, n = 0.0, 0
    for fn, rows in OE.group_questions_by_conversation():
        p = preds.get(fn)
        if p is None:
            continue
        for qi, row in enumerate(rows):
            if int(row['label']) != 1:
                continue
            gold = gold_evidence(row)
            sp = evidence_interval(p['evidence_start'][qi], p['evidence_end'][qi]) if p['answers'][qi] else None
            counts[classify(gold, sp)] += 1
            tot += temporal_iou(gold, sp) if sp else 0.0
            n += 1
    return {'tiou_yes': round(tot / n, 4) if n else 0.0, **counts}


def conv_score(pred, fn):
    return OE.summary(OE.score({fn: pred}, 'all'))['score']


def boot_ci(diffs, b=4000, seed=2026):
    if not diffs:
        return float('nan'), float('nan'), float('nan')
    rnd = random.Random(seed)
    n = len(diffs)
    ms = sorted(sum(diffs[rnd.randrange(n)] for _ in range(n)) / n for _ in range(b))
    return ms[int(0.025 * (b - 1))], ms[int(0.975 * (b - 1))], sum(1 for m in ms if m > 0) / b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('preds', nargs='+')
    ap.add_argument('--mode', type=int, default=1)
    ap.add_argument('--margin', type=int, default=2)
    ap.add_argument('--max-frac', type=float, default=0.6)
    ap.add_argument('--min-keep', type=float, default=0.8)
    ap.add_argument('--min-hits', type=int, default=2)
    ap.add_argument('--min-drop', type=float, default=0.2)
    ap.add_argument('--end-strict', type=int, default=1)
    ap.add_argument('--show', type=int, default=0, help='print this many fired trims')
    ap.add_argument('--save', default='')
    ap.add_argument('--grid', default='', help='mode:margin:max_frac:min_hits,...')
    a = ap.parse_args()
    grid = []
    for spec in (a.grid.split(',') if a.grid else ['']):
        kw = dict(mode=a.mode, margin=a.margin, max_frac=a.max_frac, min_keep=a.min_keep,
                  min_hits=a.min_hits, min_drop=a.min_drop, end_strict=a.end_strict)
        if spec:
            f = spec.split(':')
            kw['mode'] = int(f[0])
            if len(f) > 1:
                kw['margin'] = int(f[1])
            if len(f) > 2:
                kw['max_frac'] = float(f[2])
            if len(f) > 3:
                kw['min_hits'] = int(f[3])
            if len(f) > 4:
                kw['end_strict'] = int(f[4])
        grid.append((spec or 'cli', kw))
    ctx = load_ctx()
    cache = {path: json.load(open(path)) for path in a.preds}
    for spec, kw in grid:
        print(f'=== {spec} {kw}', flush=True)
        run_one(a, ctx, cache, kw)


def run_one(a, ctx, cache, kw):
    dev, test = OE.split_ids('dev'), OE.split_ids('test')
    pooled, fired_tot, shown = [], 0, 0
    for path in a.preds:
        base = cache[path]
        new, fired, moved = apply_trim(base, ctx, **kw)
        fired_tot += fired
        b_all = OE.summary(OE.score(base, 'all'))['score']
        n_all = OE.summary(OE.score(new, 'all'))['score']
        diffs = {fn: conv_score(new[fn], fn) - conv_score(base[fn], fn) for fn in base}
        pooled += [(fn, d) for fn, d in diffs.items()]
        d_dev = sum(d for fn, d in diffs.items() if fn in dev) / max(1, len(dev))
        d_test = sum(d for fn, d in diffs.items() if fn in test) / max(1, len(test))
        print(f'{os.path.basename(path):<18} base {b_all:.4f} -> trim {n_all:.4f} '
              f'({n_all - b_all:+.4f})  dev {d_dev:+.4f} test {d_test:+.4f}  fired {fired}')
        print(f'   modes base {modes(base)}')
        print(f'   modes trim {modes(new)}')
        if a.show and not shown:
            rows_by_conv = {fn: rows for fn, rows in OE.group_questions_by_conversation()}
            det = []
            for fn, k, sp, t, note in moved:
                row = rows_by_conv.get(fn, [])[k] if k < len(rows_by_conv.get(fn, [])) else None
                if row is None or int(row['label']) != 1:
                    det.append((0.0, fn, k, sp, t, 'no gold (hard_negative/off_topic)', row))
                    continue
                g = gold_evidence(row)
                b, c = temporal_iou(g, sp), temporal_iou(g, t)
                det.append((round(c - b, 3), fn, k, sp, t, f'gold {g} tIoU {b:.3f} -> {c:.3f}', row))
            for d, fn, k, sp, t, msg, row in sorted(det):
                q = (row or {}).get('question', '')
                ww = ctx[fn]['words']
                txt = ''.join(('[' + w['w'].strip() + ']' if t[0] - 0.01 <= (w['s'] + w['e']) / 2 <= t[1] + 0.01
                               else ' ' + w['w'].strip() + ' ')
                              for w in ww if sp[0] - 0.05 <= (w['s'] + w['e']) / 2 <= sp[1] + 0.05)
                frac = ((sp[1] - sp[0]) - (t[1] - t[0])) / max(1e-9, sp[1] - sp[0])
                print(f'   {d:+.3f} {fn} q{k} {sp} -> {(round(t[0], 2), round(t[1], 2))} drop {frac:.2f}  {msg}  | {q}')
                print(f'          {txt}')
            shown = 1
        if a.save:
            json.dump(new, open(os.path.join(a.save, os.path.basename(path)), 'w'))
    if len(a.preds) > 1:
        per = {}
        for fn, d in pooled:
            per.setdefault(fn, []).append(d)
        md = {fn: sum(v) / len(v) for fn, v in per.items()}
        lo, hi, pp = boot_ci(list(md.values()))
        mean = sum(md.values()) / len(md)
        dd = sum(v for fn, v in md.items() if fn in dev) / max(1, len(dev))
        dt = sum(v for fn, v in md.items() if fn in test) / max(1, len(test))
        better = sum(1 for v in md.values() if v > 1e-9)
        worse = sum(1 for v in md.values() if v < -1e-9)
        print(f'POOLED {len(a.preds)} reps  paired {mean:+.4f} [{lo:+.4f},{hi:+.4f}] p_pos {pp:.3f}  '
              f'dev {dd:+.4f} test {dt:+.4f}  {better} better / {worse} worse  fired {fired_tot}')


if __name__ == '__main__':
    main()
