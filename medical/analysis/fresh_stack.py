"""Out-of-sample check of the FB6 zero-LLM stack (U rule, coverage-next, quote gate 'any') on the 8 FRESH
fast-r5-2 LLM runs, on top of the onset start rule and against production. No LLM calls.

Why: the FB6 arms (medical-fast-r6-2 fb6_replay.py) were scored on the same 4 cached reps that the U rule's own
lane (Allmquote-g4-2, 15:02) and the onset rule were derived on. The fast-r5-2 runs (FB5-B, 15:37: 2 interleaved
replicates x 4 prompt arms; arm0 = production config) were produced after both rules were fixed, so they are an
out-of-sample test for every FB6 arm. Arm2 used MED_UNIT_MODE=pause, so its cited ids are rebuilt on pause units.

Code under test is the medical-fast-r6-2 branch copy of spans.py / fb6_replay.py (git show into $R62), scored with
the frozen offline_eval.py (identical on main and r6-2). Pooling as in fb6_replay/replay_rules: per-conversation
score diffs summed over runs, se over conversations.
    R62=<dir with r6-2 spans.py, offline_eval.py, fb6_replay.py> python3 fresh_stack.py [fr5_dir]
"""
from __future__ import annotations

import glob
import json
import os
import statistics as st
import sys

R62 = os.environ.get('R62', '/tmp/claude-0/-home-claude/323fa704-a9b4-5fa1-bde8-352ff3b28cce/scratchpad/med/r62')
sys.path.insert(0, R62)
os.environ.setdefault('UPSTREAM', '/home/claude/upstream-work')
import numpy as np  # noqa: E402

import fb6_replay as FB  # noqa: E402
import offline_eval as OE  # noqa: E402
import spans as S  # noqa: E402

WORK = os.environ.get('MED_UND_DIR', '/tmp/claude-0/-home-claude/323fa704-a9b4-5fa1-bde8-352ff3b28cce/scratchpad/med')
PROD_CAL = (0.2, 0.0, 0, 0.3, 0.0, 0.0)
ONS_CAL = (0.0, 0.0, 0, 0.3, 0.0, 0.0)
# arm: (gate, fallback, coverage, onset)
ARMS = {
    'prod': ('first', 'first', False, False),
    'prod+U': ('first', 'unit', False, False),
    'onset': ('first', 'first', False, True),
    'onset+U': ('first', 'unit', False, True),
    'onset+cov': ('first', 'first', True, True),
    'onset+any': ('any', 'first', False, True),
    'fb6-4': ('any', 'unit', True, True),
    'onset+U+any': ('any', 'unit', False, True),
}


def span_for(cd, r, q, arm):
    gate, fb, cov, ons = ARMS[arm]
    span, src = S.quote_span(cd['units'], r['ids'], r['quote'] or '', gate=gate, fallback=fb, question=q,
                             duration=cd['dur'])
    if span is not None and ons:
        span = S.onset_sentstart(span, cd['words'], cd['onsets'], 4, -0.04)
    if span is not None and cov:
        span = S.coverage_next_end(span, cd['words'], q)
    if span is not None:
        span = S.calibrate_span(span, cd['words'], *(ONS_CAL if ons else PROD_CAL), duration=cd['dur'])
        s, e = float(span[0]), float(span[1])
        if e <= s:
            e = s + 0.3
        span = (round(s, 2), round(e, 2))
    return span, src


def conv_data(fns, mode):
    out = {}
    for fn in fns:
        base = fn.replace('.mp3', '')
        tx = S.load_transcript(os.path.join(WORK, 'tx', 'large-v3-turbo', base + '.json'))
        units = S.build_units(tx, mode)
        words = [w for u in units for w in u['words']]
        ons = S.energy_onsets(np.load(os.path.join(WORK, 'pcm', base + '.npy')))
        out[fn] = {'units': units, 'words': words, 'dur': tx.get('duration'), 'onsets': ons}
    return out


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else os.path.join(WORK, 'fr5')
    # arm2 (MED_UNIT_MODE=pause) is excluded: its cited ids do not rebuild on build_units(tx, 'pause') (the prod
    # replay reproduces 16/156 of its served spans vs 100% for arms 0/1/3), so no rule can be replayed on it.
    runs = [rp for rp in sorted(glob.glob(os.path.join(root, 'fr5_rep*', 'arm*.detail.json')))
            if not rp.endswith('arm2.detail.json')]
    qs = {fn: [r['question'] for r in rows] for fn, rows in OE.group_questions_by_conversation()}
    fns = sorted(qs)
    cds = {m: conv_data(fns, m) for m in ('sentence',)}
    dev = OE.split_ids('dev')
    per = {arm: {fn: [] for fn in fns} for arm in ARMS}   # per-conversation score per run
    fires = {arm: 0 for arm in ARMS}
    repro = [0, 0]
    for rp in runs:
        mode = 'sentence'
        det = json.load(open(rp))
        tag = '/'.join(rp.split('/')[-2:])
        line = f'{tag:28s} {mode:8s}'
        spans_by_arm = {}
        for arm in ARMS:
            preds = {}
            sp_all = []
            for fn in fns:
                a, s, e = [], [], []
                for q, r in zip(qs[fn], det[fn]):
                    sp, _ = span_for(cds[mode][fn], r, q, arm)
                    yes = bool(r['answer'])
                    a.append(yes)
                    s.append(sp[0] if yes and sp else None)
                    e.append(sp[1] if yes and sp else None)
                    sp_all.append(sp if yes else None)
                    if arm == 'prod' and yes and r.get('span') and sp:
                        repro[1] += 1
                        repro[0] += int(abs(sp[0] - r['span'][0]) < 0.011 and abs(sp[1] - r['span'][1]) < 0.011)
                preds[fn] = {'answers': a, 'evidence_start': s, 'evidence_end': e}
                per[arm][fn].append(OE.score({fn: preds[fn]}, 'all').final_score)
            spans_by_arm[arm] = sp_all
        for arm in ARMS:
            ref = spans_by_arm['onset' if arm.startswith('onset') or arm == 'fb6-4' else 'prod']
            fires[arm] += sum(1 for x, y in zip(spans_by_arm[arm], ref) if x != y)
        for arm, refarm in (('onset', 'prod'), ('onset+U', 'onset'), ('fb6-4', 'onset'), ('fb6-4', 'prod')):
            d = [per[arm][fn][-1] - per[refarm][fn][-1] for fn in fns]
            dd = [x for fn, x in zip(fns, d) if fn in dev]
            dt = [x for fn, x in zip(fns, d) if fn not in dev]
            line += f' | {arm}-{refarm} dev {np.mean(dd):+.4f} test {np.mean(dt):+.4f}'
        print(line)
    print(f'prod replay reproduces served spans: {repro[0]}/{repro[1]}')
    print('fires (spans changed vs its reference arm, summed over runs):', fires)

    def pooled(arm, ref, sub=None):
        res = {}
        for half in ('dev', 'test', 'all'):
            hf = [fn for fn in fns if half == 'all' or ((fn in dev) == (half == 'dev'))]
            cs = []
            n = 0
            for fn in hf:
                ks = range(len(per[arm][fn])) if sub is None else sub
                cs.append(sum(per[arm][fn][k] - per[ref][fn][k] for k in ks))
                n += len(list(ks))
            m = sum(cs) / n
            se = st.stdev(cs) * len(hf) ** 0.5 / n
            npos = sum(1 for c in cs if c > 1e-9)
            nneg = sum(1 for c in cs if c < -1e-9)
            res[half] = (m, se, npos, nneg)
        return res

    print('\n--- pooled paired SCORE gains, all runs (se over conversations; +conv/-conv counts) ---')
    arm0 = [k for k, rp in enumerate(runs) if rp.endswith('arm0.detail.json')]
    for arm, ref in (('onset', 'prod'), ('prod+U', 'prod'), ('onset+U', 'onset'), ('onset+cov', 'onset'),
                     ('onset+any', 'onset'), ('onset+U+any', 'onset'), ('fb6-4', 'onset'), ('fb6-4', 'prod'),
                     ('onset+U', 'prod'), ('onset+any', 'prod'), ('onset+U', 'onset+any')):
        for label, sub in (('all runs', None), ('arm0 x2', arm0)):
            r = pooled(arm, ref, sub)
            print(f'{arm:12s} vs {ref:6s} [{label:8s}] ' + ' | '.join(
                f'{h} {m:+.4f}+-{se:.4f} ({m / se if se else 0:+.1f} se, {p}+/{q}-)' for h, (m, se, p, q) in r.items()))


if __name__ == '__main__':
    main()
