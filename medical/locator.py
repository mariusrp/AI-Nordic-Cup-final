"""Evidence locator without an LLM: cross-encoder over phrase-level candidate runs.

The transcript is cut into 'phrase' units (spans.build_units(mode='phrase')), every
contiguous run of 1..MAX_RUN phrases is a candidate span, and a small cross-encoder
(BAAI/bge-reranker-v2-m3) scores (question, run text). The chosen span maximises

    score(run) - alpha * |log(duration / 3 s)|        (length prior, gold spans ~3 s)

Two uses:
  (a) standalone: the whole transcript is the candidate pool;
  (b) restricted: only runs inside the neighbourhood of a cited span (LLM/heuristic
      citation, +-`nb` seconds), falling back to the cited span when the best-run
      margin over the runner-up is below `margin`.

Pod (GPU):   python locator.py score --tx-dir TX --csv CSV --out scores.json
Local (CPU): python locator.py tune --scores scores.json --tx-dir TX [--cited preds_debug.json]
"""
from __future__ import annotations

import json
import math
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import spans as S  # noqa: E402

MAX_RUN = 4
DEFAULT_MODEL = 'BAAI/bge-reranker-v2-m3'


def candidate_runs(units: Sequence[dict], max_run: int = MAX_RUN, max_dur: float = 20.0):
    out = []
    for i in range(len(units)):
        for j in range(i, min(len(units), i + max_run)):
            if units[j]['end'] - units[i]['start'] > max_dur and j > i:
                break
            out.append((i, j))
    return out


def run_text(units, i, j):
    return ' '.join(units[k]['text'] for k in range(i, j + 1))


class Reranker:
    def __init__(self, model: str = DEFAULT_MODEL, device: str = 'cuda'):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model)
        self.m = AutoModelForSequenceClassification.from_pretrained(
            model, torch_dtype=torch.float16 if device == 'cuda' and os.environ.get('LOC_FP32') != '1' else torch.float32).to(device).eval()
        self.device = device

    def score(self, q: str, texts: List[str], bs: int = 256) -> List[float]:
        out = []
        with self.torch.no_grad():
            for k in range(0, len(texts), bs):
                enc = self.tok([q] * len(texts[k:k + bs]), texts[k:k + bs], padding=True, truncation=True,
                               max_length=256, return_tensors='pt').to(self.device)
                lg = self.m(**enc).logits
                lg = lg[:, 0] if lg.shape[-1] == 1 else lg[:, -1]  # relevance / last class
                out.extend(float(x) for x in lg.float().cpu())
        return out


def lexical_topk(units: Sequence[dict], runs: Sequence[Tuple[int, int]], question: str, k: int) -> List[int]:
    """Indices of the k runs with the highest idf-weighted coverage of the question's content
    tokens (ties: shorter run first). Cheap prefilter so the CPU cross-encoder scores only
    k pairs per question instead of every run."""
    q = set(S.content_tokens(question))
    if k <= 0 or k >= len(runs) or not q:
        return list(range(min(len(runs), k if k > 0 else len(runs))))
    utoks = [set(S.content_tokens(u['text'])) for u in units]
    df: Dict[str, int] = {}
    for ts in utoks:
        for t in ts:
            df[t] = df.get(t, 0) + 1
    n = len(units)
    idf = {t: math.log((n + 1) / (df.get(t, 0) + 0.5)) for t in q}
    tot = sum(idf.values()) + 1e-9
    key = []
    for r, (i, j) in enumerate(runs):
        ts = set().union(*utoks[i:j + 1])
        key.append((-sum(idf[t] for t in q if t in ts) / tot, j - i, r))
    return [r for _, _, r in sorted(key)[:k]]


def fallback_decide(units: Sequence[dict], runs: Sequence[Tuple[int, int]], scores: Sequence[float],
                    thr: float, alpha: float, target: float, shift_s: float, shift_e: float
                    ) -> Tuple[bool, Optional[Tuple[float, float]], float]:
    """No-LLM answer for one question from the scored candidate runs: yes iff the best
    cross-encoder logit > thr; span = best run under the length prior, shifted
    (start + shift_s, end + shift_e). Returns (yes, span, max logit)."""
    if not scores:
        return False, None, -1e9
    mx = max(scores)
    sp, _, _ = pick(units, runs, scores, alpha, target)
    if sp is not None:
        st = max(0.0, sp[0] + shift_s)
        sp = (st, max(st + 0.1, sp[1] + shift_e))
    return mx > thr, sp, mx


def _rows(csv_path):
    import csv
    rows = list(csv.DictReader(open(csv_path)))
    by = {}
    for r in rows:
        by.setdefault(r['transcript_id'], []).append(r)
    return by


def cmd_score(a):
    import time
    rr = Reranker(a.model)
    by = _rows(a.csv)
    res = {}
    t0 = time.time()
    for tid, rows in sorted(by.items()):
        p = os.path.join(a.tx_dir, f'conversation_{tid}.json')
        if not os.path.exists(p):
            continue
        units = S.build_units(S.load_transcript(p), 'phrase')
        runs = candidate_runs(units)
        texts = [run_text(units, i, j) for i, j in runs]
        t = time.time()
        qs = {r['question_id']: rr.score(r['question'], texts) for r in rows}
        res[tid] = {'runs': runs, 'qs': qs}
        print(tid, len(units), 'units', len(runs), 'runs', f'{time.time() - t:.2f}s', flush=True)
    json.dump(res, open(a.out, 'w'))
    print('total', round(time.time() - t0, 1), 's')


def pick(units, runs, scores, alpha: float, target: float = 3.0,
         window: Optional[Tuple[float, float]] = None) -> Tuple[Optional[Tuple[float, float]], float, float]:
    """Best run by score - alpha*|log(dur/target)|, optionally within `window`.
    Returns (span, best value, margin over the best non-overlapping-start alternative)."""
    best, second, bspan = -1e9, -1e9, None
    for (i, j), s in zip(runs, scores):
        st, en = units[i]['start'], units[j]['end']
        if window is not None and (st < window[0] - 1e-6 or en > window[1] + 1e-6):
            continue
        d = max(en - st, 0.2)
        v = s - alpha * abs(math.log(d / target))
        if v > best:
            second, best, bspan = best, v, (st, en)
        elif v > second:
            second = v
    return bspan, best, best - second


def cmd_tune(a):
    import statistics as st
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from offline_eval import score as oscore, split_ids, summary
    scores = json.load(open(a.scores))
    by = _rows(a.csv)
    dev = split_ids('dev')
    units_by = {tid: S.build_units(S.load_transcript(os.path.join(a.tx_dir, f'conversation_{tid}.json')), 'phrase')
                for tid in scores}
    cited = json.load(open(a.cited)) if a.cited else None

    def spans_for(alpha, target, mode, nb=0.0, margin=0.0):
        """mode: 'free' | 'cited' (restrict to cited span +-nb) | 'citedonly'."""
        out = {}
        for tid, d in scores.items():
            us = units_by[tid]
            for r in by[tid]:
                sc = d['qs'][r['question_id']]
                win = None
                if mode != 'free' and cited is not None:
                    c = cited.get(r['question_id'])
                    if c is None:
                        out[r['question_id']] = None
                        continue
                    if mode == 'citedonly':
                        out[r['question_id']] = tuple(c)
                        continue
                    win = (c[0] - nb, c[1] + nb)
                sp, v, mg = pick(us, [tuple(x) for x in d['runs']], sc, alpha, target, win)
                if mode == 'cited' and (sp is None or mg < margin):
                    sp = tuple(cited[r['question_id']])
                out[r['question_id']] = sp
        return out

    def preds_from(spans, answers='gold', ps=0.0, pe=0.0):
        P = {}
        for tid in scores:
            rows = by[tid]
            ans = [int(r['label']) if answers == 'gold' else answers[r['question_id']] for r in rows]
            ss = [spans.get(r['question_id']) if x else None for r, x in zip(rows, ans)]
            ss = [(max(0.0, s[0] - ps), max(s[0] - ps + 0.1, s[1] + pe)) if s else None for s in ss]
            P[f'conversation_{tid}.mp3'] = {'answers': ans, 'evidence_start': [s[0] if s else None for s in ss],
                                            'evidence_end': [s[1] if s else None for s in ss]}
        return P

    def ev(P, split):
        return summary(oscore(P, split))

    res = {}
    modes = ['free'] + (['cited', 'citedonly'] if cited else [])
    for mode in modes:
        grid = []
        for alpha in ([0.0] if mode == 'citedonly' else [0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0]):
            for target in ([3.0] if mode == 'citedonly' else [2.0, 3.0, 4.0]):
                for nb in ([0.0] if mode != 'cited' else [0.0, 1.0, 3.0]):
                    for margin in ([0.0] if mode != 'cited' else [0.0, 0.5, 1.0, 2.0]):
                        P = preds_from(spans_for(alpha, target, mode, nb, margin))
                        d, t = ev(P, 'dev'), ev(P, 'test')
                        grid.append(((alpha, target, nb, margin), d, t))
        bestk = max(grid, key=lambda g: g[1]['tiou'])
        res[mode] = {'params': bestk[0], 'dev': bestk[1], 'test': bestk[2]}
        print(f'[{mode}] best-on-dev params (alpha,target,nb,margin)={bestk[0]}  '
              f'dev score {bestk[1]["score"]} tIoU {bestk[1]["tiou"]}  test score {bestk[2]["score"]} tIoU {bestk[2]["tiou"]}')
        for g in grid[:: max(1, len(grid) // 12)]:
            print('   ', g[0], 'dev tIoU', g[1]['tiou'], 'test tIoU', g[2]['tiou'])
        # stage 2: global boundary calibration (whisper start times run early on large-v3)
        sp = spans_for(*bestk[0][:2], mode, *bestk[0][2:])
        sg = []
        for ps in (0.0, -0.1, -0.2, -0.3, -0.4):
            for pe in (-0.1, 0.0, 0.1, 0.2):
                P = preds_from(sp, ps=ps, pe=pe)
                sg.append(((ps, pe), ev(P, 'dev'), ev(P, 'test')))
        b2 = max(sg, key=lambda g: g[1]['tiou'])
        res[mode + '+shift'] = {'params': bestk[0], 'shift': b2[0], 'dev': b2[1], 'test': b2[2]}
        print(f'[{mode}+shift] best-on-dev (pad_start,pad_end)={b2[0]}  dev score {b2[1]["score"]} tIoU '
              f'{b2[1]["tiou"]}  test score {b2[2]["score"]} tIoU {b2[2]["tiou"]}')
    # P(yes) feature: max reranker score per question -> AUC on all questions
    ys, xs = [], []
    for tid, d in scores.items():
        for r in by[tid]:
            ys.append(int(r['label']))
            xs.append(max(d['qs'][r['question_id']]))
    pos = [x for x, y in zip(xs, ys) if y]
    neg = [x for x, y in zip(xs, ys) if not y]
    auc = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))
    print(f'max-reranker-score as P(yes) feature: AUC {auc:.3f} (n={len(xs)})')
    res['auc_maxscore'] = auc
    if a.out:
        json.dump(res, open(a.out, 'w'), indent=1)


def cmd_e2e(a):
    """End-to-end preds from a pipeline detail file (heuristic or LLM): answers from the
    pipeline, optionally re-decided with the reranker's max score, spans = reranker run
    inside the cited span (+-nb s), fallback to the cited span on low margin, then the
    global boundary shift. All parameters tuned on the dev split only."""
    from offline_eval import score as oscore, summary
    scores = json.load(open(a.scores))
    by = _rows(a.csv)
    detail = json.load(open(a.detail))
    tuned = json.load(open(a.tuned))['cited+shift']
    alpha, target, nb, margin = tuned['params']
    ps, pe = tuned['shift']
    units_by = {tid: S.build_units(S.load_transcript(os.path.join(a.tx_dir, f'conversation_{tid}.json')), 'phrase')
                for tid in scores}

    def lg(p):
        p = min(max(p, 1e-4), 1 - 1e-4)
        return math.log(p / (1 - p))

    def build(w, thr, use_rr_span=True, use_shift=True):
        P = {}
        for tid, d in scores.items():
            fn = f'conversation_{tid}.mp3'
            if fn not in detail:
                continue
            ans, ss, ee = [], [], []
            for r, o in zip(by[tid], detail[fn]):
                sc = d['qs'][r['question_id']]
                if w is None:
                    yes = bool(o['answer'])
                else:
                    yes = max(sc) + w * lg(float(o.get('p', 0.5))) > thr
                c = o.get('cand_span') or o.get('span') or o.get('unit_span')
                sp = tuple(c) if c else None
                if use_rr_span and c:
                    win = (c[0] - nb, c[1] + nb)
                    rsp, v, mg = pick(units_by[tid], [tuple(x) for x in d['runs']], sc, alpha, target, win)
                    if rsp is not None and mg >= margin:
                        sp = rsp
                if sp is None:
                    rsp, _, _ = pick(units_by[tid], [tuple(x) for x in d['runs']], sc, alpha, target)
                    sp = rsp
                if sp and use_shift:
                    sp = (max(0.0, sp[0] - ps), max(sp[0] - ps + 0.1, sp[1] + pe))
                ans.append(yes)
                ss.append(sp[0] if yes and sp else None)
                ee.append(sp[1] if yes and sp else None)
            P[fn] = {'answers': ans, 'evidence_start': ss, 'evidence_end': ee}
        return P

    def ev(P):
        return {k: summary(oscore(P, k)) for k in ('dev', 'test', 'all')}

    def show(name, r):
        print(f'{name:48s} ' + '  '.join(f'{k} {v["score"]:.4f} (acc {v["accuracy"]:.3f} tIoU {v["tiou"]:.3f})'
                                           for k, v in r.items()), flush=True)

    show('pipeline as-is (answers+spans)', ev(build(None, 0, False, False)))
    show('pipeline answers + shift only', ev(build(None, 0, False, True)))
    show('pipeline answers + reranker span + shift', ev(build(None, 0, True, True)))
    grid = []
    for w in (0.0, 0.25, 0.5, 1.0, 2.0):
        for thr in [x / 2 for x in range(-12, 13)]:
            P = build(w, thr)
            grid.append(((w, thr), summary(oscore(P, 'dev'))['score'], P))
    (w, thr), _, P = max(grid, key=lambda g: g[1])
    r = ev(P)
    show(f'answers = rr_max + {w}*logit(p) > {thr} (dev-tuned)', r)
    if a.out:
        json.dump(P, open(a.out, 'w'))
        json.dump({'w': w, 'thr': thr, **tuned, 'result': r}, open(a.out.replace('.json', '') + '.params.json', 'w'), indent=1)


def main():
    import argparse
    up = os.environ.get('UPSTREAM', '/home/claude/Nordic-AI-Cup-2026')
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['score', 'tune', 'e2e'])
    ap.add_argument('--tx-dir', required=True)
    ap.add_argument('--csv', default=os.path.join(up, 'medical-appointment/data/question_train.csv'))
    ap.add_argument('--model', default=DEFAULT_MODEL)
    ap.add_argument('--scores')
    ap.add_argument('--cited', help='json {question_id: [start,end]} of cited spans (LLM/heuristic)')
    ap.add_argument('--out')
    ap.add_argument('--detail', help='pipeline .detail.json (e2e)')
    ap.add_argument('--tuned', help='tune --out json (e2e)')
    a = ap.parse_args()
    {'score': cmd_score, 'tune': cmd_tune, 'e2e': cmd_e2e}[a.cmd](a)


if __name__ == '__main__':
    main()
