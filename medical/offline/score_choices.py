#!/usr/bin/env python3
"""Score island choices against the training gold (tIoU per gold-yes question, mean per conversation and overall,
plus accuracy and the 0.4*acc + 0.6*tIoU score).

usage: score_choices.py --pack <pack dir> --choices <choices.json> [--rule rule.json] [--spans spans.json] [-v]
  --choices  as for spans_from_choice.py (training samples only are scored; others are skipped)
  --spans    score an already computed spans.json (output of spans_from_choice.py) instead
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from medlib import *
from spans_from_choice import spans_for_pack

ap = argparse.ArgumentParser()
ap.add_argument('--pack', required=True); ap.add_argument('--choices', default=None); ap.add_argument('--spans', default=None)
ap.add_argument('--rule', default=None); ap.add_argument('-v', action='store_true')
a = ap.parse_args()
rule = json.load(open(a.rule)) if a.rule else None
if a.spans:
    spans = json.load(open(a.spans))
else:
    spans = {}
    for sid, cmap in json.load(open(a.choices)).items():
        sid = sample_id(sid)
        spans[sid] = spans_for_pack(json.load(open(os.path.join(a.pack, sid + '.json'))), cmap, rule)
gold = load_gold()
T, A, per = [], [], {}
for sid, sp in spans.items():
    g = gold.get(sid)
    if not g:
        continue
    gq = {q['question']: q for q in g}
    ts, acc = [], []
    for qi, q in enumerate(sp['questions']):
        r = gq.get(q)
        if r is None: continue
        acc.append(int(bool(sp['answers'][qi]) == (r['answer'] == 'yes')))
        if r['gold']:
            t = tiou((sp['evidence_start'][qi], sp['evidence_end'][qi]), r['gold']); ts.append(t)
            if a.v:
                print('%s q%d tiou %.3f ours [%.2f-%.2f] gold [%.2f-%.2f] %s' % (sid, qi, t, sp['evidence_start'][qi], sp['evidence_end'][qi], r['gold'][0], r['gold'][1], q[:60]))
    per[sid] = (sum(ts) / len(ts) if ts else None, sum(acc) / len(acc) if acc else None)
    T += ts; A += acc
    print('%s tIoU %.4f (n=%d) acc %.2f' % (sid, per[sid][0] or 0, len(ts), per[sid][1] or 0))
if T:
    mt, ma = sum(T) / len(T), sum(A) / len(A)
    print('ALL conversations %d gold-yes %d  mean tIoU %.4f  acc %.4f  score(0.4acc+0.6tiou) %.4f' % (len(per), len(T), mt, ma, 0.4 * ma + 0.6 * mt))
