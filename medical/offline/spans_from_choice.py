#!/usr/bin/env python3
"""Island-id choices -> evidence spans (seconds) with the fitted boundary rule (medical/offline/rule.json).

usage: spans_from_choice.py --pack <pack dir> --choices <choices.json> [--out spans.json] [--rule rule.json] [--print]

choices.json: {"<sample_N>": {"<q index or question text>": CHOICE, ...}, ...}
CHOICE forms (island ids are the pack's gap-0.15 ids):
  7             one island
  [7, 9]        island range 7..9 inclusive (also "7-9" / "7..9")
  {"i0": 7, "i1": 9, "w0": 2, "w1": -2}   word-level trims: start at word index w0 of island i0 (0-based),
                                          end at word index w1 of island i1 (inclusive, negative = from the end)
  {"i0": 7, "i1": 7, "answer": false}     optional answer override (default yes when a span is chosen)
  null / "none"  no evidence: answer no, span = production fallback (validation) or [0, 0.42]
Output: {"<sample_N>": {"answers": [...], "evidence_start": [...], "evidence_end": [...], "questions": [...]}}
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from medlib import *


def parse_choice(c):
    if c is None or (isinstance(c, str) and c.strip().lower() in ('none', 'no', '')):
        return None
    if isinstance(c, (int, float)):
        return {'i0': int(c), 'i1': int(c)}
    if isinstance(c, str):
        s = c.replace('..', '-').replace(' ', '')
        a, _, b = s.partition('-')
        return {'i0': int(a), 'i1': int(b or a)}
    if isinstance(c, (list, tuple)):
        return {'i0': int(c[0]), 'i1': int(c[-1])}
    if isinstance(c, dict):
        d = dict(c); d['i0'] = int(d.get('i0', d.get('i', 0))); d['i1'] = int(d.get('i1', d['i0'])); return d
    raise ValueError('bad choice %r' % (c,))


def spans_for_pack(pack, choices, rule=None):
    isl = [{'id': u['id'], 'start': u['start'], 'end': u['end'], 'text': u['text'], 'words': u['words']} for u in pack['islands']]
    rec = {'onsets': pack.get('onsets'), 'duration': pack.get('duration')}
    qs = pack['questions']
    ans, es, ee, src = [], [], [], []
    for qi, q in enumerate(qs):
        c = choices.get(str(qi), choices.get(q, choices.get(qi)))
        d = parse_choice(c) if c is not None else None
        if d is None:
            p = pack.get('prod')
            if p and qi < len(p['spans']):
                a, sp = False, p['spans'][qi]
            else:
                a, sp = False, (0.0, 0.42)
            ans.append(a); es.append(round(float(sp[0]), 2)); ee.append(round(float(sp[1]), 2)); src.append('fallback'); continue
        i0, i1 = min(d['i0'], d['i1']), max(d['i0'], d['i1'])
        if not (0 <= i0 < len(isl) and 0 <= i1 < len(isl)):
            raise ValueError('%s q%d: island id out of range %s (n=%d)' % (pack['sid'], qi, (i0, i1), len(isl)))
        s, e = span_from_range(isl, i0, i1, rec, rule, d.get('w0'), d.get('w1'))
        ans.append(bool(d.get('answer', True))); es.append(s); ee.append(e); src.append('choice')
    return {'answers': ans, 'evidence_start': es, 'evidence_end': ee, 'questions': qs, 'src': src}


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--pack', required=True); ap.add_argument('--choices', required=True); ap.add_argument('--out', default=None)
    ap.add_argument('--rule', default=None); ap.add_argument('--print', action='store_true')
    a = ap.parse_args()
    rule = json.load(open(a.rule)) if a.rule else None
    ch = json.load(open(a.choices))
    out = {}
    for sid, cmap in ch.items():
        sid = sample_id(sid)
        pack = json.load(open(os.path.join(a.pack, sid + '.json')))
        out[sid] = spans_for_pack(pack, cmap, rule)
        if a.print:
            for qi, q in enumerate(pack['questions']):
                print('%s q%d %s %s [%.2f-%.2f] %s' % (sid, qi, out[sid]['src'][qi], 'yes' if out[sid]['answers'][qi] else 'no', out[sid]['evidence_start'][qi], out[sid]['evidence_end'][qi], q[:70]))
    if a.out:
        json.dump(out, open(a.out, 'w'), indent=1); print('wrote', a.out, len(out), 'conversations')
