#!/usr/bin/env python3
"""Build annotation packs from the transcript dumps.

usage: build_pack.py --tx <dir with conversation_sample_N.json> --out <pack dir> [--val <dir with <ts>_conversation_sample_N.json>]
                     [--gap 0.15] [--gold question_train.csv]
Writes <out>/<sample>.md (human: numbered islands [start-end] + text, questions, gold spans for training,
production answer/span for validation) and <out>/<sample>.json (machine: islands with words, words, questions).
Island ids are those of the chosen --gap (default 0.15 = production); the other gap's islands are listed too."""
import argparse, glob, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from medlib import *

ap = argparse.ArgumentParser()
ap.add_argument('--tx', required=True); ap.add_argument('--out', required=True); ap.add_argument('--val', default=None)
ap.add_argument('--gap', type=float, default=0.15); ap.add_argument('--gold', default=GOLD_CSV)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
gold = load_gold(a.gold) if os.path.exists(a.gold) else {}
prod = {}
for p in sorted(glob.glob(os.path.join(a.val or '/nonexistent', '*_conversation_sample_*.json'))):
    v = json.load(open(p)); prod[sample_id(v['audio_filename'])] = v
n = 0
for sid, rec in sorted(load_tx_dir(a.tx).items(), key=lambda kv: int(kv[0].split('_')[1])):
    isl = islands_of(rec, a.gap)
    alt_gap = 0.3 if a.gap == 0.15 else 0.15
    alt = islands_of(rec, alt_gap)
    words = words_of(rec)
    g = gold.get(sid, [])
    qs = rec.get('questions') or [r['question'] for r in g]   # holdout dumps carry no questions (pod CSV lacks them)
    gq = {q['question']: q for q in g}
    pv = prod.get(sid)
    md = ['# %s (%s, %.1f s, %d islands gap %s)' % (sid, rec['kind'], rec['duration'] or 0, len(isl), a.gap), '']
    md.append('## Islands (id [start-end] text)  -- start/end = first word start / last word end; "w:" = word indices for trims (w0/w1)')
    for u in isl:
        md.append('%3d [%6.2f-%6.2f] %s' % (u['id'], u['start'], u['end'], u['text']))
        if len(u['words']) >= 5:
            md.append('      w: ' + ' '.join('%d:%s' % (k, w['w'].strip()) for k, w in enumerate(u['words'])))
    md.append(''); md.append('## Questions')
    for i, q in enumerate(qs):
        line = 'Q%d: %s' % (i, q)
        if q in gq:
            r = gq[q]
            line += '  | gold=%s type=%s' % (r['answer'], r['type'])
            if r['gold']:
                t, i0, i1 = best_range(isl, r['gold'])
                line += ' span=[%.2f-%.2f] islands=%s..%s oracle_tiou=%.3f' % (r['gold'][0], r['gold'][1], i0, i1, t)
        if pv:
            res = pv['results'][i] if i < len(pv.get('results', [])) else None
            if res:
                sp = (pv['response']['evidence_start'][i], pv['response']['evidence_end'][i])
                line += '  | prod=%s span=[%.2f-%.2f] ids=%s p=%.2f' % ('yes' if pv['response']['answers'][i] else 'no', sp[0], sp[1], res.get('ids'), res.get('p') or 0)
                # map production span to island ids of this gap
                ov = [u['id'] for u in isl if min(u['end'], sp[1]) - max(u['start'], sp[0]) > 0]
                line += ' prod_islands=%s' % (ov[0] if ov else None) + ('..%s' % ov[-1] if len(ov) > 1 else '')
        md.append(line)
    md.append(''); md.append('## Alternative islands gap %s (id [start-end] text)' % alt_gap)
    for u in alt:
        md.append('%3d [%6.2f-%6.2f] %s' % (u['id'], u['start'], u['end'], u['text']))
    open(os.path.join(a.out, sid + '.md'), 'w').write('\n'.join(md) + '\n')
    mach = {'sid': sid, 'kind': rec['kind'], 'duration': rec['duration'], 'gap': a.gap, 'onsets': rec.get('onsets'),
            'islands': [{'id': u['id'], 'start': u['start'], 'end': u['end'], 'text': u['text'], 'words': u['words']} for u in isl],
            'islands_alt': {str(alt_gap): [{'id': u['id'], 'start': u['start'], 'end': u['end'], 'text': u['text']} for u in alt]},
            'words': words, 'questions': qs,
            'gold': [{'question': r['question'], 'answer': r['answer'], 'type': r['type'], 'span': r['gold']} for r in g] or None,
            'prod': ({'answers': pv['response']['answers'], 'spans': list(zip(pv['response']['evidence_start'], pv['response']['evidence_end'])),
                      'ids': [r.get('ids') for r in pv['results']]} if pv else None)}
    json.dump(mach, open(os.path.join(a.out, sid + '.json'), 'w'))
    n += 1
print('packs', n, '->', a.out)
