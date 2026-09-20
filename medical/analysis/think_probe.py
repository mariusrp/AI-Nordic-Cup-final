"""Probe (cycle 2): can the LLM's own judgement fix the SYSTEMATIC wrong-location misses?

Runs on POD=gpu as a light client of the shared vLLM (:8001). For a fixed item list (the 17 gold-yes questions that
production misses in 4/4 cached runs, the 4 intermittent ones, and 30 random never-missed controls; visible
conversations only, built locally from the worker CSV), asks the production prompt in three arms:
  A0  production prompt, think off (paired baseline in the same session)
  A1  production prompt, think ON (max_tokens 3000)
  A2  production prompt + one 'which mention' line, think off
and replays the production span path (align_quote -> 3-unit proximity gate -> unit fallback -> +0.2 s start).
    python3 think_probe.py ITEMS.json TX_DIR MED_CODE_DIR OUT.json
"""
import concurrent.futures as cf
import json
import os
import sys
import time

import requests

ITEMS, TXD, CODE, OUT = sys.argv[1:5]
sys.path.insert(0, CODE)
import pipeline as P  # noqa: E402
import spans as S  # noqa: E402

URL = 'http://127.0.0.1:8001/v1'
MENTION = ('If the transcript states the answer more than once, cite the line where it is stated most explicitly '
           "in the question's own terms, often a later summary, conclusion or confirmation (e.g. \"So, X? Yes.\"), "
           'rather than the first passing mention.')
SYS2 = P.SYSTEM.replace('\n\nReply with exactly three lines', '\n' + MENTION + '\n\nReply with exactly three lines')
assert SYS2 != P.SYSTEM
MODEL = requests.get(URL + '/models', timeout=10).json()['data'][0]['id']


def tiou(a, b):
    if a is None or b is None:
        return 0.0
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    u = max(a[1], b[1]) - min(a[0], b[0])
    return inter / u if u > 0 else 0.0


def span_of(units, words, ids, quote, dur):
    span = None
    if quote:
        span = S.align_quote(units, ids, quote)
        if span is not None and ids:
            us = S.span_for_ids(units, ids, max_gap=2)
            if us and (span[1] < us[0] - 3 or span[0] > us[1] + 3):
                span = None
    if span is None:
        span = S.span_for_ids(units, ids, duration=dur)
    if span is None:
        return None
    return S.calibrate_span(span, words, 0.2, 0.0, 0, 0.3, 0.0, 0.0, duration=dur)


def ask(item, arm):
    tx = json.load(open(os.path.join(TXD, item['fn'].replace('.mp3', '.json'))))
    units = S.build_units(tx, 'sentence')
    words = [w for u in units for w in u['words']]
    system = SYS2 if arm == 'A2' else P.SYSTEM
    body = {'model': MODEL, 'messages': [{'role': 'system', 'content': system},
                                         {'role': 'user', 'content': P.build_user(units, item['question'])}],
            'temperature': 0.0, 'max_tokens': 3000 if arm == 'A1' else 160, 'logprobs': True, 'top_logprobs': 10,
            'chat_template_kwargs': {'enable_thinking': arm == 'A1'}}
    t = time.time()
    try:
        ch = requests.post(URL + '/chat/completions', json=body, timeout=240).json()['choices'][0]
    except Exception as e:
        return {'arm': arm, 'qid': item['qid'], 'err': repr(e)[:200]}
    text = (ch.get('message') or {}).get('content') or ''
    reasoning = (ch.get('message') or {}).get('reasoning_content') or ''
    ans, ids, quote = P.parse_reply(text, len(units))
    p = P._answer_prob(ch)
    yes = (p is not None and p > 0.23) or (p is None and bool(ans))
    sp = span_of(units, words, ids, quote, tx.get('duration')) if yes else None
    return {'arm': arm, 'qid': item['qid'], 'group': item['group'], 'p': p, 'yes': yes, 'ids': ids, 'quote': quote,
            'span': sp, 'tiou': tiou(tuple(item['gold']), sp), 'prod_tiou': item['prod_tiou'],
            'secs': round(time.time() - t, 1), 'n_reason_chars': len(reasoning) + (len(text) if '</think>' in text else 0),
            'finish': ch.get('finish_reason')}


def main():
    items = json.load(open(ITEMS))
    jobs = [(it, arm) for arm in ('A0', 'A2', 'A1') for it in items]
    out = []
    with cf.ThreadPoolExecutor(8) as ex:
        for r in ex.map(lambda a: ask(*a), jobs):
            out.append(r)
            json.dump(out, open(OUT, 'w'))
    print('done', len(out))


if __name__ == '__main__':
    main()
