"""Never-tried D-island probe, restricted to the LLM's OWN cited clusters: can a one-token LLM judgement
P(yes | question, excerpt) pick the gold cluster when the LLM cites 2+ clusters (35% of the post-merge location
misses cite the gold line but serve another cluster)?

Pre-registered (fixed before any LLM call): cases = gold-yes questions answered yes in the 6 fresh runs (deploy
config spans from deploy_check.py) whose cited ids form >= 2 clusters (spans.clusters, max_gap 2) with at least one
cluster overlapping gold. Picks compared on cluster accuracy (picked cluster overlaps gold): served (the cluster the
served span overlaps), first, last, verifier argmax P(yes). Verifier prompt below, temperature 0, max_tokens 1,
P(yes) = yes/(yes+no) mass in the top-10 logprobs of the first token.
    python3 cluster_verifier.py build <dc_deploy.json> cases.json    # local
    python3 cluster_verifier.py ask cases.json out.json               # on POD=gpu (shared vLLM :8001), light client
    python3 cluster_verifier.py score cases.json out.json             # local
"""
import collections
import concurrent.futures as cf
import glob
import json
import math
import os
import sys

SYSTEM = ('You judge whether a short excerpt from a doctor-patient conversation transcript states the answer to a '
          'yes/no question. Reply with one word: yes or no.')
USER = ('Question: {q}\nExcerpt: "{x}"\nDoes this excerpt itself establish that the answer to the question is yes? '
        'Answer yes or no.')


def build(served_path, out_path):
    import common as C
    import fresh_stack as FS
    import spans as S
    served = json.load(open(served_path))
    runs = [rp for rp in sorted(glob.glob(os.path.join(C.WORK, 'fr5', 'fr5_rep*', 'arm*.detail.json')))
            if not rp.endswith('arm2.detail.json')]
    cv = C.convs()
    cds = FS.conv_data(sorted(cv), 'sentence')
    cases = []
    for rp in runs:
        det = json.load(open(rp))
        tag = '/'.join(rp.split('/')[-2:])
        for fn, rows in cv.items():
            units = cds[fn]['units']
            for r, o, d in zip(rows, served[tag][fn], det[fn]):
                g = C.gold(r)
                if int(r['label']) != 1 or g is None or not o['answer'] or not o['span']:
                    continue
                ids = [i for i in (d['ids'] or []) if isinstance(i, int) and 0 <= i < len(units)]
                cl = S.clusters(ids, 2)
                if len(cl) < 2:
                    continue
                cs = [(units[c[0]]['start'], units[c[-1]]['end'], ' '.join(units[i]['text'] for i in c)) for c in cl]
                hit = [min(g[1], e) - max(g[0], s) > 0 for s, e, _ in cs]
                if not any(hit):
                    continue
                sp = o['span']
                ovl = [max(0.0, min(sp[1], e) - max(sp[0], s)) for s, e, _ in cs]
                served_k = max(range(len(cs)), key=lambda k: ovl[k]) if max(ovl) > 0 else -1
                cases.append({'run': tag, 'qid': r['question_id'], 'q': r['question'], 'clusters': cs, 'hit': hit,
                              'served': served_k, 'served_tiou': C.tiou(g, tuple(sp))})
    json.dump(cases, open(out_path, 'w'))
    print(len(cases), 'cases;', len({c['qid'] for c in cases}), 'unique questions;',
          len({(c['q'], x[2]) for c in cases for x in c['clusters']}), 'unique (question, excerpt) pairs')


def ask(cases_path, out_path):
    import requests
    url = os.environ.get('MED_LLM_URL', 'http://127.0.0.1:8001/v1').rstrip('/')
    s = requests.Session()
    model = s.get(url + '/models', timeout=10).json()['data'][0]['id']
    cases = json.load(open(cases_path))
    pairs = sorted({(c['q'], x[2]) for c in cases for x in c['clusters']})

    def one(p):
        body = {'model': model, 'temperature': 0.0, 'max_tokens': 1, 'logprobs': True, 'top_logprobs': 10,
                'chat_template_kwargs': {'enable_thinking': False},
                'messages': [{'role': 'system', 'content': SYSTEM},
                             {'role': 'user', 'content': USER.format(q=p[0], x=p[1])}]}
        r = s.post(url + '/chat/completions', json=body, timeout=60).json()
        top = r['choices'][0]['logprobs']['content'][0]['top_logprobs']
        py = sum(math.exp(a['logprob']) for a in top if a['token'].strip().lower().startswith('yes'))
        pn = sum(math.exp(a['logprob']) for a in top if a['token'].strip().lower().startswith('no'))
        return py / (py + pn) if py + pn > 0 else 0.5

    with cf.ThreadPoolExecutor(6) as ex:
        res = list(ex.map(one, pairs))
    json.dump([[q, x, p] for (q, x), p in zip(pairs, res)], open(out_path, 'w'))
    print(len(pairs), 'pairs scored')


def score(cases_path, out_path):
    cases = json.load(open(cases_path))
    pv = {(q, x): p for q, x, p in json.load(open(out_path))}
    acc = collections.Counter()
    per_q = collections.defaultdict(lambda: collections.Counter())
    for c in cases:
        ps = [pv[(c['q'], x[2])] for x in c['clusters']]
        picks = {'served': c['served'], 'first': 0, 'last': len(ps) - 1,
                 'verifier': max(range(len(ps)), key=lambda k: (ps[k], -k))}
        for name, k in picks.items():
            ok = int(k >= 0 and c['hit'][k])
            acc[name] += ok
            per_q[c['qid']][name] += ok
        per_q[c['qid']]['n'] += 1
    n = len(cases)
    print(f'{n} cases ({len(per_q)} unique questions): cluster accuracy ' +
          ', '.join(f'{k} {v}/{n} ({v / n:.2f})' for k, v in acc.items()))
    better = sum(1 for q in per_q.values() if q['verifier'] > q['served'])
    worse = sum(1 for q in per_q.values() if q['verifier'] < q['served'])
    print(f'per unique question, verifier vs served: {better} better, {worse} worse, '
          f'{len(per_q) - better - worse} same')


if __name__ == '__main__':
    {'build': build, 'ask': ask, 'score': score}[sys.argv[1]](*sys.argv[2:])
