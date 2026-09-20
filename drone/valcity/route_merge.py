"""Offline proxy for a class-routed two-detector stack: per frame, keep answers of classes in ROUTE from file B and all
other classes from file A (sorted by conf, capped at 100). Usage: python route_merge.py A.jsonl B.jsonl cls1,cls2 OUT.jsonl"""
import json, sys
a, b, route, out = sys.argv[1], sys.argv[2], set(sys.argv[3].split(',')), sys.argv[4]
A = {json.loads(l)['frame']: json.loads(l) for l in open(a)}
B = {json.loads(l)['frame']: json.loads(l) for l in open(b)}
with open(out, 'w') as f:
    for fr in sorted(A):
        ann = [x for x in A[fr]['ann'] if x[0] not in route] + [x for x in B.get(fr, {'ann': []})['ann'] if x[0] in route]
        ann.sort(key=lambda x: -x[2])
        f.write(json.dumps(dict(frame=fr, ann=ann[:100])) + '\n')
