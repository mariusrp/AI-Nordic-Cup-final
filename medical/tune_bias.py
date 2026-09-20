"""Re-threshold saved predictions without re-running the LLM.

    python tune_bias.py preds.detail.json [--spans quote|unit]

Sweeps the yes-bias (added to logit p_yes) and reports score on the dev/test
split of offline_eval (tune on dev, read test) and on all conversations.
"""
import argparse
import json
import math

import offline_eval as OE


def build(detail, bias, spans='quote'):
    preds = {}
    for fn, res in detail.items():
        a, s, e = [], [], []
        for r in res:
            p = min(max(r['p'], 1e-6), 1 - 1e-6)
            yes = math.log(p / (1 - p)) + bias > 0
            sp = r.get('cand_span') if spans == 'quote' else r.get('unit_span')
            a.append(bool(yes))
            s.append(sp[0] if yes and sp else None)
            e.append(sp[1] if yes and sp else None)
        preds[fn] = {'answers': a, 'evidence_start': s, 'evidence_end': e}
    return preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('detail')
    ap.add_argument('--spans', default='quote', choices=['quote', 'unit'])
    ap.add_argument('--biases', default='-2,-1,-0.5,0,0.5,1,1.5,2,3,4,6')
    a = ap.parse_args()
    detail = json.load(open(a.detail))
    print(f"{'bias':>5} {'yes%':>5} | {'dev':>6} {'test':>6} {'all':>6} | acc   tIoU  pos   hard  off")
    for b in [float(x) for x in a.biases.split(',')]:
        preds = build(detail, b, a.spans)
        yes = sum(sum(p['answers']) for p in preds.values()) / max(1, sum(len(p['answers']) for p in preds.values()))
        sd = OE.summary(OE.score(preds, 'dev'))
        stt = OE.summary(OE.score(preds, 'test'))
        sa = OE.summary(OE.score(preds, 'all'))
        print(f"{b:5.1f} {yes:5.2f} | {sd['score']:.4f} {stt['score']:.4f} {sa['score']:.4f} | "
              f"{sa['accuracy']:.3f} {sa['tiou']:.3f} {sa.get('positive', 0):.3f} {sa.get('hard_negative', 0):.3f} "
              f"{sa.get('off_topic', 0):.3f}")


if __name__ == '__main__':
    main()
