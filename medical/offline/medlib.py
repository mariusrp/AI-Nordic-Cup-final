"""Shared helpers for the offline medical annotation lane (packs, choice -> span, scoring).

Transcript dumps (/workspace/medical_val/tx/<conversation_sample_N>.json, made by /workspace/medtx/tx_all.py on
POD=gpu) hold: segments[].words (s, e, w, p), onsets (energy onsets, s), islands_0.15 / islands_0.3 (units:
id, start, end, text, n_words; start/end are WORD times = first word start / last word end, the acoustic edges
are not stored). Gold (question_train.csv): evidence_start/end in seconds for label==1 rows."""
import csv, glob, json, os, re

GOLD_CSV = os.environ.get('MED_GOLD_CSV', '/Users/adrian/blinq/projects/nac-n3-tmp/upstream/medical-appointment/data/question_train.csv')


def tiou(a, b):
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


def sample_id(name):
    """'conversation_sample_4.json' / 'sample_4' / 'conversation_sample_4.mp3' -> 'sample_4'"""
    m = re.search(r'sample_(\d+)', name)
    return 'sample_%s' % m.group(1) if m else name


def load_tx(path):
    rec = json.load(open(path))
    rec['sid'] = sample_id(rec.get('file') or os.path.basename(path))
    return rec


def load_tx_dir(d):
    return {r['sid']: r for r in (load_tx(p) for p in sorted(glob.glob(os.path.join(d, 'conversation_sample_*.json'))))}


def words_of(rec):
    out = []
    for s in rec['segments']:
        for w in s.get('words') or []:
            out.append({'s': float(w['s']), 'e': float(w['e']), 'w': w['w'], 'p': float(w.get('p', 1.0))})
    return out


def islands_of(rec, gap=0.15):
    isl = rec.get('islands_%s' % gap) or []
    words = words_of(rec)
    out = []
    for u in isl:
        st, en = float(u['start']), float(u['end'])
        ws = [w for w in words if st - 1e-6 <= (w['s'] + w['e']) / 2 <= en + 1e-6]
        out.append({'id': int(u['id']), 'start': st, 'end': en, 'text': u['text'], 'words': ws})
    return out


def load_gold(path=GOLD_CSV):
    """sid -> list of rows (question, answer, label, question_type, gold=(s,e) or None)"""
    g = {}
    for r in csv.DictReader(open(path)):
        gold = None
        if r['label'] == '1' and r['evidence_start'] != '':
            gold = (float(r['evidence_start']), float(r['evidence_end']))
        g.setdefault(r['transcript_id'], []).append({'qid': r['question_id'], 'question': r['question'], 'answer': r['answer'],
                                                     'label': int(r['label']), 'type': r['question_type'], 'gold': gold})
    return g


def snap_onset(t, onsets, back=0.05, fwd=0.6, t_end=None):
    """First energy onset in [t-back, min(t+fwd, t_end)] (production onset_sentstart: whisper first-word starts
    are early-clamped, the gold start is the acoustic onset); None when there is none."""
    hi = t + fwd if t_end is None else min(t + fwd, t_end)
    for o in onsets or []:
        if t - back <= o <= hi:
            return o
    return None


RULE_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'rule.json')
DEFAULT_RULE = {'ds': 0.2, 'de': -0.02, 'onset': 1, 'onset_back': 0.05, 'onset_fwd': 0.6, 'onset_ds': -0.04, 'onset_in_word': 1}
if os.path.exists(RULE_JSON):
    try:
        DEFAULT_RULE = {**DEFAULT_RULE, **json.load(open(RULE_JSON))}
    except Exception:
        pass


def span_from_range(isl, i0, i1, rec=None, rule=None, w0=None, w1=None):
    """Island id range [i0..i1] (inclusive) -> (start, end) seconds with the boundary rule.
    w0/w1: optional word-index trims inside the first/last island (0-based; w1 inclusive, negative allowed)."""
    rule = {**DEFAULT_RULE, **(rule or {})}
    a, b = isl[i0], isl[i1]
    if w0 is not None and a['words']:
        w0 = max(-len(a['words']), min(len(a['words']) - 1, int(w0)))
    if w1 is not None and b['words']:
        w1 = max(-len(b['words']), min(len(b['words']) - 1, int(w1)))
    st = a['words'][w0]['s'] if (w0 is not None and a['words']) else a['start']
    en = b['words'][w1]['e'] if (w1 is not None and b['words']) else b['end']
    s = None
    if rule.get('onset') and rec is not None:
        wobj = a['words'][w0] if (w0 is not None and a['words']) else (a['words'][0] if a['words'] else None)
        t_end = wobj['e'] if (wobj and rule.get('onset_in_word', 1)) else None
        o = snap_onset(st, rec.get('onsets'), rule['onset_back'], rule['onset_fwd'], t_end)
        if o is not None:
            s = o + rule['onset_ds']
    if s is None:
        s = st + rule['ds']
    e = en + rule['de']
    s = max(0.0, s)
    if rec is not None and rec.get('duration'):
        e = min(float(rec['duration']), e)
    return (round(s, 2), round(max(e, s + 0.05), 2))


def best_range(isl, gold, max_span=6):
    """Oracle: island range [i0..i1] with the best tIoU against gold (raw word edges). Returns (tiou, i0, i1)."""
    best = (0.0, None, None)
    ov = [i for i, u in enumerate(isl) if min(u['end'], gold[1]) - max(u['start'], gold[0]) > 0]
    if not ov:
        # nearest island
        i = min(range(len(isl)), key=lambda i: min(abs(isl[i]['start'] - gold[0]), abs(isl[i]['end'] - gold[1]))) if isl else None
        return (0.0, i, i)
    lo, hi = max(0, ov[0] - 1), min(len(isl) - 1, ov[-1] + 1)
    for i0 in range(lo, hi + 1):
        for i1 in range(i0, min(hi, i0 + max_span - 1) + 1):
            t = tiou((isl[i0]['start'], isl[i1]['end']), gold)
            if t > best[0]:
                best = (t, i0, i1)
    return best
