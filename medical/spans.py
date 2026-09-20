"""Transcript units and evidence-span construction.

A transcript (faster-whisper output with word timestamps) is turned into short,
numbered *units* (sentences, optionally split further at long pauses). The LLM
cites unit ids; `span_for_ids` turns the cited ids into one (start, end) span.

Transcript format (as cached by work/med/transcribe.py and produced by
pipeline.transcribe):
    {"duration": float, "segments": [{"start","end","text",
                                      "words":[{"s","e","w","p"}]}]}

Run `python spans.py` for the offline span analysis against the gold spans
(no LLM needed): oracle tIoU for single units / best contiguous runs and a
grid search over padding rules, with a held-out split.
"""
from __future__ import annotations

import json
import os
import re
from typing import Dict, List, Optional, Sequence, Tuple

Span = Tuple[float, float]

SENT_END = re.compile(r'[.?!]["\')\]]*$')
CONJ = {'and', 'but', 'with', 'so', 'because', 'which', 'while', 'although', 'then'}

# Defaults chosen by the analysis in EXPERIMENTS.md (E1). Keep in sync.
DEFAULT_UNIT_MODE = 'island'
DEFAULT_PAD_START = 0.0
DEFAULT_PAD_END = 0.0
DEFAULT_MAX_GAP_UNITS = 2


def load_transcript(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def words_of(tx: dict) -> List[dict]:
    """Flat word list [{s,e,w}] with sanity fixes on whisper timestamps."""
    out = []
    for seg in tx.get('segments', []):
        ws = seg.get('words') or []
        if not ws:  # segment-level only transcript: treat as one "word"
            out.append({'s': float(seg['start']), 'e': float(seg['end']),
                        'w': ' ' + seg.get('text', '').strip(), 'seg_start': True})
            continue
        for i, w in enumerate(ws):
            s, e = float(w['s']), float(w['e'])
            if e < s:
                e = s
            out.append({'s': s, 'e': e, 'w': w['w'], 'seg_start': i == 0})
    # enforce monotonic
    for i in range(1, len(out)):
        if out[i]['s'] < out[i - 1]['s']:
            out[i]['s'] = out[i - 1]['s']
        if out[i]['e'] < out[i]['s']:
            out[i]['e'] = out[i]['s']
    return out


def speech_islands(pcm, sr: int = 16000, thr_db: float = -40.0, min_sil: float = 0.3, end_trim: float = 0.1,
                   win: int = 400, hop: int = 160) -> List[Tuple[float, float]]:
    """Speech islands [(start, end)] from frame RMS (25 ms / 10 ms @ 16 kHz): runs above thr_db, merged when
    the silence between them is shorter than min_sil; the end is trimmed by end_trim (the annotators end a span
    at the last word's decay, ~0.1-0.18 s before the room-tone floor). The conversations are stitched TTS
    clips with 0.4-0.8 s gaps, so islands ~ utterance clips (n3 audio-only analysis of all 39 conversations)."""
    import numpy as np
    x = np.asarray(pcm, dtype=np.float64)
    n = (len(x) - win) // hop
    if n <= 0:
        return []
    fr = np.lib.stride_tricks.sliding_window_view(x, win)[::hop][:n]
    rms = np.sqrt((fr ** 2).mean(1) + 1e-12)
    db = 20 * np.log10(rms + 1e-9)
    mask = (db > thr_db).astype(np.int8)
    d = np.diff(np.concatenate([[0], mask, [0]]))
    starts = np.where(d == 1)[0]
    ends = np.where(d == -1)[0]
    hop_s = hop / sr
    out: List[List[float]] = []
    for s_i, e_i in zip(starts, ends):
        st, en = s_i * hop_s, e_i * hop_s
        if out and st - out[-1][1] < min_sil:
            out[-1][1] = en
        else:
            out.append([st, en])
    return [(st, max(st, en - end_trim)) for st, en in out]


def _island_units(words: Sequence[dict], islands: Sequence[Tuple[float, float]], max_words: int) -> List[dict]:
    """One unit per speech island; words are assigned by midpoint (nearest island when outside all of them);
    islands longer than max_words are split at sentence ends. Unit times stay word-based (onset rule and
    coverage-next operate on words) so only the GROUPING changes versus sentence units."""
    groups: List[List[dict]] = [[] for _ in islands]
    for w in words:
        mid = (w['s'] + w['e']) / 2
        best, bd = 0, float('inf')
        for i, (st, en) in enumerate(islands):
            dist = 0.0 if st <= mid <= en else min(abs(mid - st), abs(mid - en))
            if dist < bd:
                best, bd = i, dist
                if dist == 0.0:
                    break
        groups[best].append(w)
    units: List[dict] = []
    for g in groups:
        if not g:
            continue
        parts: List[List[dict]] = [[]]
        for w in g:
            parts[-1].append(w)
            if len(g) > max_words and SENT_END.search(w['w'].strip() or 'x') and len(parts[-1]) >= 3:
                parts.append([])
        for cur in parts:
            if cur:
                units.append({'id': len(units), 'start': cur[0]['s'], 'end': cur[-1]['e'],
                              'text': ''.join(w['w'] for w in cur).strip(), 'words': list(cur)})
    return units


def build_units(tx: dict, mode: str = DEFAULT_UNIT_MODE,
                pause: float = 1.0, max_words: int = 40) -> List[dict]:
    """Split the transcript into units.

    mode:
      'segment'  - whisper segments as-is
      'sentence' - split at sentence-final punctuation (. ? !)
      'clause'   - sentence + split at commas/semicolons when the clause >= 4 words
    Also splits whenever the silence between two words exceeds `pause` seconds
    (speaker change is usually a pause) and caps units at `max_words`.
    """
    words = words_of(tx)
    if mode == 'island':
        pcm = tx.get('_pcm')
        if pcm is not None:
            isl = speech_islands(pcm, thr_db=float(os.environ.get('MED_ISLAND_THR', '-40')),
                                 min_sil=float(os.environ.get('MED_ISLAND_GAP', '0.15')),
                                 end_trim=float(os.environ.get('MED_ISLAND_END_TRIM', '0.1')))
            if isl and words:
                return _island_units(words, isl, max_words)
        mode = 'sentence'  # no PCM (onset rule off / decode failed): fall back to sentence units
    units: List[dict] = []
    cur: List[dict] = []

    def flush():
        if cur:
            units.append({'id': len(units), 'start': cur[0]['s'], 'end': cur[-1]['e'],
                          'text': ''.join(w['w'] for w in cur).strip(),
                          'words': list(cur)})
            cur.clear()

    for i, w in enumerate(words):
        if cur:
            gap = w['s'] - cur[-1]['e']
            if mode == 'segment':
                if w.get('seg_start'):
                    flush()
            elif gap > pause or len(cur) >= max_words:
                flush()
            elif w.get('seg_start') and SENT_END.search(cur[-1]['w'].strip() or 'x'):
                flush()
        cur.append(w)
        t = w['w'].strip()
        if mode in ('sentence', 'clause', 'phrase', 'comma') and SENT_END.search(t):
            flush()
        elif mode == 'clause' and t.endswith((',', ';', ':')) and len(cur) >= 4:
            flush()
        elif mode in ('phrase', 'comma') and t.endswith((',', ';', ':')) and len(cur) >= 2:
            flush()
        elif (mode == 'phrase' and i + 1 < len(words) and len(cur) >= 3
              and words[i + 1]['w'].strip().lower() in CONJ):
            flush()
    flush()
    return units


def unit_lines(units: Sequence[dict]) -> str:
    """Numbered transcript for the prompt: `[12] (41.2) text`."""
    return '\n'.join(f"[{u['id']}] ({u['start']:.1f}) {u['text']}" for u in units)


def tiou(a: Optional[Span], b: Optional[Span]) -> float:
    if a is None or b is None:
        return 0.0
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


def clusters(ids: Sequence[int], max_gap: int) -> List[List[int]]:
    ids = sorted(set(int(i) for i in ids))
    out: List[List[int]] = []
    for i in ids:
        if out and i - out[-1][-1] <= max_gap:
            out[-1].append(i)
        else:
            out.append([i])
    return out


def span_for_ids(units: Sequence[dict], ids: Sequence[int],
                 pad_start: float = DEFAULT_PAD_START, pad_end: float = DEFAULT_PAD_END,
                 max_gap: int = DEFAULT_MAX_GAP_UNITS, duration: Optional[float] = None,
                 pick: str = 'first') -> Optional[Span]:
    """Cited unit ids -> one span. Non-contiguous citations are grouped into
    clusters (ids at most `max_gap` apart); `pick` selects 'first' cluster,
    'largest' cluster, or 'all' (min..max)."""
    ids = [i for i in ids if isinstance(i, int) and 0 <= i < len(units)]
    if not ids:
        return None
    cl = clusters(ids, max_gap)
    if pick == 'all':
        c = [min(ids), max(ids)]
    elif pick == 'largest':
        c = max(cl, key=len)
    else:
        c = cl[0]
    s = units[c[0]]['start'] - pad_start
    e = units[c[-1]]['end'] + pad_end
    s = max(0.0, s)
    if duration:
        e = min(float(duration), e)
    if e <= s:
        e = s + 0.5
    return (round(s, 2), round(e, 2))


STOP = set("""a an the is was were are be been being do does did done has have had having will would
shall should can could may might must of to in on at for by with from as and or but not no nor so if
then than that this these those it its it's he she they them his her their you your yours i me my we us our
what which who whom whose when where why how any some there here up out about into over after before
again also just only very too still right correct isn't wasn't didn't doesn't don't aren't weren't hasn't
haven't won't wouldn't shouldn't patient doctor""".split())


def _norm_tok(t: str) -> str:
    t = re.sub(r"[^a-z0-9.]", "", t.lower()).strip('.')
    for suf in ('ing', 'ed', 'es', 's', 'ly'):
        if len(t) > len(suf) + 3 and t.endswith(suf):
            t = t[: -len(suf)]
            break
    return t[:6]


def content_tokens(text: str) -> List[str]:
    return [_norm_tok(t) for t in re.split(r"[\s/-]+", text) if t and re.sub(r"[^a-z0-9]", "", t.lower()) not in STOP
            and _norm_tok(t)]


def trim_to_phrases(units: Sequence[dict], span: Span, question: str,
                    tx_phrases: Optional[Sequence[dict]] = None, min_keep: float = 0.0) -> Span:
    """Shrink a cited span to the contiguous run of phrase-units inside it that
    best matches the question's content words. Returns `span` unchanged when
    nothing inside matches."""
    if tx_phrases is None:
        return span
    inside = [p for p in tx_phrases if p['start'] >= span[0] - 0.05 and p['end'] <= span[1] + 0.05]
    if len(inside) <= 1:
        return span
    q = set(content_tokens(question))
    if not q:
        return span
    scores = []
    for p in inside:
        toks = content_tokens(p['text'])
        hit = sum(1 for t in toks if t in q)
        scores.append((hit, len(toks)))
    best, bij = 0.0, None
    for i in range(len(inside)):
        h = n = 0
        for j in range(i, len(inside)):
            h += scores[j][0]
            n += scores[j][1]
            v = h - 0.25 * (n - h)
            if h > 0 and v > best:
                best, bij = v, (i, j)
    if bij is None:
        return span
    return (inside[bij[0]]['start'], inside[bij[1]]['end'])


def _subtoks(text: str) -> List[str]:
    return [_norm_tok(t) for t in re.findall(r"[A-Za-z0-9]+(?:\.[0-9]+)?", text) if _norm_tok(t)]


def align_quote(units: Sequence[dict], ids: Sequence[int], quote: str,
                min_ratio: float = 0.75) -> Optional[Span]:
    """Find `quote` (verbatim words the LLM copied) inside the cited units (+-1
    unit), else anywhere, and return its word-level span. Tokenisation splits
    hyphens/punctuation the same way on both sides; each sub-token maps back to
    its whisper word."""
    import difflib
    qt = _subtoks(quote)
    if not qt:
        return None

    def search(words):
        wt, owner = [], []
        for k, w in enumerate(words):
            for t in _subtoks(w['w']):
                wt.append(t)
                owner.append(k)
        if not wt:
            return 0.0, None
        n = len(qt)
        best, bs = 0.0, None
        sm = difflib.SequenceMatcher(autojunk=False)
        sm.set_seq2(qt)
        for L in range(max(1, n - 2), n + 3):
            for i in range(0, max(1, len(wt) - L + 1)):
                sm.set_seq1(wt[i:i + L])
                r = sm.ratio()
                if r > best + 1e-9:
                    best, bs = r, (owner[i], owner[min(len(wt), i + L) - 1])
        return best, bs

    pools = []
    ids = [i for i in ids if isinstance(i, int) and 0 <= i < len(units)]
    if ids:
        lo, hi = max(0, min(ids) - 1), min(len(units) - 1, max(ids) + 1)
        pools.append([w for u in units[lo:hi + 1] for w in u['words']])
    pools.append([w for u in units for w in u['words']])
    for words in pools:
        if not words:
            continue
        r, bs = search(words)
        if bs and r >= min_ratio:
            return (words[bs[0]]['s'], words[bs[1]]['e'])
    return None


_ELLIPSIS = re.compile(r'\.\.\.+|\u2026')
_PIECES2 = re.compile(r'\.\.\.+|\u2026|(?<=[.?!])\s+')


def quote_span(units: Sequence[dict], ids: Sequence[int], quote: str, gate: str = 'first',
               ellipsis: int = 0, fallback: str = 'first', question: str = '',
               duration: Optional[float] = None) -> Tuple[Optional[Span], Optional[str]]:
    """Quote -> (span, span_src), the pipeline's quote/units choice (Allmquote-g4-2).

    gate='first' (legacy): the aligned quote must lie within +-3 s of the FIRST cluster of the
      cited ids (clusters with max_gap=2), else it is dropped for the first cluster's unit span.
    gate='any' (R0): it may lie within +-3 s of ANY cluster of the cited ids.
    ellipsis=1 (R0b): when the cited ids form >= 2 clusters and the quote contains '...'/'…',
      align each piece (cited ids +-1 unit only, ratio >= 0.75) and keep the LAST piece that
      aligns and passes the gate (the restatement / conclusion). ellipsis=2 (R0c, post-hoc): the
      same, but also splits at sentence ends (quotes that concatenate sentences of two clusters).
    fallback (only when the quote is dropped and the ids form >= 2 clusters):
      'first' = legacy first-cluster unit span; 'qoverlap' = the cluster with the most question
      content-word overlap; 'quoteoverlap' = the cluster with the most quote content-word overlap
      (ties -> earliest); 'unit' = the FB6-1/'U' rule below (special-cased, ignores gate/ellipsis)."""
    if fallback == 'unit':
        return _quote_span_unit(units, ids, quote, duration)
    ids_ok = [i for i in ids if isinstance(i, int) and 0 <= i < len(units)]
    cl = clusters(ids_ok, 2) if ids_ok else []
    cspans = [(units[c[0]]['start'], units[c[-1]]['end']) for c in cl]
    if gate == 'first':
        cspans = cspans[:1]

    def gated(sp):
        if sp is None or not cspans:
            return sp
        for a, b in cspans:
            if not (sp[1] < a - 3 or sp[0] > b + 3):
                return sp
        return None

    span = None
    if quote:
        splitter = _PIECES2 if ellipsis == 2 else _ELLIPSIS
        if ellipsis and len(cl) >= 2 and splitter.search(quote):
            pieces = [x.strip() for x in splitter.split(quote) if _subtoks(x)]
            lo, hi = max(0, min(ids_ok) - 1), min(len(units) - 1, max(ids_ok) + 1)
            for piece in reversed(pieces):
                sp = gated(align_quote(units[lo:hi + 1], [], piece))
                if sp is not None:
                    span = sp
                    break
        if span is None:
            span = gated(align_quote(units, ids, quote))
    if span is not None:
        return span, 'quote'
    pick_ids = ids
    if len(cl) >= 2 and fallback in ('qoverlap', 'quoteoverlap'):
        ref = set(content_tokens(question if fallback == 'qoverlap' else quote or ''))
        best, bc = -1, cl[0]
        for c in cl:
            toks = [t for k in c for t in content_tokens(units[k]['text'])]
            h = sum(1 for t in toks if t in ref)
            if h > best:
                best, bc = h, c
        pick_ids = bc
    span = span_for_ids(units, pick_ids, duration=duration)
    return span, ('units' if span is not None else None)


def _quote_span_unit(units: Sequence[dict], ids: Sequence[int], quote: str,
                     duration: Optional[float] = None) -> Tuple[Optional[Span], Optional[str]]:
    """FB6-1 / the g4-1 'U' rule (Allmquote-g4-2, MED_MULTI_FALLBACK=unit): on multi-cluster
    citations, locate the cluster the quote aligns to (gate='any', hardcoded regardless of the
    caller's gate) and return that cluster's WHOLE unit span instead of the tight quote span; if
    the quote does not align anywhere, use the first cluster's unit span (same as legacy). Single-
    cluster citations are untouched (legacy quote_span). Verified positive on both halves in 7/7
    caches (medical-evolve-g4-2): dev +0.0117+-0.0066, test +0.0112+-0.0122, ~15 fires/run, never
    combined with the onset start rule before FB6."""
    ids_ok = [i for i in ids if isinstance(i, int) and 0 <= i < len(units)]
    cl = clusters(ids_ok, 2) if ids_ok else []
    if len(cl) < 2:
        return quote_span(units, ids, quote, duration=duration)
    sp, src = quote_span(units, ids, quote, gate='any', duration=duration)
    pick = cl[0]
    if src == 'quote' and sp is not None:
        for c in cl:
            a, b = units[c[0]]['start'], units[c[-1]]['end']
            if not (sp[1] < a - 3 or sp[0] > b + 3):
                pick = c
                break
    span = span_for_ids(units, pick, duration=duration)
    return span, ('unit_U' if span is not None else None)


def coverage_next_end(span: Optional[Span], words: Sequence[dict], question: str,
                      min_missing: int = 1, max_gap: float = 9.0) -> Optional[Span]:
    """FB6-2: the A-island coverage-next rule (understanding-lane cycle 2,
    medical/analysis/coverage_extend.py 'nextcov1'). Extends the span's END over the immediately
    ADJACENT NEXT sentence only when that sentence holds >= min_missing question content tokens
    the span currently lacks, and only when the span's current end already sits at a sentence
    boundary. No previous-sentence extension (that loses; LESSONS.md #43). Replay: next-only
    +0.005+-0.004 (test +0.010, dev 0), about 10-20 fires/run."""
    if span is None or not words or not question:
        return span
    s, e = float(span[0]), float(span[1])
    sent_ranges: List[Tuple[int, int]] = []
    i0 = 0
    for k, w in enumerate(words):
        if SENT_END.search(w['w'].strip()):
            sent_ranges.append((i0, k))
            i0 = k + 1
    if i0 < len(words):
        sent_ranges.append((i0, len(words) - 1))
    if not sent_ranges:
        return span
    sid = [0] * len(words)
    for n, (a, b) in enumerate(sent_ranges):
        for kk in range(a, b + 1):
            sid[kk] = n
    i = min(range(len(words)), key=lambda idx: abs(words[idx]['s'] - s))
    j = min(range(len(words)), key=lambda idx: abs(words[idx]['e'] - e))
    if j < i:
        j = i
    cur = sid[j]
    if sent_ranges[cur][1] != j:
        return span  # our end is not at a sentence boundary; do not fire
    if cur + 1 >= len(sent_ranges):
        return span
    na, nb = sent_ranges[cur + 1]
    if words[na]['s'] - words[j]['e'] >= max_gap:
        return span
    qt = set(content_tokens(question))
    cur_toks = set(content_tokens(' '.join(words[k]['w'] for k in range(i, j + 1))))
    missing = qt - cur_toks
    next_toks = set(content_tokens(' '.join(words[k]['w'] for k in range(na, nb + 1))))
    if len(missing & next_toks) >= min_missing:
        new_e = words[nb]['e']
        return (s, max(new_e, s + 0.3))
    return span


_PUNCT_END = re.compile(r'[.,;:?!]["\')\]]*$')


def _strong_start(words: Sequence[dict], j: int, pause: float) -> bool:
    """Is there a natural boundary right before word j (pause or punctuation)?"""
    if j <= 0:
        return True
    return (words[j]['s'] - words[j - 1]['e'] >= pause) or bool(_PUNCT_END.search(words[j - 1]['w'].strip()))


def _strong_end(words: Sequence[dict], j: int, pause: float) -> bool:
    """Is there a natural boundary right after word j?"""
    if j >= len(words) - 1:
        return True
    return (words[j + 1]['s'] - words[j]['e'] >= pause) or bool(_PUNCT_END.search(words[j]['w'].strip()))


def energy_onsets(pcm, sr: int = 16000, thr_db: float = -50.0, min_sil: float = 0.1,
                  win: int = 400, hop: int = 160) -> List[float]:
    """Speech-onset times (s): silence -> speech transitions preceded by >= min_sil s of
    silence, on frame RMS (25 ms window / 10 ms hop @ 16 kHz). Verified in
    medical/analysis/{acoustic,boundary_sources,replay_rules}.py (understanding lane, cycle 2).
    """
    import numpy as np
    x = np.asarray(pcm, dtype=np.float64)
    n = max(1, (len(x) - win) // hop)
    if n <= 0:
        return []
    fr = np.lib.stride_tricks.sliding_window_view(x, win)[::hop][:n]
    rms = np.sqrt((fr ** 2).mean(1) + 1e-12)
    db = 20 * np.log10(rms + 1e-9)
    mask = (db > thr_db).astype(np.int8)
    d = np.diff(np.concatenate([[0], mask, [0]]))
    starts = np.where(d == 1)[0]
    ends = np.where(d == -1)[0]
    hop_s = hop / sr
    ons = []
    prev_end = None
    for s_i, e_i in zip(starts, ends):
        sil_before = (s_i - prev_end) * hop_s if prev_end is not None else s_i * hop_s
        if sil_before >= min_sil:
            ons.append(s_i * hop_s)
        prev_end = e_i
    return ons


def onset_sentstart(span: Optional[Span], words: Sequence[dict], onsets: Optional[Sequence[float]],
                    k: int = 4, off: float = -0.04, minsil_win: float = 0.1) -> Optional[Span]:
    """B-island start rule (VERIFIED, LESSONS.md cycle 2): move the span's first word back to
    its sentence start when <= k words away (sentstart(k)), then, if an acoustic speech onset
    (energy_onsets) falls inside that first word, start there (minus the measured `off` lead)
    instead of at whisper's (often early-clamped) word start. Leaves the end untouched.
    Replay on 3 cached production reps (fast lane, 2026-09-18): pooled +0.014-0.015 dev,
    +0.011-0.015 test (offline_eval.py score, positive in 3/3 reps on both halves).
    Call BEFORE calibrate_span's shift_s (this replaces the need for a start shift).
    """
    if span is None or not words or onsets is None:
        return span
    s, e = float(span[0]), float(span[1])
    i = min(range(len(words)), key=lambda idx: abs(words[idx]['s'] - s))
    kk = i
    while kk > 0 and not SENT_END.search(words[kk - 1]['w'].strip()) and i - kk < k:
        kk -= 1
    if kk == 0 or SENT_END.search(words[kk - 1]['w'].strip()):
        i = kk
    ws, we_first = words[i]['s'], words[i]['e']
    cands = [t for t in onsets if ws - 0.1 <= t <= we_first - 0.05]
    new_s = (cands[0] + off) if cands else ws
    if new_s < e:
        s = new_s
    return (s, e)


def calibrate_span(span: Optional[Span], words: Sequence[dict], shift_s: float = 0.0, shift_e: float = 0.0,
                   snap: int = 0, snap_pause: float = 0.3, clamp_len: float = 0.0, clamp_w: float = 0.0,
                   duration: Optional[float] = None) -> Optional[Span]:
    """Post-alignment boundary calibration of a word-aligned evidence span.

    1. snap (word units): 0 = off; 1 = if an edge is not at a natural boundary (pause
       >= snap_pause s or punctuation), move it OUTWARD by up to one word to one that is;
       2 = same, but the nearer of +-1 word (inward allowed, never collapsing the span).
    2. soft length clamp: if the span is shorter than clamp_len, grow it symmetrically by
       clamp_w * (clamp_len - len) (gold spans are clause level, ~3 s median).
    3. separate start / end shifts (whisper word starts run early).
    """
    if span is None:
        return None
    s, e = float(span[0]), float(span[1])
    if snap and words:
        i = min(range(len(words)), key=lambda k: abs(words[k]['s'] - s))
        j = min(range(len(words)), key=lambda k: abs(words[k]['e'] - e))
        if j < i:
            j = i
        if not _strong_start(words, i, snap_pause):
            cands = [i - 1] if snap == 1 else [i - 1, i + 1]
            for c in cands:
                if 0 <= c <= j and _strong_start(words, c, snap_pause):
                    i = c
                    break
        if not _strong_end(words, j, snap_pause):
            cands = [j + 1] if snap == 1 else [j + 1, j - 1]
            for c in cands:
                if i <= c < len(words) and _strong_end(words, c, snap_pause):
                    j = c
                    break
        s, e = words[i]['s'], words[j]['e']
    if clamp_len > 0 and clamp_w > 0 and e - s < clamp_len:
        g = clamp_w * (clamp_len - (e - s)) / 2
        s, e = s - g, e + g
    s, e = s + shift_s, e + shift_e
    s = max(0.0, s)
    if duration:
        e = min(float(duration), e)
    if e <= s:
        e = s + 0.3
    return (round(s, 2), round(e, 2))


# --------------------------------------------------------------------------- #
# Offline analysis
# --------------------------------------------------------------------------- #

def _analysis():  # pragma: no cover - script
    import argparse
    import csv
    import os
    import random
    import statistics as st

    ap = argparse.ArgumentParser()
    ap.add_argument('--tx', default='/home/claude/work/med/tx_small.en')
    ap.add_argument('--csv', default=os.path.join(os.environ.get(
        'UPSTREAM', '/home/claude/Nordic-AI-Cup-2026'), 'medical-appointment/data/question_train.csv'))
    args = ap.parse_args()

    rows = [r for r in csv.DictReader(open(args.csv)) if r['evidence_start']]
    txs = {}
    for r in rows:
        p = os.path.join(args.tx, f"conversation_{r['transcript_id']}.json")
        if os.path.exists(p) and r['transcript_id'] not in txs:
            txs[r['transcript_id']] = load_transcript(p)
    rows = [r for r in rows if r['transcript_id'] in txs]
    print(f'{len(rows)} gold spans over {len(txs)} transcripts')

    # held-out split by conversation
    # same split as offline_eval.split_ids (seed 2026 over the audio filenames)
    ids = sorted(f'conversation_{t}.mp3' for t in txs)
    random.Random(2026).shuffle(ids)
    dev = {i[len('conversation_'):-4] for i in ids[: len(ids) // 2]}

    def gold(r):
        return float(r['evidence_start']), float(r['evidence_end'])

    for mode in ('segment', 'sentence', 'clause', 'comma', 'phrase'):
        for pause in (0.6, 1.0, 99):
            unitsd = {t: build_units(tx, mode, pause=pause) for t, tx in txs.items()}
            single, run, nunits = [], [], []
            for r in rows:
                us = unitsd[r['transcript_id']]
                g = gold(r)
                single.append(max(tiou(g, (u['start'], u['end'])) for u in us))
                best, bl = 0, 1
                for i in range(len(us)):
                    for j in range(i, min(len(us), i + 8)):
                        v = tiou(g, (us[i]['start'], us[j]['end']))
                        if v > best:
                            best, bl = v, j - i + 1
                run.append(best)
                nunits.append(bl)
            ulen = st.mean(u['end'] - u['start'] for us in unitsd.values() for u in us)
            print(f'mode={mode:8s} pause={pause:4}: units/conv={st.mean(len(v) for v in unitsd.values()):5.1f} '
                  f'mean unit len={ulen:4.2f}s  oracle single-unit tIoU={st.mean(single):.3f}  '
                  f'oracle contiguous-run tIoU={st.mean(run):.3f}  mean run len={st.mean(nunits):.2f}')

    # word-level oracle (upper bound imposed by ASR timestamps)
    wo = []
    for r in rows:
        ws = words_of(txs[r['transcript_id']])
        g = gold(r)
        best = 0
        for i in range(len(ws)):
            if ws[i]['s'] > g[1] + 2:
                break
            for j in range(i, min(len(ws), i + 60)):
                best = max(best, tiou(g, (ws[i]['s'], ws[j]['e'])))
        wo.append(best)
    print(f'word-level oracle tIoU={st.mean(wo):.3f}')

    # Padding grid on the oracle-chosen run (sentence units): tune on dev, report test
    for mode in ('sentence', 'clause', 'comma', 'phrase'):
        unitsd = {t: build_units(tx, mode) for t, tx in txs.items()}
        chosen = []
        for r in rows:
            us = unitsd[r['transcript_id']]
            g = gold(r)
            best, bij = -1, (0, 0)
            for i in range(len(us)):
                for j in range(i, min(len(us), i + 8)):
                    v = tiou(g, (us[i]['start'], us[j]['end']))
                    if v > best:
                        best, bij = v, (i, j)
            chosen.append((r, us, bij))
        grid = [x / 10 for x in range(-4, 7)]
        res = {}
        for ps in grid:
            for pe in grid:
                for part in ('dev', 'test'):
                    vals = [tiou(gold(r), (us[i]['start'] - ps, us[j]['end'] + pe))
                            for r, us, (i, j) in chosen if (r['transcript_id'] in dev) == (part == 'dev')]
                    res[(ps, pe, part)] = st.mean(vals)
        bd = max(grid, key=lambda _: 0)
        best = max(((ps, pe) for ps in grid for pe in grid), key=lambda k: res[(k[0], k[1], 'dev')])
        print(f'[{mode}] pad (0,0): dev {res[(0.0, 0.0, "dev")]:.3f} test {res[(0.0, 0.0, "test")]:.3f}; '
              f'best-on-dev pad_start={best[0]} pad_end={best[1]}: dev {res[(best[0], best[1], "dev")]:.3f} '
              f'test {res[(best[0], best[1], "test")]:.3f}')
        # signed boundary errors
        ds = [us[i]['start'] - gold(r)[0] for r, us, (i, j) in chosen]
        de = [us[j]['end'] - gold(r)[1] for r, us, (i, j) in chosen]
        print(f'[{mode}] start error (pred-gold) median {st.median(ds):+.2f}s  end error median {st.median(de):+.2f}s')

    # Trimming heuristic: oracle *sentence* run, shrunk to the phrase sub-run that
    # best matches the question's content words (no LLM, no gold used for trimming).
    sent = {t: build_units(tx, 'sentence') for t, tx in txs.items()}
    phr = {t: build_units(tx, 'phrase') for t, tx in txs.items()}
    base, trimmed, quoted = [], [], []
    for r in rows:
        us = sent[r['transcript_id']]
        g = gold(r)
        best, bij = -1, (0, 0)
        for i in range(len(us)):
            for j in range(i, min(len(us), i + 8)):
                v = tiou(g, (us[i]['start'], us[j]['end']))
                if v > best:
                    best, bij = v, (i, j)
        sp = (us[bij[0]]['start'], us[bij[1]]['end'])
        base.append(tiou(g, sp))
        trimmed.append(tiou(g, trim_to_phrases(us, sp, r['question'], phr[r['transcript_id']])))
        q = ' '.join(w['w'] for w in words_of(txs[r['transcript_id']]) if w['s'] >= g[0] - 0.1 and w['e'] <= g[1] + 0.1)
        a = align_quote(us, list(range(bij[0], bij[1] + 1)), q)
        quoted.append(tiou(g, a) if a else tiou(g, sp))
    print(f'oracle sentence run: {st.mean(base):.3f}; + phrase trim by question overlap: {st.mean(trimmed):.3f}; '
          f'+ perfect verbatim quote alignment: {st.mean(quoted):.3f}')


if __name__ == '__main__':
    _analysis()
