# Medical appointment — experiments

Score = 0.4·accuracy + 0.6·mean tIoU (tIoU averaged over the annotated-yes
questions only). All numbers are on the 39 training conversations (390 q, 195
gold spans) unless marked dev/test. The dev/test split is a fixed 50/50 split
by conversation (`offline_eval.py --split dev|test`, seed 2026). Tune on dev,
read test.

## E1 — span construction (no LLM), ASR = faster-whisper small.en int8 (CPU)

`python spans.py --tx ~/work/med/tx_small.en`

Gold spans line up with whisper word timestamps almost perfectly (median
start/end error of the best unit run: +0.00 s / +0.02 s), and they are
**sub-sentence**: usually the one clause that answers the question
("100 milligrams daily" out of "100 milligrams daily for two weeks.").

| unit type | units/conv | oracle best single unit | oracle best contiguous run |
|---|---|---|---|
| whisper segment | 37.6 | 0.617 | 0.695 |
| sentence (. ? !) + pause>1 s | 53.3 | 0.735 | 0.834 |
| clause (sentence + comma if ≥4 words) | 58.8 | 0.729 | 0.876 |
| comma (sentence + every comma) | 62.9 | 0.715 | 0.886 |
| phrase (comma + before and/but/with/so/…) | 66.8 | 0.706 | 0.921 |
| word-level oracle (best word range) | – | – | **0.950** |

- Padding: best padding tuned on dev (grid −0.4…+0.6 s) is 0/0 for every unit
  type (sentence: dev 0.824 / test 0.844). No padding.
- Merging neighbours: the best run is 1.3–1.7 units on average; merging is
  needed only for Q/A pairs across turns.
- Lexical trimming of the right sentence to the phrase that overlaps the
  question: **hurts** (0.834 → 0.777; best grid point 0.84, noise). Not used.
- Verbatim-quote alignment (the LLM copies the shortest answering excerpt,
  we fuzzy-match it to the words): with a perfect quote 0.834 → **0.944**.

Decision: show the LLM sentence units (readable, few ids), ask for EVIDENCE ids
plus a verbatim QUOTE; span = aligned quote (word timestamps) if it matches
(≥0.75 token ratio, near the cited lines), else the cited sentence run
(first cluster of ids ≤2 apart). Expected tIoU given the right answer:
between 0.83 (quote ignored) and 0.94 (perfect quote).

## E2 — heuristic answerer (fallback, CPU, no LLM), small.en transcripts

IDF-weighted content-word coverage over 1–2 sentence windows + a numbers check.
`python pipeline.py --tx-dir ~/work/med/tx_small.en --backend heuristic`, then
`tune_bias.py`:

| heur bias | yes% | dev | test | all | acc | tIoU |
|---|---|---|---|---|---|---|
| 1.0 | 0.34 | 0.536 | 0.502 | 0.518 | 0.754 | 0.361 |
| 2.0 | 0.47 | 0.616 | 0.521 | 0.567 | 0.769 | 0.432 |
| **3.0** (default) | 0.70 | **0.619** | 0.560 | 0.588 | 0.700 | 0.513 |
| 4.0 | 0.81 | 0.608 | 0.560 | 0.583 | 0.651 | 0.538 |

Baseline to beat: always-yes/no-span = 0.200; heuristic = **0.588**.

## E3 — decision threshold: do NOT target 50 % yes

Per question, answering yes earns (0.4 + 1.2·t)/N in expectation with prob p
(t = tIoU when right; N questions, half of them positive), answering no earns
0.4/N with prob 1−p. So yes is optimal iff **p > 0.4/(0.8+1.2·t)** ≈ 0.23 for
t≈0.8 (≈0.29 for t≈0.6). With a calibrated P(yes) that is a logit bias of
≈ +1.2. The heuristic's sweep (E2) confirms: the optimum sits at 70 % yes,
not 50 %. Default `MED_YES_BIAS=1.2` for the LLM; re-tune on the pod with
`tune_bias.py` (dev split), since LLM probabilities are not calibrated.

## E4 — plumbing (CPU): tiny.en ASR + mock LLM through upstream local_evaluator

`MED_ASR_MODEL=tiny.en python server.py` + `python mock_llm.py 8001` +
upstream `local_evaluator.py`: 39/39 conversations answered, 0 failures,
0 timeouts, 7.5 s mean / 14.4 s worst per conversation on 2 CPUs. Score 0.475
(mock = heuristic with a double bias; plumbing check only).
Robustness: LLM slower than the deadline → heuristic answers returned at the
deadline (29.5 s with MED_DEADLINE=30); undecodable audio / bad base64 /
ASR over budget → valid all-yes/null response, HTTP 200.

## Pod results

Since the holdout split, workers see 31 of the 39 conversations (310 q). The dev/test halves
below are the offline_eval split restricted to those 31 (15 dev / 16 test).

### E5: ASR on the pod (39 convs, before the split)
large-v3-turbo: 4.7 s mean / 10.1 s max per conversation, heuristic 0.555. large-v3: 16.9 s,
heuristic 0.530. Turbo also gives better span oracles (phrase units 0.870/0.823 vs 0.786/0.778).
Decision: **turbo**.

### E6: LLM answerer, turbo transcripts, think=0, bias 1.2 (31 convs)
| model | dev | test | all | acc | tIoU | QA s/conv |
|---|---|---|---|---|---|---|
| Qwen/Qwen3.5-35B-A3B-GPTQ-Int4 | 0.756 | 0.786 | **0.770** | 0.990 | 0.623 | 1.4 / 2.6 max |
| cyankiwi/Qwen3.8-27B-AWQ-INT4 | 0.744 | 0.795 | 0.768 | 0.997 | 0.616 | 6.6 / 10.2 max |

Sweeping the bias hardly matters (0.765 to 0.773 across biases -1 to 4) because accuracy is
saturated at about 99%. **The bottleneck is tIoU (0.62, against a 0.90 perfect-quote oracle).**
Choice: 35B-A3B, which is tied on score and 4.7 times faster. See SHARED_LLM.md.

### E7: end to end with the upstream local_evaluator on the pod (31 convs)
server.py with turbo ASR and the 35B-A3B model: **score 0.768** (acc 0.994, tIoU 0.618), 0 failures, 0 timeouts,
13.2 s mean / 25.7 s worst per conversation. The GPU was shared with drone training at 100% util,
so ASR took 3 to 12 s. The first run scored 0.324: a drone job took the GPU and whisper hit CUDA OOM,
so the server gave all-yes/no-span answers. Fix: reload and retry on a CUDA error, then fall back to CPU small.en.

### E8: span boundary calibration on turbo (medical-r1-1)
`spans.calibrate_span` runs after quote alignment. It applies separate start/end shifts (MED_SPAN_SHIFT_S / _E),
edge snapping to a pause >= MED_SPAN_SNAP_PAUSE or to punctuation within +-1 word (MED_SPAN_SNAP 1=outward,
2=either way), and a soft length clamp toward MED_SPAN_CLAMP_LEN (weight MED_SPAN_CLAMP_W).
`calib_sweep.py` replays the cached LLM output (no new calls) over the grid, on the **dev** half only.
Dev base 0.755. Best: shift_s 0.2 + snap = 0.765 (+0.0097 ± 0.0049 paired). The simplest setting within 1 SE
is **shift_s = +0.2 s alone** (0.763, +0.008 ± 0.0045). Snapping and the length clamp add nothing beyond noise
(the LLM quotes already end at clause punctuation), and end shifts are at best neutral.
Test, reported once for that choice: replay 0.7795 -> 0.790 (tIoU 0.637 -> 0.654).
A fresh pod run with the new default (new LLM calls) gave dev 0.759 / test 0.797 / all 0.777. The baseline run
the same day gave 0.757 / 0.780 / 0.767.
The default is now MED_SPAN_SHIFT_S=0.2, with the other knobs off.
Lesson: turbo word starts run about 0.2 s early (large-v3 about 0.3 s). The only calibration worth keeping is a start shift.


| exp | ASR | LLM | think | bias | dev | test | all | acc | tIoU | s/conv mean / max |
|---|---|---|---|---|---|---|---|---|---|---|

### E9: medical-evolve-g2-1: merge + redeploy + portal validation #2 (2026-09-18 ~14:00 CEST)
Branch medical-evolve-g2-1 = main + medical-r1-1 (bring-up + MED_SPAN_SHIFT_S=0.2) + a per-request JSON log in server.py
(MED_REQ_LOG, default /workspace/logs/med_requests.jsonl: asr path gpu/gpu_retry/cpu, n_llm/n_heur, span source, latency,
in-flight count; no behaviour change) + conc_eval.py (parallel load test).
- Offline, same run, fresh LLM calls: shift 0.2 = 0.7801 (0.7632 / 0.7995) vs shift 0 = 0.7646 (0.7501 / 0.7813);
  paired per-conversation dev +0.0136 +- 0.0063, test +0.0148 +- 0.0054 (both > 2 se).
- e2e (upstream local_evaluator, conc 1, 31 convs): 0.774, acc 0.990, tIoU 0.630, 8.5 s mean / 21.7 s worst, 0 heuristic answers.
- conc 2 on the first 10 convs: 0.733 vs 0.739 offline on the same 10 (one accuracy flip), p95 20.7 s (ASR serialised by the lock), 0 heuristic.
- Production :9054 now runs /workspace/med-evolve-g2-1/medical/server.py (job medg21-prod9054, log /workspace/logs/med_server_g21.log,
  requests /workspace/logs/med_requests_g21.jsonl). Rollback: /workspace/medical-bringup-1/medical/server.py.
- Portal validation #2: **0.7207** (vs 0.7152 #1). During the window: 19 requests, all 190 answers from the LLM, 0 heuristic,
  0 CPU ASR, never more than 1 request in flight (the portal is sequential), p95 8.3 s, max 8.8 s.
  So the local-vs-portal gap (0.774 vs 0.721) is not deployment (timeouts, fallbacks, concurrency). It comes from the conversation set:
  the 19 validation conversations are disjoint from the 31 practice ones, and 19 conversations carry an se of about 0.02-0.025.
  The quote-alignment failure share (span from cited units instead of the quote) is similar: 20% portal vs 18% local.

### E10: medical-evolve-g4-2 (Tllmfolded-g4-3): CPU bge locator as the per-question LLM-failure fallback + redeploy + portal validation #3 (2026-09-18 ~16:00 CEST)
Before this, a question the LLM did not answer (server dead, HTTP error, per-question timeout, deadline) got
`heuristic_answer` (turbo transcripts: 0.580 dev / 0.534 test). locator.py (medical-asr-r1-2) was not wired in.
- **Why CPU and a sidecar.** The gpu pod has a CPU quota of 7.65 cores (cgroup cfs_quota 765000/100000) shared with
  vLLM, whisper and the drone server, so bge-reranker-v2-m3 over every phrase run is far too slow on CPU
  (2500 pairs = 91-96 s; 250 pairs = 13-21 s, fp32 or dynamic int8). So `locator.lexical_topk` keeps only the K runs with
  the best idf-weighted question-token coverage, and the cross-encoder scores those K per failed question.
  venv-med has no transformers, so the model runs in `loc_server.py` (venv-vllm, CUDA hidden, fp32, 4 threads,
  :9061, job medg43-loc9061). The /predict server never imports torch for it and uses no GPU memory for it.
  fp32 CPU logits match GPU fp32 exactly (max |dlogit| 0.000 on 30 questions), and latency is 0.44 s per question at K=8.
- **Tuning (loc_tune.py, dev only; test reported once).** Full-candidate bge scores on the production turbo transcripts
  (gpu3, fp32). Span knobs used gold answers, then the yes threshold was set on dev score. The pre-registered rule was
  the smallest K within 0.01 of the best dev score:
  K=4 0.619/0.633, 6 0.617/0.635, 8 0.627/0.638, **12 0.640/0.629**, 16 0.643/0.628, all 0.645/0.612 (dev/test).
  Chosen: K=12, yes iff max logit > -2.5, span = argmax logit - 3.0*|log(dur/2 s)|, start +0.3 s.
- **Failure tests (fallback_tests.sh: pipeline.py on the pod, 31 convs, frozen offline_eval, paired.py per conversation):**
  LLM dead: heuristic 0.5799/0.5338 -> locator 0.6396/0.6294, dev +0.059 +- 0.022, test +0.102 +- 0.039 (2.6 se each),
  qa 8.0 s mean / 13.0 s max. 50% of LLM calls get HTTP 500 (flaky_llm.py): 0.674 -> 0.715 (+0.037 +- 0.017 all; 148 loc / 162 llm).
  50% of calls hang (deadline path, 5 convs): the LLM phase stops at deadline-6 s (MED_LOC_RESERVE, only while the sidecar is up),
  then 27 loc + 1 heur (out of time), max 43.8 s.
- **Happy path.** Deterministic mock LLM: old (g2-1) vs new code give byte-identical answers/spans on 31 convs (MOCK_IDENTICAL).
  Real LLM: same-run old 0.7697 vs new 0.7741, paired +0.0036 +- 0.0037, with 310/310 LLM answers. The difference is vLLM
  nondeterminism: old vs old2 differ in 14 fields, old vs new in 11.
- **e2e on a new port (:9064):** upstream local_evaluator conc 1: 0.774 (acc 0.984, tIoU 0.634), 4.2 s mean / 7.4 s worst,
  0 fallbacks. conc_eval conc 2 (31): 0.773, p95 8.9 s, 0 fallbacks. With the LLM dead, conc 2: 0.635, 310/310 loc,
  p95 18.4 s / max 21.5 s (the sidecar serialises requests).
- **Redeploy:** production :9054 = /workspace/med-evolve-g4-2/medical/server.py (job medg43-prod9054, log
  /workspace/logs/med_server_g43.log, requests /workspace/logs/med_requests_g43.jsonl, with n_loc) + sidecar job medg43-loc9061
  (log /workspace/logs/medg43-loc9061.log, or `bash medical/start_loc.sh`). If the sidecar is down, the old heuristic
  fallback applies, and the server notices within 10 s.
  Rollback: /workspace/med-evolve-g2-1/medical/server.py.
- **Portal validation #3** (uuid 584e7b1d977d447bbfdc3b8dd40bb203): **0.7209** (#2 0.7207, #1 0.7152), 0 errors;
  server log 19 convs, 190/190 LLM, 0 heur, 0 loc, p95 5.3 s.
- Not bundled: the handoff medical-fast-r5-onset (onset+sentstart4 start rule) fails the >2 se gate on test
  (pooled +0.0211 +- 0.0117 = 1.8 se), and its live-audio decode path has not been verified end to end.

### E11: medical-evolve-g1-1-onset (I1Perquest-g1-1): onset + gate-any as code defaults, SHIFT trap removed in code, redeploy + portal validation #5 (2026-09-18 ~16:50 CEST)
- **Code.** main + medical-fast-r6-2 (spans.quote_span / onset_sentstart / coverage_next_end + knobs). Defaults: MED_SPAN_ONSET_SENTSTART=1,
  MED_QUOTE_GATE=any (multi_fallback=first, coverage_next=0 unchanged). `_request_onsets` returns the onsets or None; with onsets the
  effective start shift is 0.0 whatever MED_SPAN_SHIFT_S says (override: MED_ALLOW_SHIFT_WITH_ONSET=1); with None (no PCM, energy_onsets
  error, < MED_SPAN_ONSET_MIN_RATE=2 onsets/10 s) the legacy +0.2 applies, with a WARNING and `ONSET_STATS` (exposed on GET /api).
  The bge fallback path keeps loc_shift_s 0.3 and never gets the onset rule. env.sh no longer pins MED_SPAN_SHIFT_S=0 (it would disable
  the legacy fallback). pipeline.py --tx-dir now decodes the audio for onsets (same decode as serving); --save-tx drops `_pcm`.
  The request log (MED_REQ_LOG) now has onsets_n and per-question a/p/ids/quote/span/src.
- **Regression (mock LLM = cached replies of the 6 fresh fr5 runs, understanding-lane deploy_check.py):** default code == r6-2 with the
  deploy env (JSON identical, 930/930 yes spans == replay onset+any); explicit MED_SPAN_SHIFT_S=0.2 still 930/930 (trap gone);
  no-PCM + gate=first == production byte-identical. vs production: dev +0.0141+-0.0037, test +0.0153+-0.0071, all +0.0147+-0.0040.
- **Live decode path:** pod venv-med faster-whisper 1.2.1 / av 18.1.0. Live /predict (:9074) on samples 19/23/57: onsets fired 3/3
  (49/49/105), 16/16 served yes spans equal the offline replay (ffmpeg PCM, cached turbo) to 0.000 s. Onset cost 0.1-0.25 s/conv (onset_timing.py).
- **Fresh paired run (liveness only):** pipeline.py on cached turbo: 0.792 (0.767 dev / 0.820 test), 310/310 LLM; replay of prod on the same
  replies: dev +0.0150+-0.0041 (3.7 se), test +0.0179+-0.0089 (2.0 se), 25/31 conversations gain.
- **e2e (upstream local_evaluator, conc 1, :9074):** 0.793 (worst 29.8 s during a concurrent paired_run by another worker, ASR 9-21 s) and
  0.792 (acc 0.984, tIoU 0.665, mean 10.8 s, worst 17.6 s); 0 heuristic, 0 loc, onsets 31/31.
- **Redeploy:** :9054 = /workspace/med-g11onset/medical/server.py (job medg11-prod9054, log /workspace/logs/med_server_g11_prod.log,
  requests /workspace/logs/med_requests_g11_prod.jsonl); replaced the orchestrator's onset-only deploy (/workspace/med-prod-onset, validation #4 0.7423).
  Sidecar :9061 unchanged. Rollback: /workspace/med-prod-onset/medical/server.py (onset only) or /workspace/med-evolve-g4-2/medical/server.py.
- **Portal validation #5** (uuid be5b3b4e8e69400bb1235f858e85d7ae): **0.7441** (#4 onset-only 0.7423, #3 0.7209), 0 errors; 190/190 LLM,
  0 heur, onsets fired 21/21, mean 4.6 s / max 10.5 s. Recorded, not tuned on.
