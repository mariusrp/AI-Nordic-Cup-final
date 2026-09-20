# Medical big-bets lane

## Researched approaches
- **BB1: annotator-unit ranker.** Annotators mark 1-3 whole ASR sentences, often across a speaker change. The idea: enumerate sentence runs and clause runs within +-3 s of the greedy LLM quote, keep the quote itself as a fallback candidate, and let a fine-tuned cross-encoder pick the run with the highest expected tIoU. The 35B LLM still gives the yes/no answer and the anchor.
- **BB3: extractive QA span model.** Fine-tune a SQuAD2 reader on the gold spans over the word stream, which includes pause and speaker tokens. Decode with dense start/end probabilities x a Gaussian prior around the LLM anchor x a sentence-boundary bonus x a length prior (gold median 2.7 s).

## Critique and ranking
BB1 ranked first because it is cheap, reuses the LLM anchor, and matches how annotators work. BB3 ranked second because it gives a dense boundary prior but has little training data (31 conversations). The main risk for both: selecting from a candidate pool has been unsolved in evolve (the C span-scorer track).

## Built and scored (frozen scorer, offline_eval.py)
| Bet | Iteration | Score | Base | Outcome |
|---|---|---|---|---|
| BB1 | it3: pooled 31-conversation LOCO gate-first re-selector (LoRA Qwen3-Reranker picker), pre-registered in bb1_pool.py (f553011) | 0.7874+-0.0062 | 0.7801 | not better than the best; discarded |
| BB3 | it3: extractive reader over the no-sampling unit pool | 0.7749+-0.0127 | 0.7672 | dead end |

The it2 gate for BB1 was trained on one half only. The halves have different bad-greedy base rates (dev 0.37, test 0.25), so that gate almost never fired. Pooled LOCO fixed that, but the gain (+0.007) is still below the 2 se bar on both halves.
BB3's commits are on medical-bigbet-2-it3 (084a06a), which descends from medical-bigbet-2. Nothing was handed off or audited, and pipeline.py is unchanged.

## Should either become an evolve island?
No. Both fold into the existing lesson: choosing among a candidate pool does not beat the greedy LLM quote by 2 se with 31 conversations of supervision. Reopen only with a new source of labels or a much stronger picker signal.

# Round 2: timing big bets (BB-T1, BB-T3)

## Researched approaches
- **BB-T1: find the annotation timestamp source, then fit a LOCO start-edge ensemble.** All 154 visible gold edges sit on a 0.02 s grid, which is Whisper's DTW resolution. Gold ends match turbo word ends: 91.6% are within 0.1 s, and the median offset is 0.00. Gold starts match turbo word starts in only 26.6% of cases, but faster-whisper small.en word starts in 74%. Plan: identify the source, rebuild the start edge from that source's time for the same first word (matched by text), then fit a LOCO regressor over several timing sources. Ends stay on turbo.
- **BB-T3: learned frame-level annotator-start detector.** A small head over frozen encoder frames at 20 ms predicts where an annotator starts a span. Use it two ways: to snap the production start, and as a new signal for the BB1 unit-pool re-selector.

## Critique and ranking
BB-T1 ranked first. It has a measurable target (the source that generated the labels) and costs no LLM calls. BB-T3 ranked second. It has only 154 positives, and it only helps selection if the detector is strong. Risk for both: start error might be only a small part of the tIoU loss, so even a correct fix might not clear 2 se.

## Built and scored
| Bet | Iteration | Score | Base | Outcome |
|---|---|---|---|---|
| BB-T1 | it2: closed the source bank (turbo is the only candidate); tested onset and segment-start snap arms on 5 pred files | dev only, test <= 1.7 se | - | discarded |
| BB-T1 | it3: pre-registered single-cell arm S3 (segment-start snap, w=0.5, 56f1aaa) on 5 unseen pred files (1a3eceb) | 0.7735+-0.0032 | 0.7632 | discarded: test +1.6-1.8 se on 5/5 files, below the 2 se bar; dev 3.1-3.4 se |
| BB-T3 | it1-3: start head (snap) | EDGE start head dev +0.0072 (3.5 se) | 0.7801 | dev only |
| BB-T3 | it4: learned END head, EDGE-ALL, nested-LOCO variant selection; pre-registered at 6c6ce6c and 6dd0423, results at e014c9f | 0.7858+-0.0022 | 0.7801 | discarded: END head null; test +0.0041 (0.98 se); the independent re-score did not pass |

Branches: medical-bigbet-1-t1-it3 (it contains it2 and t1) and medical-bigbet-2-bbt3-it4 (e014c9f; bbt3-it3 was fast-forwarded to the same commit). The base branches could not be committed to, because other worktrees had them checked out. No handoff was submitted, and production and pipeline.py are unchanged.

## Outcome and island recommendation
Neither bet passed. Both show the same pattern: consistent, pre-registered start-edge gains of about +0.004 to +0.008 that clear 2 se on dev but not on test. The start edge is a real but small lever. It is too small to clear the bar alone at this sample size.
- BB-T1: folded. Do not make it an evolve island. Keep the lesson that the gold grid is Whisper-DTW and that ends are already turbo-exact. The only use left is to stack S3 on top of a future selection win.
- BB-T3: closed after 4 iterations. Not an island. The END head is null. The start head adds nothing beyond S3-class snapping, and as a selector feature it gave no significant lift.

# Round 3: location arbiter (BB-R3a, UPR question likelihood)

## Researched approach and critique
Wrong location is the largest remaining loss (about 0.070 of score). The questions are paraphrases generated from the gold passage, so log P(question | window) (UPR / noisy channel) should favour the gold window. UPR alone as a picker was already null. The bet was to use it only as an arbiter: keep the production 35B span unless another proposer puts a non-overlapping span elsewhere AND the UPR top window (Qwen3-8B-Base, lam=1 per extra sentence, maxL=3, fixed) overlaps that alternative. No fitted parameters. Pre-registered before each scoring run (d92c8a3, ab8e226, bfd304d).
The main worry going in was sparsity. Disagreements only happen where the proposers disagree, so the ceiling is small.

## Built and scored (frozen offline_eval.py, paired, 3 fresh runs plus 2 cached fr5 runs; base r6-2 0.7973)
| iter | proposers | result | verdict |
|---|---|---|---|
| it1 | 35B prompt arms 1 and 3 | about +0.0036 (1.3-1.5 se), 0 bad switches, about 4 fires per run on 5 unique questions | safe, underpowered |
| it2 | + zero-LLM bge-reranker locator | A -0.0070+-0.0091 (dev +0.0048, test -0.0181); 6 better / 5 worse | KILLED |
| it3 | + 35B's own other cited clusters | D +0.0005+-0.0042 (0.7978); bad switches, e.g. 37_q03 -0.94, 79_q03 -0.91 | KILLED, bet closed |
Branches: medical-bigbet-1-r3a, -r3a-it2, -r3a-it3 (code in medical/bigbets_r3a/).

## Outcome and island recommendation
No winner and no handoff. UPR is only safe among the LLM's own prompt-arm proposals, and there it fires too rarely to reach 2 se. Retriever proposers are wrong in the same places UPR is, because both latch onto generic lexical matches. UPR also ignores negation and polarity, so on the LLM's alternative clusters it picks the lexical mention over the confirming answer.
- BB-R3a: closed. Not an evolve island. Any future location arbiter must be polarity-aware (an LLM P(yes) verifier), not question likelihood. 8B lazy scoring also costs 3.4-3.9 s and 17-19 GiB, and POD=gpu has no room for that.
