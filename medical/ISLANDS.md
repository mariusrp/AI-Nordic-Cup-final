# Medical islands (medical meta-reviewer, restarted evolve workflow)

Generation 2 of 2, 2026-09-18 ~20:45 CEST, lean token mode. The deadline is Sun 20 Sep 16:00 CEST. **STALLED:** the best score has not improved since 17:34 (previous workflow g3, the new workflow's g1, and restart g1).

**How candidates are scored.**
- Use the frozen `medical/offline_eval.py` on the 31 visible conversations, paired against a base from the SAME run (paired_run.py + paired_pool.py; zero-LLM rules via deploy_check.py or the dc_deploy.json replay).
- Accept only a gain > 2 se on BOTH halves, with the setting fixed before test is scored.
- Compute: POD=gpu, as a light client of the shared vLLM :8001. Never restart :8001 or :9054; use spare ports for any experimental server.

**Overall best (unchanged).** main 668bfdc, served from /workspace/med-g11onset on :9054.
- Paired offline: 0.792 +- 0.0049 (0.767 dev / 0.820 test).
- e2e 0.792-0.793. Portal #5 0.7441.
- Closest candidate: aeq, +0.0095 +- 0.0052 (1.84 se all; dev 1.1 se, test 1.4 se).

**Where the loss is.** Wrong location is 30% of the lost tIoU, and gold is usually a LATER restatement. Extent errors are next: start late 21%, end early 16%. All of it is LLM-side.

**Pending handoffs:** none (`tools/handoff.py list medical` is empty).

| island | thesis (line of approach) | champion (branch/file) | score | executed g1? | status |
|---|---|---|---|---|---|
| **A. Per-question LLM quote (production line; absorbs folded track T-llm / medical-bringup-1)** | One greedy call per question gives P(yes) plus a verbatim quote, aligned to turbo word times with onset starts. The submission comes from here, and every island stacks onto it. It owns the decoding format and the prompt content. | main 668bfdc (:9054). Candidate: **medical-evolve-g2-1-aeqconf 9e66972** (aeqmc, g2 rerun) | 0.792 prod; g2 aeqconf 0.8043 +- 0.0052 vs base 0.7929 (+0.011, still < 2 se on both halves) | yes (aeq) | alive, leading candidate. Next: 2 more fresh paired reps of aeq vs eqa (pool 4) with the threshold fixed from dev. If it clears 2 se on both halves, merge and validate once. Optional stack: the external wording "minimal clause with subject and verb" (+0.0095 over 3 runs in the other team) as a separate arm, and early stop on ANSWER: no. MED_SPK (8425265) is still unscored and low priority. Owes the locfb merge (76ec81b) |
| **B. Comparative location re-selection (pairwise / exclusion)** | Pointwise scores saturate (verifier 143/162, heat map 3/17), so ask comparatively, and only where greedy samples disagree (greedy tIoU there is 0.45-0.63). (i) Pairwise "which of passage A / B states the answer to Q", both orders; switch only on an order-consistent win. (ii) Exclusion re-query: mask the cited cluster and ask for a more explicit (often later) statement or NONE, then compare pairwise. The heat caches from D (medical/analysis/heat/) can seed the candidates. | medical-fast-r3-2-fr3b (FR3-B1 +0.0073, 1.2 se, 8/4); sampling plumbing in medical-fast-r4-2-fr4b | +0.0073 +- 0.0061 (best attempt); g2 pairctx failed probe gate (medical-evolve-g2-2-pairctx) | yes (g2, gate fail) | stalled, but the core idea has never run. Gate: >= 146/162 on the cached multi-cluster probe AND <= 2 worse uq, before any end-to-end run |
| **C. Edge judge (span extent from the LLM)** | The start ceiling is real: +0.033 with pause-segment (pg) candidates (23 new uq), and pg is judged well (AUC 0.903) while sp is not (0.739). The last idea is a comparative judge that shows span X and X-plus-prefix side by side, both orders, pg only. The second team found that gold starts sit just before the acoustic onset and whisper ends are near-perfect, so end work is dead. | medical-evolve-g1-2-sp 297ec21 (analysis/edge_prev.py); edge_verifier.py (fde1ff8) | best +0.0043 (0.9 se); dev-only +0.0023 (1/1 uq) | **NO -> priority 3** | stalled. This is the final attempt: replay only; retire unless a dev threshold has zero worse uq |
| **E. T-asr: LLM-free localisation (folded track medical-asr)** | ASR, alignment and a cross-encoder/bge locator with no LLM in the loop. It is the insurance path when :8001 is down or out of time, and the only island independent of the LLM. | medical-evolve-g3-2-locfb 76ec81b (bge :9061 + onset starts); medical-asr-r1-2 | fallback 0.640/0.629; locfb +0.0034 dev / +0.0040 test on the fallback path | **NO -> priority 2 (cheap)** | alive (insurance). Next: merge locfb into main, then deploy_check 930/930 (happy path byte-identical). No portal spend |
| D. Dense LLM evidence scan (heat map) | Score every sentence with a 1-token prefix-cached call and propose out-of-cluster windows. | medical-evolve-g1-1-dense e6a3713 | 0.798 +- 0.0037 vs 0.7996 | yes | **retired**. Gate 3/17 (needed >= 6) and the dev margin came out negative. Its caches go to B as a candidate source |

## Retired / folded
- **D dense heat map:** retired in g1 (above).
- **Folded track T-llm (medical-main / medical-bringup-1):** its line is the per-question LLM quote. It was merged long ago and is continued as island A.
- **Adaptive thinking:** 96-107 s per question, so it is infeasible. The idea lives on in B without thinking.
- **Joint conversation-level decoding:** 0.7891 vs 0.7951. Acoustic re-timing was harvested by onset (CTC on top of onset is dead).

## Gen 2 plan (rotation: B, E and C did not execute in g1)
1. **B:** pairwise both-orders plus exclusion probe on the 162 cached multi-cluster cases (gate above).
2. **E:** locfb merge + deploy_check. No score change is expected; this is insurance.
3. **A:** aeq confirmation reps. This is the only candidate near the bar, so run it as soon as a slot is free, with the threshold fixed on dev.
4. **C:** comparative pg judge replay, then retire it if it fails.

## History
| gen | best | champion | note |
|---|---|---|---|
| previous workflow g1-g3 | 0.792 offline / portal 0.7441 | medical-evolve-g1-1-onset | onset + gate-any landed; verifier, edge, joint, restate, CTC all < 2 se |
| new workflow g1 (19:05) | 0.792 | main 668bfdc | C killed at gate 2; spk unscored; B and D not run (cost pause) |
| restart g2 (~21:30) | 0.792 | main 668bfdc | A aeqconf 0.8043 vs 0.7929 (<2se, discard); B pairctx failed probe gate; no merge |
| restart g1 (20:20) | 0.792 | main 668bfdc | D re-seeded as a dense scan; thinking folded into B |
| restart g1 results (20:36) | 0.792 | main 668bfdc | D dense: 3/17 gate, 0.798 vs 0.7996 -> retired. A aeq: +0.0095 +- 0.0052 (dev 1.1 se, test 1.4 se), the closest yet -> confirm |
| restart g2 (20:45, this review) | 0.792 (se 0.0049), portal 0.7441 | main 668bfdc | stalled. Islands A, B, C, E (T-asr) active; D retired; B/E/C have rotation priority |
