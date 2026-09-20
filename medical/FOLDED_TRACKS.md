# Medical tracks folded into medical-evolve (13:15). Each is its OWN island; keep its line of approach.
Islands share only their best result and condensed lessons, once per generation (at the meta-review).

| Island (folded track) | Thesis / line of approach | Champion | Score | Status |
|---|---|---|---|---|
| T-llm (medical-main) | LLM verifier: Qwen3.5-35B-A3B GPTQ via vLLM (POD=gpu :8001), P(yes) per question, quote -> word-timestamp span | branch medical-bringup-1 (head cd8bd76; production server runs from it) | 0.768 end-to-end on pod; 0.715 validation | NOT in main yet: handed off for merge. r1 ideas (word-boundary calibration of quote spans; CTC forced alignment) interrupted, WIP on medical-r1-1/2 |
| T-asr (medical-asr) | Timing + evidence localisation without LLM: ASR choice, forced alignment, cross-encoder/reranker evidence locator | medical-asr-r1-2 | 0.629 +- 0.022 vs 0.521 (reranker as no-LLM answerer, +0.11) | reranker doesn't beat LLM verbatim quotes for boundaries; cheap win: global time shift for large-v3 timestamps; r1-1 ASR ceiling WIP |
