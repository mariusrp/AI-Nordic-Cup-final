# Brief for the teammate who hosts and submits the finals

Deadlines today (20 Sep 2026): **16:00 CEST** submit the three endpoints; **20:00 CEST** top-5 teams hand in
training code + trained models (this repo is that hand-in — the final drone weights are `drone/weights/ft5r_last.pt`).

API keys are NOT in this repo. Get the RunPod key and the competition portal token from Adrian directly.

## Paste this to your assistant

You are helping team Phillips finish the Nordic AI Cup 2026 (submission deadline Sun 20 Sep 16:00 CEST). Your
operator hosts the final servers and submits. Read HANDOVER.md and LESSONS.md in this repo, and review it together
with your operator's own repo.

ONLY THE FINAL COUNTS. Each case is one attempt on unseen data: survival = 3 games AVERAGED, drone = a different
250-frame flight over an unseen city, medical = 38 unseen conversations. The validation leaderboard uses different
data and decides nothing — never tune against it. Benchmark final from a strong team: 870.77 / 0.234 / 0.800, from
validations of 1903 / 0.950 / 0.895. Our expectation: ~1280 / 0.20-0.35 / 0.72-0.78.

RULES: one attempt per case, irreversible — submit only when your operator says so. No cloud/hosted API inside any
/predict path (local models only at serve time). Validation runs are unlimited; use them.

HOSTING AND LATENCY (measured, matters wherever you host)
- The evaluator calls from FINLAND: Cf-Connecting-Ip 46.62.240.126, Cf-Ipcountry FI (Hetzner hel1), python-httpx,
  a NEW TCP connection per request. Host close to Helsinki, plain HTTP on a direct port (TLS/tunnels cost a
  handshake per request; a Cloudflare tunnel measured ~90 ms/tick, an HTTPS proxy once added ~400 ms and scored 0).
  Reference: ~40 ms/tick from AWS Stockholm, ~67 ms from Azure Sweden, 29-43 ms from our Stockholm pod.
- survival: CPU only, ~3 ms/tick, no weights — trivial to host anywhere. The simulator waits 10 s per reply and ENDS
  the run once it has spent 1200 s waiting in total: a 3000 s game is 30000 ticks, so stay under ~35 ms/tick.
- drone: needs a GPU AND bandwidth. The evaluator emits a frame every 333 ms and silently SKIPS any frame whose
  previous reply is still in flight (skipped frames are not errors but score as misses). Each request carries a
  ~1.3 MB base64 PNG, so at 100 Mbit/s the upload alone is ~100 ms; inference is 65-86 ms on a quiet GPU.
- medical: 60 s per conversation and we use ~4.8 s, so latency is irrelevant; it needs the heavy stack
  (Qwen3.5-35B-A3B-GPTQ on vLLM ~32 GB + faster-whisper large-v3-turbo).
- WARM UP every model and resolution path before the first real frame: a cold drone server lost 17 of 249 frames
  (0.509 vs 0.580 for the identical stack once warmed).

VERIFY EACH ENDPOINT WITH A REAL VALIDATION BEFORE SUBMITTING
- survival: from `tools/portal.py history survival 5`, ticks = score*10, ms/tick = 1000*wall/ticks — reject > ~40.
- drone: count frames the server answered (grep the log for "f<N> idx<N>"): want 248-249 of 249, errors 0.
- medical: run medical/conc_eval.py over the 39 training conversations — errors 0, p95 well under 60 s.
Then submit one case at a time, keep the host quiet during a run, and run drone before medical if they share a GPU.

ALREADY TESTED AND REJECTED — don't redo: detector ensembling/WBF (+0.006 off-city, 2.1-2.4x false positives, p90
443 ms > budget); an extra drone training run with empty-terrain negatives (lost 0.088 on real off-city data);
medical rules-v4, self-consistency, contraction verifier, and a clause trim (gained +0.011 on training data but lost
all three interleaved portal pairs on unseen data — stays off). The NVIDIA Kaggle-playbook article is reviewed:
pseudo-labeling and diverse baselines are already in use; its hill-climbing advice assumes validation and test share
a distribution, which is false here and is exactly what turns a 0.95 validation into a 0.234 final.

IF YOU ADOPT ANYTHING FROM EITHER REPO: judge it only on data it did not train or tune on. Traps we measured — our
offline drone scenes are saturated (all checkpoints 0.983-0.986 while real scores span 0.146-0.443; use mAP@0.5:0.95
and false-positives-per-frame); survival per-seed results are not reproducible across processes (16-seed paired se is
55-70 with identical code — use 64+ seeds); the medical portal has ±0.005-0.01 noise on 19 conversations. A survival
policy change needs a fresh 3-game rehearsal before it can be trusted.

FALLBACK: our pods are live and already in final configuration if you'd rather not re-host —
survival http://194.68.245.133:22164/predict, drone http://194.68.245.26:22174/predict,
medical https://5g74a0atgzlxwf-9054.proxy.runpod.net/predict (restart commands and rollbacks in HANDOVER.md).
