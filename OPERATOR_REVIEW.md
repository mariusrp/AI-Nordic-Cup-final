# Final preparation — 20 September 2026

Use the team stack as the final candidate for all three cases. Retain its live
configuration while validating delivery. The existing local workspace contains
useful experiment controls and operating tools, but no demonstrated inference
change that justifies altering the rehearsed stack before the deadline.

This is a snapshot of `phils0n/AI-Nordic-Cup` at
`b99ebb33f0d8a7f1383386209aeed3d842a734c4`, with fresh Git history. All prediction
code and the included ft5r checkpoint are unchanged. The source file hashes are
in `docs/SOURCE_SNAPSHOT.json`; additions and operator-tool changes are described
below. The original clone and local workspace remain available separately.

## Which implementation to use

| Case | Evidence and decision |
| --- | --- |
| Survival | Keep team `lin_map.py`. It incorporates lineage selection, senescence conversion and late shared-map foraging, with documented 64+ seed experiments and a three-game HTTP rehearsal averaging 1208.1. The local ration policy's 1057.0 was a selection result, not a comparable independent score. There is no fresh head-to-head proving the size of a team-stack advantage. |
| Drone | Keep team ft5r, one detector, no verifier or second-model routing, `DRONE_NEW_THR=0.15`, replay/recording/survey disabled. Its strongest transfer evidence is the split-trained recipe's 0.284 versus 0.188 on unseen instances, not ft5r's contaminated validation score. The local Modal code and saved results do not establish a better unseen-city option. |
| Medical | Keep the team's local Qwen/Whisper pipeline, rules v3 and continuation, clause trim off. The local workspace contains organizer examples rather than a competing custom medical system. |

The team code is the better-supported final candidate, not a guarantee of better
scores on the hidden data. Do not select models using the validation leaderboard.

## Local workspace review

The local survival status document was stale: its follow-up experiments have
finished. Raw summaries are preserved in `docs/workspace_evidence/`.

- Lean confirmation, 24 fresh seeds: 968.4 versus ration 944.5; paired gain
  +23.9, 95% bootstrap interval [-100.9, +153.0], 11/24 wins. Do not adopt.
- Patch navigation, eight seeds: 940.7 versus ration 1065.2. Patch plus birth
  changes: 961.0. Neither passed its screening threshold.
- Older local "team" graft screen, eight seeds: 979.2 versus 932.3, interval
  [-127.7, +250.9]. This is not a comparison against the current team `lin_map`.
- Local drone saved direct-Modal validations include 0.0851, 0.1052, 0.0796 and
  0.1079. These scores are diagnostics for those configurations and transport
  paths; they do not establish transfer superiority over ft5r.
- Local drone service has useful request caching, bounded per-sequence state,
  hash-based capture paths and inference-time headers. These are candidates for
  later hardening, not score improvements already validated on the team stack.
- Local experiment/release tools freeze code, configs, seeds and hashes. Preserve
  that provenance practice. Their `Controller` interface and relative imports
  differ from the team's `Hivemind`; copying them directly would not work.
- Do not copy the local HTTP benchmark unchanged: it hardcodes the old 600-second
  survival budget. The local drone latency probe reuses HTTP connections, unlike
  the evaluator, and its timing is from the operator's location.
- Local Modal README comments differ from the code: production sets sweep and
  tracking on; two ASGI paths permit zero warm containers. The direct ServeFast
  path does request one warm container. Verify the deployed path explicitly.

## What was carried over

1. Environment-based portal credentials (`NORDIC_API_KEY`) in `tools/portal.py`,
   with the existing external secret-file convention retained as a fallback.
2. Frozen provenance: SHA-256 hashes of the complete source snapshot and selected
   local experiment summaries, without copying unrelated models, caches or keys.
3. Validation report persistence and queue-excluded wall-time reporting. The
   portal helper accepts only status, verify and validation routes; it has no
   final-submission path. Its reports default to ignored `operator-results/`.
4. Ignore rules for local environment files, private keys and generated reports.

No local policy, detector, tracker, camera strategy, ensemble or prompt was grafted
into the team's prediction paths. Such a change needs independent evidence and
another exact-configuration rehearsal.

## Current live checks

At approximately 13:30 CEST, both required pods were running in Sweden. RunPod
balance was $7.87 at $0.843/hour (about 9.3 hours). GPU utilization was 0%, root
disk had 7.6 GB free, and local vLLM and bge backends answered. Survival's server
reported zero errors and approximately 3.1 ms average processing time. These are
point-in-time observations, not guarantees through the final.

Survival's most recent completed portal validation at intake had score 1174.08,
zero errors, but 1912.3 seconds between started_at and finished_at. The supplied
score-times-ten estimate gives **162.9 ms/tick**, above the operational cutoff.
Server processing time alone cannot explain or clear this. A fresh controlled
validation and request-count telemetry are required before recommending its final.

Fresh validations were started sequentially: drone, medical, survival. Current
results are in `docs/VALIDATION_RESULTS.json`:

| Check | Result | Readiness |
| --- | --- | --- |
| Drone portal | 0.5430, zero errors, 84.6 s; **241/249 frames** answered; server median 70 ms, max 141 ms | Below the 248–249 frame target. Do not equate zero errors with complete delivery. |
| Medical portal | 0.7795, zero errors, 89.2 s | Pass |
| Medical training load test | 39 conversations / 390 answers, zero errors, mean 4.65 s, p95 6.35 s, max 7.97 s; zero yes-without-span answers | Pass |
| Survival portal | Queued 13:40 CEST, attempt `f3f6403582bd4812a26bf2dd12466515` | Pending; earlier slow run remains unresolved |

Drone missed frames 53–57, 127–128 and 183. The logged inference times around
these gaps were 59–85 ms. The cause lies outside those recorded processing
durations; transport, request scheduling and evaluator timing remain possible.
No model change is justified by this delivery result. A repeat after a concrete
delivery check is the next step, rather than changing detection thresholds.

Read-only inspection confirmed the running drone process has ft5r, an empty
second checkpoint, threshold 0.15, empty replay and recording paths, and replay
fast/survey disabled. The rules medical process has aeq, island gap 0.15/threshold
-35, rules v3, continuation, coverage-next and quote use disabled, matching the
handover. No model-serving processes were restarted. No finals are authorized.

## Final configuration and operator actions

Use these existing prediction URLs after their readiness checks pass:

- Survival: `http://194.68.245.133:22164/predict`
- Drone: `http://194.68.245.26:22174/predict`
- Medical: `https://5g74a0atgzlxwf-9054.proxy.runpod.net/predict`

Read HANDOVER.md for exact restart commands. Its obsolete "drone must be switched"
and "clause trim accepted" statements are superseded by its final configuration
sections and the operator's current instructions. Medical `env.sh` alone does not
apply the final settings; survival without `POLICY` selects v1; drone defaults may
enable recording under `/workspace`. Use explicit final settings when rehosting.

The medical `/api` check initially returned 403 to Python urllib without a custom
user agent, but curl returned the correct backend health response. The real portal
validation is the relevant client-compatibility check.

One validation per case checks the exact served system; it does not establish
generalization. Require survival mean transport time below roughly 35–40 ms/tick;
drone 248–249 answered frames and zero errors; medical zero errors and sufficient
latency margin. The additional 39-training-conversation medical load test checks
robustness across the full training set; it is not a model-selection exercise.
Run final cases sequentially, drone before medical, with the shared host quiet.
The operator must explicitly authorize each irreversible final submission.

## Reproducing locally

The organizer checkout already present at `../competition` is clean at
`acfc31a4003a5f91bf11032a02cd98c178ddbd7e`. Set `UPSTREAM` to its absolute path
when using tools requiring organizer DTOs or evaluators. On another machine,
clone `https://github.com/amboltio/Nordic-AI-Cup-2026.git` separately and pin the
appropriate reviewed revision. Do not copy an embedded `.git` directory here.

The 18 MB `drone/weights/ft5r_last.pt` is included and matches the handover MD5
`4898c60e64844e3e49127106335eff01`. Medical's large model weights, the installed
GPU environments, pod control state and training images are external dependencies.
Creating this repository does not recreate or move the live servers.

With `NORDIC_API_KEY` already set in your shell, validation commands are:

```sh
python3 tools/portal.py validate drone http://194.68.245.26:22174/predict
python3 tools/portal.py validate medical https://5g74a0atgzlxwf-9054.proxy.runpod.net/predict
python3 tools/portal.py validate survival http://194.68.245.133:22164/predict
```

Wait for each to finish before starting the next. Do not run these again while
the operator's current validation or final is active.
