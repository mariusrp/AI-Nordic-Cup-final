# Drone islands (approach families), maintained by the drone meta-reviewer

Low-token restart, generation 1 of 2 (2026-09-18 ~19:55 CEST). Compute: POD=gpu (PRODUCTION pod) with ALLOW_GPU_MAIN=1, INFERENCE ONLY. gpu3/gpu4 are stopped.
- POD=gpu state at 19:55: 31.9/49 GB GPU used, load ~10 (medical vLLM + medical :9054 + survival + drone prod). Drone A/B servers on spare ports only, <= 10 GB each, no training. Never touch medical :9054, vLLM :8001 or the port-22 drone server (:22174).
- Assets on POD=gpu: /workspace/venv-drone, /workspace/drone_weights/{v2,r11}.pt, verifier bank /workspace/drone_evolve_prod_g22/drone/bankAll.pt (and drone_evolve_prod), DINOv2 in /workspace/i4/hf.
- NOT on POD=gpu: bigbet-3 weights (gpu4 /workspace/bb3) and the g41 re-look code (gpu3 /workspace/g41; branch drone-evolve-g4-1 has no re-look commit). Skip bigbet-3; re-implement re-looks from scratch if run.
Merge authority for drone: drone-evolve. Condensed lessons: LESSONS.md "Drone".

OVERALL BEST (production): main 435b191 = v2 + r11 routed (helicopter, jet_plane, small_plane) + SAFER + I4 verifier (VERIFY_A 0.5). Validation 0.1459+-0.0105 (paired +0.022+-0.006 over 0.1236). Holdout 0.871 (tie). Leaderboard top 0.69. Deadline Sun 20 Sep 16:00 CEST.

## STALLED: YES
Best 0.1459 since ~14:45: 5 flat generations (old g3, old g4 killed, relaunch g1, relaunch g2 stopped at 19:40 with no rows, this restart). Only platform pairs count this generation; no more screens without a platform follow-up.

## Gen 1 plan (one run per island, <= ~40 tool calls each, platform pairs first)
- **R1 (I4), top priority:** DRONE_VERIFY_A=0.7 split-frame platform A/B vs production (DRONE_SPLIT_FRAME=181, B arm env only) on two spare-port servers on POD=gpu. >= 3 interleaved same-session pairs; check /predict and log p50/load per run. Win > 2 se: hand off for merge + redeploy. Else close handoff drone-fast-r2-1-fr2c as REJECT and declare the I4 knob space closed.
- **R2 (T-main):** sub-threshold L2 re-looks, rewritten small (the g41 code is gone). Target only object-views with a same-/wrong-class box in 0.05..birth threshold (46+59 of 232). Gate: p50 <= +20 ms, Helsinki realtime no-harm (>= 3 interleaved reps, log arm mapping), then split A/B. If no-harm loses > 1 se, switch the line to flight-start coverage (frames 1-23).
- **R3 (I5, NEW, never tried):** open-vocabulary recall proposer for L1 ground objects. Run a small pretrained open-vocab detector (YOLO-World-S/M or OWLv2-base, whichever is in the pip cache or on HF, inference only) on L1 views only with prompts "tank / military vehicle / truck / trailer / launcher", keep only boxes where v2 has nothing, and class them via the I4 DINOv2 prototypes. Gates: offline recall on the 105 no-box L1 object-views from recorded valcity frames <= 150 (target >= 15 recovered at <= 2x FP), p50 <= +25 ms, then split A/B.
- **R4 (I1), only if the pod has headroom:** L1 upscale (extra v2 pass at imgsz 1920 on L1 views, routed to tank/ta-ta/small_launcher/mine_roller) and condor->r11 route, as ONE split A/B arm each. Gates as before (p50 <= +20 ms).
- Handoff triage: drone-fast-r5-2 REJECT; drone-fast-r1-1 REJECT; drone-bigbet-3 REJECT as a swap (weights unavailable); drone-fast-r2-1-fr2c decided by R1.

| # | island | thesis | champion | score | status |
|---|---|---|---|---|---|
| I1 | Synthetic cut-out YOLO + class routing | A fast single-stage detector trained on cut-outs, each class routed to the model best at it | main 435b191 routed v2+r11 (OVERALL BEST). Candidates: condor->r11 (+0.0126, 1.6 se), L1 upscale (unscored), routed small_launcher AdaBN (AP 0.239->0.462) | 0.1459+-0.0105 validation | alive, holds production (R4) |
| I4 | Foundation-feature verification (DINOv2 prototypes) | Frozen broadly-pretrained features separate our 16 assets from urban clutter; re-score, never relabel | main drone/proto_verify.py (A 0.5 in prod). Candidate: drone-fast-r2-1-fr2c A 0.7 | 0.141 vs 0.104 validation; A 0.7 valcity +0.037 (z 7.7), holdout tie, no platform pairs | alive, PRIORITY (R1) |
| T-main | Planner: where the camera looks | Answer layer is exhausted; the lever is L2 re-looks on views with a sub-threshold box and flight-start coverage | SAFER in main; re-look code lost with gpu3 | SAFER 0.1317 vs 0.1008; addressable pool 105/232 L1 object-views | alive (R2) |
| I5 | Open-vocabulary recall proposer (NEW) | 45% of L1 object-views get no box of any class from a YOLO trained on synthetic cut-outs; a detector pretrained on real imagery proposes the missed vehicles, I4 prototypes assign the class | none yet | unscored | new (R3) |
| I3 | Target-domain adaptation | Synthetic-to-real gap closable on the flown city | drone-bigbet-3 (+0.024, 1.7 se over prod; weights unavailable on POD=gpu) | global AdaBN -0.035/-0.119 | RETIRED this restart: no training allowed, weights gone; its routed small_launcher arm moves to I1 |
| T-alt | Exemplar/template NCC | Same 16 assets in every city | drone-evolve-g3-3 | valcity -0.017 | RETIRED (gen 4) |

## History
| gen | best score (island) | note |
|---|---|---|
| 1 | 0.094 validation (I1 v2) | islands designed |
| 1 (end) | 0.141 validation (I4 verifier + SAFER) | I4 g1-1 merged |
| 2 (end) | 0.1459 validation (I1 routed v2+r11) | g2-2 merged 435b191 |
| 3 (end) | 0.1459 (unchanged) | all 3 discarded |
| 4 (killed) | 0.1459 (unchanged) | the evolve runs never finished; partial AdaBN screen negative |
| R1 (relaunch gen 1) | 0.1459 (unchanged) | executed I3 and I1; no committed result; audit: nothing > 2 se |
| R2 (relaunch gen 2, start) | 0.1459 validation (main 435b191) | STALLED (4 flat). Global AdaBN dead (-0.035/-0.119). Priority: I4 A 0.7 platform A/B, T-main re-looks with proper no-harm, I1 L1 upscale re-run |
| restart g1 (start) | 0.1459 validation (main 435b191) | STALLED (5 flat). POD=gpu inference-only. I3 retired (no training, weights gone), I5 open-vocab recall proposer added. Priority: I4 A 0.7 platform pairs, T-main re-looks, I5 recall screen |
