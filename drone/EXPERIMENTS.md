# Drone flyby: experiments log

Local numbers come from upstream `local_evaluator.py` on helsinki (25 frames, 16 objects). The scene is tiny and
the detector's training data comes from these same frames, so the local number is optimistic and noisy (about ±0.04
between near-identical runs). The platform validation run (249 frames, a different scene) is the real benchmark.

## Facts
- Motion is one constant source-frame homography per flight. A least-squares H fitted on GT box centres predicts
  1 step with 0.2/0.6 px mean error (x/y), 10 steps with 1.0/4.6 px. The online ORB+RANSAC estimate (pooled
  `q = H^k p` fit) matches GT within 6-20 px at 10 steps after about 3 frames.
- Box propagation with the online H: fraction of boxes with IoU < 0.5 after k frames is 0.00 / 0.03 / 0.10 / 0.16
  for k = 3 / 6 / 10 / 15 (GT-fit H: 0 / 0 / 0 / 0.07). Re-observing old tracks matters for small objects.
- Latency is decisive. The simulated detector with +250 ms injected latency in `--realtime` loses 10/25 frames and
  mAP drops from 0.85 to 0.44. The round trip including the network must stay well under 333 ms.

## Serving / latency (18 Sep, platform validation runs with DRONE_DETECTOR=none)
- The pod has a ~7.6-CPU cgroup quota (cpu.cfs_quota_us=765000) shared by every case, although nproc says 96.
  Under load even a PNG decode takes ~100 ms (throttling stalls). Keep CPU-heavy jobs off the pod during validation.
- Through the RunPod HTTPS proxy (port 9053) the platform answered only 102/249 frames (≈400 ms network per cycle),
  plus 4 refused camera commands: when slow, the next frame is rendered before our previous command lands.
- Serving on pod port 22, which RunPod maps to a public TCP port (env RUNPOD_PUBLIC_IP:RUNPOD_TCP_PORT_22,
  here http://194.68.245.26:22174/predict), answered 229/249 frames with 0 refused commands. USE THE DIRECT URL.
- Fixes: camera commands are now chosen legal from both the view position and our last requested position;
  registration runs in a background thread after 3 fits; cv2/torch limited to 2 threads.

## Pipeline results (simulated GT detector `DRONE_DETECTOR=oracle`: p(detect) is a sigmoid of short side in view px, p50 = 7 px, plus noise, class confusion and FPs)
| change | offline mAP | realtime mAP |
|---|---|---|
| tracker + greedy coverage planner (L1 top-band sweep emerges) | 0.833 | |
| + per-track residual velocity + track-refresh bonus | 0.817 | |
| + object-density prior in the planner value | 0.847 | 0.847 (0 skipped, 91 ms) |
| + vectorised planner, registration every 3rd frame after convergence | | 0.800 (0 skipped, 72 ms mean) |

## Detector (YOLO on synthetic views)
(filled in below as runs complete)

## SAFER precision-first reporting (branch drone-main-r1-2, 18 Sep)
Track report score = noisy-OR(level-weighted conf) x (hits+0.5)/(hits+0.5+1.5*miss); tracks not yet confirmed
(>=2 detections, or one L2 detection with conf >= 0.5) are reported at x0.15 so they rank below all confirmed ones
(COCO AP: low-ranked extras can only add recall). Planner: unconfirmed candidates add VERIFY_W*life*score to L2 (and L1
if only seen at L0). Output capped at 100 boxes/frame (COCO maxDets).
| run | base 049aad9 | SAFER |
|---|---|---|
| oracle offline (fp 0.3, conf<=0.4) | 0.796 | 0.792 (VW=0), 0.761 (VW=0.1), 0.755/0.693/0.677 (VW=0.5/1/2) |
| oracle offline (fp 3/view, conf<=0.9) | 0.756 | 0.752 (VW=0), 0.758 (VW=0.1) |
| v2 local offline | 0.785 | 0.779 (VW=0.1), 0.785 (VW=0) |
| v2 local realtime x3 (6-10 frames skipped, noisy) | 0.551 0.526 0.498 | VW=0.1: 0.392 0.504 0.564; VW=0: 0.557 0.426 0.423 |
| **portal validation** (both via reverse relay serve pod 194.68.245.133:22165 -> gpu:9191) | **0.1008** | **0.1317** (VW=0.1) |
Relay (drone/relay.py: broker on serve pod ports 22/9055, connector on gpu pod) lets a non-production server be
validated without touching the production port; ~5 ms added.

## I4 DINOv2 prototype verifier (branch drone-evolve-g1-3, I4Foundati-g1-1, 18 Sep, POD=gpu3)
`proto_verify.py`: every L1/L2 v2 box (conf 0.05) is cropped square with 1.5x context (>= 16 view px), resized to
112 px and embedded by frozen DINOv2-S/14 (timm `vit_small_patch14_dinov2.lvd142m`, CLS + mean patch token). Class score =
top-3 mean cosine to that class's prototypes, bg score = top-3 mean cosine to a background bank,
p = softmax([16 class scores, bg] / T). Drop if p_bg > 0.5; otherwise **keep YOLO's class** and set
conf = conf^a * p(YOLO class)^(1-a) (a=0.5, T=0.03). hangar/medium_plane (no cut-outs) only get the bg test. L0 passes through.
Prototypes: the 202 Helsinki cut-outs x 8 rotations x 2 flips x L1/L2, pasted at native scale on inpainted Helsinki
backgrounds or LoveDA (x2), rendered like the evaluator (6464 embeddings). Background bank (2461): v2 FPs on LoveDA
(only ~40: v2 hardly fires on LoveDA), v2 FPs on Helsinki tiles away from GT, random object-free Helsinki patches.
Bank variant `bg_with_seen` adds 2500 v2 boxes from recorded validation runs 1062106c/a2e63304 (rejected, see below).
Serve: `DRONE_VERIFY=/workspace/i4/bankAll.pt` (+ `source i4_env.sh` for timm). Build: `proto_verify.py build`.
| gate | v2 | v2 + verifier (keep, a=.5, T=.03) |
|---|---|---|
| eval_verify comp (LoveDA + held-out block cut-outs, bank from the other block) fold B / fold A | 0.672 / 0.753 | 0.675 / 0.753 |
| eval_verify real Helsinki block (v2 trained on it, so biased to v2) B / A | 0.885 / 1.000 | 0.849 / 0.814 |
| label-free boxes >= 0.5 per recorded validation view (run 5fd5807b, never in a bank) | 2.27 | 0.40 |
| I3 eyeball-labelled validation tiles mini-AP (276 tiles, 15 TP; run 5fd5807b) | 0.522 | **0.715** (15/15 TP kept; a=.3-.7 all 0.71-0.73) |
| local helsinki offline / realtime (gpu3, same run, x2) | 0.785 0.785 / 0.566 0.558 | 0.766 0.766 / 0.536 0.569 |
| verifier cost | | embed p50 9-10 ms, p90 13-24 ms per view; server det p50 26 -> 36 ms, total p50 70 ms |
| **portal validation** (gpu3 direct TCP 194.68.245.208:22080/22081, v2 without SAFER) | **0.108, 0.099** | **0.128, 0.160, 0.118** |
| **portal validation** on top of SAFER (drone-main-r1-2 merged), alternating runs | **0.104, 0.104** | **0.171, 0.129** |
- Prototype argmax as the class (the literal hypothesis) is worse: DINOv2 nearest-prototype class accuracy on TP boxes
  68% (real) / 81% (comp) vs YOLO 100% / 73%; comp AP fold A 0.753 -> 0.683. Fusing YOLO x prototype probs is neutral.
  Keeping YOLO's class and demoting by p(class) is what works: it kills mine_roller/small_plane on roofs and cars.
- Background bank from other validation-city runs (`bg_with_seen`) is harmful, as the critique predicted: the same real
  assets are in those runs, so mini-AP keeps 0-3 of 15 TPs. Never put target-city views in the bank.
- By eye (verify_sheet.py on the recorded validation run): roof/car mine_roller 0.83 -> 0.02-0.12; wrong-class real
  aircraft (jet called condor) 0.90 -> 0.16; hangar and medium_plane untouched (no prototypes).
