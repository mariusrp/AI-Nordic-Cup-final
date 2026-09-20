# Drone big-bets lane (bigbets, 2026-09-18)

Compute: gpu4. Base at start: production validation 0.132 (I4 DINOv2 verifier + SAFER merged later as 25b0fe1).

## Approaches researched and ranked
1. **B1 valcity: rebuild the validation city as a second labelled scene** (ranked first). The validation sequence renders the same way on every run: 000001_L0 has the same md5 in all 10 recorded runs, and the README allows recording it. Tiled recorder runs give 4K frames, the ~16 objects are labelled once in global coordinates, and one homography per flight (LS fit error 0.2/0.6 px) propagates the labels to every frame. This fills the gap the team has: an honest scorer (det_eval and local Helsinki are saturated), plus target-domain training data and second exemplars.
   Critique: the scorer is built from a single city, so overfitting it is a risk. Labels are hand-made, so errors in them bias the ranking. Use it to rank candidates. It does not replace the platform validation number.
2. **B2 two-stage detector: a class-agnostic render proposer plus a DINOv2 crop classifier with mined new-city negatives.** Critique: I4 (a DINOv2 prototype verifier) landed on main during the lane and covers most of B2(b). The open question became whether B2 adds anything on top of I4.

## What was built and how it scored
| bet | branch / commit | what | score | base | outcome |
|---|---|---|---|---|---|
| B1 v1 | drone-bigbet-1 @ c43b15f (fast-forwarded from the worktree branch worktree-wf_1585b5cb-2e3-8) | valcity v1 labels, render.py (reconstructed 4K frames), a paired block-bootstrap replay gate (drone/valcity/replay_boot.py), answer-layer variants | 0.2719+-0.062 | 0.223 | variants not better than the best; the scorer infrastructure is kept (the fast lane already uses it) |
| B2 it3 | drone-bigbet-2 @ 3905f26 | training-free crispness cue and B2 clean-fold DINO, re-gated on top of the I4 champion (drone/bb2/gate_it3.py, held-run tiles) | 0.807+-0.053 | 0.831 | discard: all results within 1 se, B2 closed; the DRONE_VERIFY_CRISP knob stays off by default |

No winner was handed off, so there is no handoff and no audit.

## Round 2 (B3 / B5)
Chosen bets. **B3** fine-tunes v2 (YOLO11s) on recorded 960x540 valcity views from frames 1-150. It uses v1_core labels propagated by homography, ignore masks over unsure and unlabelled answer clusters, clutter kept as hard negatives, and 2400 Helsinki synthetic views replayed. The SAFER hedge and the I4 verifier are unchanged. **B5** inpaints labelled objects out of the 4K valcity reconstructions and pastes Helsinki cut-outs of all 16 classes onto those backgrounds, 2-3 instances per scene. The aim is to cover the classes that have no target labels.

| bet | branch / commit | what | score | base | outcome |
|---|---|---|---|---|---|
| B3 | drone-bigbet-3 | valcity target-domain fine-tune | see ledger | 0.132 platform | BUILT and handed off (the handoff is unchanged) |
| B3 routed | drone-bigbet-1 @ 1653a2c | conservative routing: v2 base + r11 helicopter/jet_plane + B3 ftAll_last for small_plane/tank/mine_roller/ta-ta. Split A/B against production 435b191 | 0.0896 | 0.0951 | INCONCLUSIVE three times. The only pair was measured under GPU contention and is not valid. Nothing handed off |
| B5 it2 | drone-bigbet-2-b5 @ a0605a4 | 100% target-city B5 replay + seed replicate, class-held-out transfer proxy | 0.2527+-0.03 | 0.174 | no handoff: the proxy gain was not confirmed on platform. B5 closed |

## Should either bet become an evolve island?
- **B1 valcity: yes, as infrastructure, not as a model island.** Every drone lane should gate candidates with valcity replay_boot (scene valcity_v1_core). Next steps: more labels, and use the reconstructed 4K frames as target-domain training data. The training-data use has not been tried yet.
- **B2: no.** The DINO verifier cue is already captured by I4. The crispness cue and clean-fold DINO added nothing measurable. Dead end unless the proposer (B2a) is tried on its own.
- **B3: yes, but only as the handed-off fine-tune.** The routed variant needs a clean platform A/B on an idle GPU before anyone retries it. The evolve result g3-2 shows the fine-tune gain overlaps with r11 routing.
- **B5: no.** The proxy gain did not carry over. Do not retry it unless the paste realism (lighting and shadows) changes.

## A it4: c1b3 flight-start L1 raster (drone-bb1-dense-it4 @ 2a8f1c0)
- The claimed valid realtime A/B was 0.5337+-0.0055 against a base of 0.4957.
- The independent confirm FAILED. The rerun did not reproduce a clear realtime win. Offline scores replicate exactly, and there were no invalid responses.
- gpu5 has a cgroup CPU quota of 7.65 cores and is throttled by other lanes, so realtime timing on gpu5 is not reliable. Outcome: discard for now. There is no handoff. Re-test the realtime A/B on an unthrottled pod before resubmitting. Island candidate: yes, as a raster/zoom-plan island once realtime is confirmed.
