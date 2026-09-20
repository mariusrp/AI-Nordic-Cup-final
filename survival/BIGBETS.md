# Survival big-bets lane (2026-09-18)

Two fundamentally different bets, each researched, built, critiqued, and confirmed on 128 fresh seeds (cpu2, seeds 2000-2127) against v1.

## Ranking
1. **BB2 FDE (face-drain-escape predator model)**: CLEAR WIN, merged on main and live as `survival/policies/bb_mpc.py`. It is the shared best for all islands.
2. **BB1 trait ratchet**: NO CLEAR WIN, discarded. It left the largest measured headroom (oracle +273).
3. **BB2 it2/it3 grafts (spawn guard, face-all, fusion, edge memory, rollout MPC)**: all within noise or negative, discarded.
4. **BB3 shared eyes (common herd frame)**: closed at its own go/no-go gate. Even the oracle frame lost to bb_mpc. Not an island.
5. **BB5 refuge nursery**: closed with no new arm. Not an island.

## BB1: directed trait evolution (branch survival-bigbet-1, survival/policies/ratchet.py)
- Idea: run breeding as a selection programme for walking speed > 15 (outruns predator sprint) and hearing >= 90 (covers the charge zone), with a gait that uses the traits.
- Scores: train 1062.8 se 51.2 vs v1 881.1 (paired +181.6 se 72.5). Rerun gave 994.7 vs 888.1. **Confirm (128 seeds): 985.6 vs 962.3, +23.3 se 29.8, z 0.78, NO CLEAR WIN.**
- Oracle (every agent forced to speed 17 and hearing 90): 1157.6 vs 884.6 (+273). Predator deaths fell 60%, but starvation deaths rose 3x. Hearing alone gave +216, speed alone +200.
- Critique: selection works (median effective speed 13.5 vs 11.1 at t=600) and cuts predator deaths by 25%, but the agents it saves starve. The trait gait adds nothing. Breeding is too slow to reach the oracle within a game.
- Island? **Yes, conditionally.** Port it as a detection/breeding track onto bb_mpc, paired with a food fix for the saved agents. BB2 it3 also concluded that further predator gains need detection (hearing), not better evasion.

## BB2: model-based predator play (branch survival-bigbet-2, survival/policies/bb_mpc.py)
- Research (M0 diag): 83% of victims saw their killer >= 10 steps before death, 77% were sprint-locked (energy < max/5), and 52% of final chases started inside 90 px.
- Built (FDE): an agent facing a predicted chaser backpedals at walk speed, so the pivoting predator drains itself (2.55/step) to sleep in about 39 steps. The agent sprints only inside 95 px. Resting predators are detected by odometry and ignored.
- Scores: train 1081.5 vs v1 880.6 (+200.9 se 47.3). **Confirm 1 (128 seeds): 1021.6 vs 957.1, +64.5 se 27.2, z 2.37, CLEAR WIN, handed off and merged.** An it3 re-confirm (tag survivalbetsurvivalbigbet2it3) gave 1016.3 vs 963.0 (+53.3), NO CLEAR WIN on that run. The true fresh-seed gain is about +55-65, roughly 3x smaller than on train.
- Sibling-pose fusion: predator deaths fell 1107 -> 817, but starvation rose and the net result was worse.
- it2 grafts (spawn guard, face-all, fuse=2, edge-memory back-off): all within noise of FDE; edge-memory back-off was -131.
- it3 rollout MPC against the exact charge rule: in a 1v1 harness it saved 39/40 vs 21/40 for straight flight. In full games the kill rate per second did not move, and scores were -35 to -140 vs FDE. Close-range seen charges are not where kills happen; unseen predators, multiple predators, and obstacles are.
- Critique: the exact-model idea pays off only at the drain-to-sleep level. Finer evasion scatters foragers.
- Island? FDE already is the shared base. **Do not reopen MPC or evasion micro-tuning** (dead end, see ledger). The follow-up is island E (break contact with resident predators).

## BB3: shared eyes (worktree wf_bdd12cfd-51c-9, branch survival-bigbet-3-it2, job bigbet3_m0a2, commit 6ab3ac1)
- Premise: BB2 M0 found that in 98% of kills a sibling saw the killer at least 10 steps before the death, but the victim first saw it inside 90 px in 86-91% of kills. The plan: an exact herd frame built from sibling sightings, dead reckoning and obstacle-corner landmarks. Every agent runs the FDE chase test against all predators the herd knows about. A predator seen only remotely gets a turn to face it, not a backpedal.
- Go/no-go: the M0a oracle (perfect shared frame) had to beat bb_mpc by more than 2 se on train16. Results from one paired run: bb_mpc 1089.1 (se 57.6), S1o oracle 1010.7 (-78, se 67), S3f oracle 1010.9 (-78, se 65), real frame R1f 992.4 (-97, se 61).
- Outcome: NO-GO. It3 ran no experiment, on purpose. M0b (pose graph), M1-M3, confirm.py and handoff were not run.
- Critique: the shared information is real, but acting on remote threats costs more than it saves. Facing turns and remote reactions interrupt foraging and change which agent the predator picks. This matches the earlier failed fusion arms (BB2 it2 fuse=2, evolve g4-1). Kills are not mainly a sharing failure for FDE. Evolve island: no.

## BB5: refuge nursery (branch survival-bigbet-5-it3, worktree wf_bdd12cfd-51c-10, commit 6ed61c2)
- Premise: newborns are 33-35% of predator victims. In the arena, predator-proof refuge gaps (10-20 px) gave 20/20 survival over 300 s. The plan: find refuges online, time births to refuge mouths and predator-free windows, and shelter sprint-locked agents at the end game.
- Outcome: closed in it3 with no new arm (base 1087.9). The commit touches only the ledger. Related evidence from evolve: the spawn guard in BB2 it2 and the tail-spawn gating in g4-3 (-35) both lost on bb_mpc, so birth timing is not the bottleneck on FDE. It3 was moved off survival-bigbet-2 so the two bets would not mix.
- Evolve island: no. The refuge finding is the only unused mechanism. If anyone reopens it, do it as a narrow end-game shelter arm for n<=3 only.

## Open questions
- FDE vs food_r1_1 on fresh seeds is untested (-21 se 63 on train).
- Detection (hearing ratchet) on top of FDE, with newborn provisioning so the saved agents do not starve.
