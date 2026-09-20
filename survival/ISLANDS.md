# Survival islands (meta-reviewer registry)

Lean restart workflow (20:15), generation 2/2, meta-review 2026-09-18 ~21:20 CEST. Scores from the frozen scorer: train32 = seeds 1000-1031; confirm = survival/confirm.py fresh seeds paired; holdout = auditor's hidden 32 seeds. Compute: POD=gpu5 only, --procs 4 (shared with drone), confirm.py --pod gpu5 --procs 4 --n 64 first, 128 only for a real candidate. (A 32-core POD=cpu4 was added for survival at 20:50 per LESSONS; use it only if the orchestrator assigns it.)

**CHAMPION (LIVE since 19:24): bb_juke** (main : survival/policies/bb_juke.py). Confirms vs bb_mpc +94.9 (z 4.0) and +124.9 (z 4.2); paired holdout +57.5 se 44; confirm absolute 1100.8 se 22.7; validation (live endpoint) not re-drawn, best portal 1185.6 (v1).
**stalled = true**: the best has not improved in the last 2 generations (previous workflow evolve g1 19:10 and this workflow gen 1 21:15 both produced no winner; bb_juke has been best since 19:24).

Goal framing: top teams validate at 1650-1794 vs our 1185.6. The median game (~1000-1150 s) must move +200 or more. Stalled => this generation spends on the two islands with EXTERNAL evidence of a structural effect (G clan breeding, S senescence conversion) and on the never-run R, not on P/N tweaks.

Coordination: the survival fast lane (worktrees .claude/worktrees/wf_0ab883c0-e9c-2 = fb1 senescence grafts, -3 = fb2 clan grafts, branches survival-fast-r1-1-fb1 / r1-2-fb2) is grafting both handoffs onto bb_juke right now. G and S generators must first check those worktrees/branches and the ledger for fb1*/fb2* results and BUILD ON the best graft (e.g. clan x sen, or clan + starvation fix) instead of re-running the same graft.

| island | thesis | champion (branch : file) | score | status | priority gen 2 |
|---|---|---|---|---|---|
| G. Heritable traits / clan breeding (folded T-food trait breeding) | Change WHO lives: breed only the top lineage (speed, sprint, 500/max_energy, vision, hearing), so herd speed climbs 10 -> 20 by t~800-1000 and outruns predators (sprint 15). Oracle speed 17 + hearing 90 gave +273 on v1; handoff team2_clan pooled +51.7 se 19.1 vs v1 (z 2.7, 256 seeds; pred -34%, starve +90%). Arm: clan breeding + relative spawn/eat bars grafted onto bb_juke (bb_juke's need-based claiming should absorb the starvation cost); then fix the fast clan's starvation (eat bar scaled by speed, fewer but fatter children). Gate vs bb_juke train32 z>2 -> confirm 64. Did NOT execute gen 1 (branch survival-evolve-g1-2-gheritable empty). | handoff ext-survival-team2-clan-* : survival/policies/team2_clan.py (on v1); graft = fast lane fb2a-d (unscored) | +51.7 se 19.1 vs v1 (v1 ~974); on bb_juke: untested | alive, did not execute -> priority | 1 |
| R. Terrain physics (river/swamp moat, gap pockets) | Energy is charged on the requested distance and movement is multiplied by the biome penalty afterwards (river 0.3, swamp 0.5): anyone moving in water pays 3.3x/px. Team2: 40-70% of kills happen in swamp/river, i.e. water is where AGENTS die. Arm 1: terrain-aware evasion: when an awake predator is < 250 px, never flee into / along water or swamp, steer the juke toward dry ground (and away from water when foraging with a predator in hearing); log act share (>= 2%) and kills-in-water before scoring. Arm 2 (only if 1 acts): lure a charging predator across a water strip the agent is already beyond. Predator-proof 10-20 px obstacle gaps (agent radius 5 fits, predator 10 does not; mean 8 per map) are arm 3. Did NOT execute gen 1. | none (base bb_juke) | - | new, did not execute -> priority | 2 |
| S. Senescence conversion (NEW, from handoff team2_sen; folded E energy allocation) | Aging past the hidden max_age drains 0.01*age per tick (29-35% of spend). Detect onset (our elder_detect.py: precision 1.000, recall 0.998, or team2's 3-tick excess drop) and convert the elder's stock into a child immediately while energy > 101, instead of burning it (elder yielding kept the drain and failed). Arm on bb_juke: sen conversion alone, then sen + clan (G) crossover. | handoff ext-survival-team2-sen-* : survival/policies/team2_sen.py (on v1); graft = fast lane fb1a-d (unscored) | +39.6 se 27 vs v1 (128 seeds); on bb_juke: untested | new | 3 |
| N. Income economy (folded T-food) | Raise income instead of re-dividing it: 30% of fruit rots at unvisited trees, herd energy-negative 300-600 s. Gen 1 Levy walk + barren-tree exit on bb_juke +20.8 se 53.5 (z 0.39, discard). Remaining arm: spawn only when the parent sees >= 2 fruit (births timed to local income), and rotting-share logging to check whether Levy even moved it. | survival-evolve-g1-1-levy : survival/policies/n_levy.py | 1199.5 se 43.7 vs bb_juke 1178.7 (train32, discard) | alive, executed gen 1 | 4 |
| P. Predator play (folded T-behaviour; T joint search as a tool) | Exploit the exact charge rule (facing makes predators pivot at sprint cost). Remaining kills: target switches 53-61%, wakes 45-51%. Gen 1 timed wake ring: acts 0.05% of steps, -58.1 se 59.3 (dead). Remaining arm: no spawn while an awake predator is in contact < 250 px (a kill refills the predator). T joint CMA-ES over ~8 juke constants only after a new base (G or S) exists. | main : survival/policies/bb_juke.py | confirm 1100.8 (champion) | alive, champion, executed gen 1 | 5 |

Parked/retired: T-params joint search (tool for P/G once a new base exists; survival-opt-r1-1/r1-2 never scored), T-food re-division (relay -56, patience -34, F7 knobs; its surviving ideas live in N and G), obstacle refuges as places (base rate, BB5 -84..-96), D roles (-4.4), rollout-MPC dodge (-35..-140), BB3 shared eyes (oracle -78), pp_wake ring (-58).

## Rules
- Base = bb_juke; baseline in every run = bb_juke (main). Report mean, se, p25, min, pred/starve/old deaths.
- Cascade: smoke (max-time 600, 8 seeds) -> train32 paired vs bb_juke on gpu5 --procs 4 (z > 2) -> confirm.py --pod gpu5 --procs 4 --n 64 -> 128 -> holdout.
- Show the rule ACTS before scoring (< 1% of steps is not worth a run). Never buy safety with foraging time; feed the agents you save.
- Commit arm + numbers + ledger row to the island branch even for a discard. Branch names: use a unique suffix (survival-evolve-g2-<n>-<tag>); g1-N names are taken. Short agents (<= ~40 tool calls).
- Handoffs (python3 tools/handoff.py list survival): team2_clan (+addendum) -> island G; team2_sen -> island S. Both UNAUDITED; they count only after a graft on bb_juke passes the cascade.

## History (lean restart 20:15)
- gen2 meta-review (21:20): gen 1 executed N (n_levy +20.8 se 53.5) and P (pp_wake -58.1 se 59.3), both discarded; R and G did not execute. Handoffs team2_clan (+51.7 se 19.1 vs v1, 256 seeds) and team2_sen (+39.6 se 27 vs v1) adopted as champions of G and S; fast lane wf_0ab883c0-e9c is grafting them onto bb_juke (fb1*/fb2*, unscored). Islands 4 -> 5 (S senescence conversion NEW). Best unchanged bb_juke; stalled = true. Priority: G, R, S, N, P.
- evolve g1 scribe: N n_levy 1199.5 se 43.7 (+20.8, discard); P pp_wake 1120.6 se 53.6 (discard, rule barely acts). Audit: no winner; nothing merged; champion bb_juke unchanged.
- gen1 meta-review (20:10): no scored experiment since 19:24 (N and G branches empty). Champion bb_juke. Islands: R river moat (NEW, priority 1; sim source shows biome penalty applies after the energy charge), N income (2), G traits (3), P predator play incl. T search (4). T demoted to a tool. stalled = false.

## History (this workflow, restart 19:50)
- gen1 meta-review (19:55): champion bb_juke live (confirm +94.9/+124.9, holdout +57.5). Islands consolidated 6 -> 4: N income economy (new, priority 1), G traits (2), P predator play (champion, 3), T joint param search (4). stalled = false.

## History (previous evolve workflow, 19:15-19:20)
- gen2 meta-review (19:20): gen 1 executed N, E, P but left no branch, ledger row or pod run (nothing measured); best unchanged = bb_juke (deploy pending confirm #2: bb_mpc arm 972.8 se 19.8 on 2128-2255, juke arm running). F7 food knobs (fast lane r4-1) all below bb_mpc. Folded tracks restored as own islands (T-food, T-params; T-behaviour = P). Six islands. stalled = false. Priority: G, T-food, T-params (did not execute gen 1), then N, P, E (re-run, gen 1 produced nothing).
- evolve g1 scribe: audit found no candidate beating the best by >2 se; nothing merged, champions and scores unchanged.
- gen1 meta-review (19:15): best = bb_juke (confirm +94.9 vs bb_mpc, deploy pending the paired holdout and confirm #2). Islands: P (champion bb_juke), G (traits on FDE, untested), N (nomadic foraging, NEW), E (energy allocation). stalled = false. Priority: N, P, G, E.

## History (previous workflow, gens 1-5)
- gen1: A 922.7, B 973.2, C 1026.5 (holdout 973.8). evolve g1-g3: all discarded. BB2 bb_mpc CONFIRMED +64.5 vs v1 (merged, live).
- evolve g4: bb_juke train 1160.6 (+123), holdout -7 (audit-fail at the time); E, C, T-food discarded.
- evolve g5: C 1055.9, T-behaviour 1089.8, T-food 982.6, A afb 1194.5 (no gain over bb_juke). None merged.
- 19:05: bb_juke confirm #1 +94.9 se 23.6 z 4.0 vs bb_mpc (128 fresh seeds), reversing the holdout verdict.
