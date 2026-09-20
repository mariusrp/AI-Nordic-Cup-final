# SURV4 - head-to-head policy pick for the FINAL (mean of 3 games) + reliability tuning

Objective changed: the final runs THREE games and AVERAGES them. Optimise mean / p25 / min, not the tail.

## Setup (all on the Mac; serve pod untouched, no portal validation)
    cd /Users/adrian/blinq/projects/AI-Nordic-Cup
    UPSTREAM=/Users/adrian/blinq/projects/nac-n3-tmp/upstream SDL_VIDEODRIVER=dummy \
    /Users/adrian/blinq/projects/nac-n3-tmp/venv/bin/python survival/evaluate.py <policy> \
      --seeds 9200..9327 --procs 3 --max-time 3000 --json <out>
Deviation from the brief: --procs 3 (not 4) so that the FOUR arms of a round run CONCURRENTLY under identical
machine load; a fair head-to-head needs the arms to face the same contention, and another agent was already
running ~35 evaluate.py workers on this Mac (load avg 40 on 12 cores).
128 fresh seeds 9200-9327 per arm, both rounds on the SAME seeds, so every arm is paired against the round-1
lin_map draw.
Analysis: scratchpad/h2h/an.py (mean, sd, p25/p50/p75, min, max, >=1500, >=2000, <1000, paired diff vs lin_map,
deaths, mean end time, and a 200k bootstrap of the 3-GAME AVERAGE that the final actually scores).

## Round 1 - head-to-head: lin_map (LIVE) vs lineage6 vs lin_tune vs lin_tail
(running)

## Round 2 - reliability arms on the LIVE policy (lin_map), same 128 seeds
- r2_disp   MAP_PARAMS {"disp_map":1}   - MAP-BASED LATE DISPERSION. lin_map already implements it and it has never
                                          been measured: from t>=800, an agent with >=2 siblings inside 80 px, energy
                                          >120 and no fruit in view walks to a REMEMBERED tree >=150 px away that it
                                          first saw <=40 s ago and saw alive <=30 s ago, and holds the claim 60 s.
                                          Rationale: the collapse is CORRELATED - 3-5 agents camp the same patch, the
                                          patch's trees die together and they all starve inside 50-100 s. Spreading
                                          them over independent trees decorrelates the death times, and the score is
                                          the time of the LAST death (max over agents), not the mean. The earlier
                                          dispersion attempt (lin_disperse, -17 se 56) failed for a stated reason that
                                          lin_map removes: the egocentric herd had no tree map, so an evicted agent
                                          hopped into empty ground.
- r2_looseA MAP_PARAMS {"route_t":400,"route_val":0,"route_gain":0,"fr_seen":80,"route_cool":0}
                                        - replication of the 48-seed arm A (loose late routing): the best MEAN of the
                                          map family (+44.1 se 55.1), rejected under the OLD objective because it lost
                                          the upper tail. The tail no longer matters.
- r2_rtree  MAP_PARAMS {"route_tree":1} - hungry agent with no remembered FRUIT walks to a remembered probably-alive
                                          TREE (the "camped tree dies" failure).
- r2_emerg  LIN_PARAMS {"herd_emerg":4,"herd_min":4} - the brief's reliability knob: rebuild the herd from 4 agents
                                          instead of 2, and hold a late floor of 4 instead of 3.
Dropped before launch: traits.py (different base, claimed gain was vs bb_juke which is ~200 below lineage6) and a
5th concurrent arm (CPU contention would have slowed every arm).

## MACHINE CONTENTION (affects wall clock, NOT the numbers)
10:12-11:10 the Mac carried 150-220 python processes / load 110-190 on 12 cores (other survival agents running
~34-worker sweeps each). My 16 round-1 processes got roughly 1-3 cores of the 12. Scores are not wall-clock
dependent, so the comparison is unaffected; only throughput is. Two consequences:
- round 2 was cut from 4 arms x 128 seeds to 3 arms x 48 seeds and started CONCURRENTLY with round 1: against
  ~200 competing processes, 12 more of my own costs round 1 about 6% of its share while buying round 2 a share
  of its own. Chaining it would have meant it never ran.
- 48-seed arms are a SCREEN (paired se ~35-50 by the LEDGER's own null measurements), not a confirmation.

## ARITHMETIC OF THE FINAL (do this before believing any arm)
score ~ seconds survived; the final is the MEAN OF THREE games. With mean 1280 and sd 310, a 3-game average is
1280 +- 179 (sd/sqrt(3)), so P(3-game average >= 2000) is about 0.003%. Getting to a 2000 AVERAGE needs the
per-game mean to move from ~1280 to ~2000: +720 s, i.e. +56%. No knob in this family has ever moved the mean by
more than ~+230 (herd_div 6), and every single-constant sweep since has landed inside the null band. So the
honest framing for the final is: pick the policy with the best MEAN, and expect a 3-game average near that mean
+- 180. "Consistent 2000+" is not reachable from 1280 by tuning; it needs a mechanism change worth +700 s.

## WHAT ACTUALLY HAPPENED TO THE MEASUREMENT (read this before trusting any n below)
The Mac could not deliver the runs. Timeline, with measured CPU accounting:
- 10:10 idle-machine calibration: 4 seeds of lin_map, --procs 4, wall 88 s, 302 CPU-s => ~75 CPU-s per game.
  512 games (4 policies x 128 seeds) therefore needs ~38,400 CPU-s = ~53 min of the whole 12-core Mac.
- 10:12 round 1 launched (4 policies x 128 seeds, --procs 3 each, concurrent).
- 10:28-11:30 other agents' sweeps grew from ~40 to ~220 python processes, load 110-190 on 12 cores. My 12
  round-1 workers accumulated 10,906 CPU-s in 78 min = 2.3 cores, i.e. 28% of the work done, ETA ~3 more hours.
- 11:32 round 1 stopped. Round 2 (3 arms x 48 seeds, started 10:50) was at 1,205 CPU-s per arm = 33%, ETA ~13:00.
- 11:42 round 2 stopped as well, and the design was changed to one that yields data under any deadline:
  scratchpad/h2h/harvest.py runs the FROZEN evaluate.py `one()` game loop with ONE FRESH PROCESS PER SEED
  (maxtasksperchild=1, imap chunksize 1) and appends every finished game to a JSONL immediately, so stopping
  early still leaves a usable paired sample. (A side benefit: a fresh process per seed removes the Pool
  worker-history contamination the LEDGER blames for its 16-seed noise.)
- 11:52 measured per-game wall under load: 740-960 s (vs 88 s idle) = ~10x slowdown. Four harvesters x 4 workers
  produce ~1 game/min for all four policies together.
NOTHING here is a 64+ seed claim. evaluate.py's Pool.map is all-or-nothing, which is why two runs that were
28% and 33% complete produced zero rows; that is the lesson to carry forward, not a result.

## RESULT 1 - HEAD-TO-HEAD on 48 fresh seeds 9200-9247 (paired, one fresh process per seed)
```
common seeds n=48: 9200..9247
policy         n    mean     se     sd     p25     p50     min     max  >=1500  >=2000    diff   se_d     z     win   pred/starve/old   a3sd   a3p25
lin_map       48  1255.5   53.8    372  1091.3  1258.0   337.6  2047.7   27.1%    2.1%     0.0    0.0  0.00    0/48  1727/793/3533       213    1112
lineage6      48  1261.7   55.7    386  1065.6  1346.6   337.6  2031.5   29.2%    4.2%     6.2   52.8  0.12   20/48  1722/750/3602       221    1113
lin_tune      48  1312.3   53.3    369  1087.4  1305.3   559.8  2152.3   27.1%    6.2%    56.8   58.3  0.97   25/48  1622/871/3658       211    1167
lin_tail      48  1213.2   51.0    353   993.1  1282.3   367.7  1757.6   25.0%    0.0%   -42.3   51.0 -0.83   18/48  1368/959/3261       202    1079
```

## RESULT 2 - RELIABILITY ARMS on the LIVE policy, 48 seeds 9200-9247 (paired vs the same lin_map draw)
```
common seeds n=48: 9200..9247
policy         n    mean     se     sd     p25     p50     min     max  >=1500  >=2000    diff   se_d     z     win   pred/starve/old   a3sd   a3p25
lin_map       48  1255.5   53.8    372  1091.3  1258.0   337.6  2047.7   27.1%    2.1%     0.0    0.0  0.00    0/48  1727/793/3533       213    1112
arm_maptune   48  1354.2   52.7    365  1109.9  1348.3   559.8  2374.8   25.0%    8.3%    98.7   57.5  1.72   21/48  1732/965/3707       209    1210
arm_disp      48  1294.4   56.8    394   974.2  1329.9   337.6  1969.8   35.4%    0.0%    38.8   32.7  1.19   23/48  1800/836/3575       225    1143
arm_looseA    48  1268.5   57.4    398  1014.1  1275.1   337.6  1961.3   33.3%    0.0%    12.9   47.3  0.27   23/48  1713/1146/3454       228    1114
arm_emerg     48  1244.6   56.4    391   925.3  1306.7   337.6  2005.4   25.0%    2.1%   -11.0   38.5 -0.29   23/48  1740/1001/3562       224    1094
lin_tune      48  1312.3   53.3    369  1087.4  1305.3   559.8  2152.3   27.1%    6.2%    56.8   58.3  0.97   25/48  1622/871/3658       211    1167
lineage6      48  1261.7   55.7    386  1065.6  1346.6   337.6  2031.5   29.2%    4.2%     6.2   52.8  0.12   20/48  1722/750/3602       221    1113
lin_tail      48  1213.2   51.0    353   993.1  1282.3   367.7  1757.6   25.0%    0.0%   -42.3   51.0 -0.83   18/48  1368/959/3261       202    1079
```
arm_maptune = lin_map + LIN_PARAMS {"young_gap":30,"spawn_bar_late":200} (lin_tune's two constants on the LIVE base)

## RESULT 3 - THE KILL SHOT: lin_tune's gain does not survive 33 more seeds
```
common seeds n=83: 9200..9282
policy         n    mean     se     sd     p25     p50     min     max  >=1500  >=2000    diff   se_d     z     win   pred/starve/old   a3sd   a3p25
lin_map       83  1270.2   40.2    366  1073.1  1260.1   163.3  2047.7   28.9%    1.2%     0.0    0.0  0.00    0/83  2831/1414/6198       210    1131
lin_tune      83  1256.1   38.9    354  1036.3  1237.0   163.3  2152.3   20.5%    3.6%   -14.0   43.2 -0.32   40/83  2823/1449/6158       203    1121
```
On the FIRST 48 seeds lin_tune is +56.8 se 58.3; over 81 seeds it is -12.2 se 44.2, so on the 33 seeds
9248-9280 alone it is -112. That is the same collapse the LEDGER already recorded for these two constants
(selection block +87, second block +58, fresh block +15.7 se 58.4). arm_maptune's +98.7 se 57.5 sits on
exactly the 48-seed block that flattered lin_tune, so it must be discounted the same way.

## WHERE THE WEAK GAMES COME FROM (48-seed block, all 7 arms on the same seeds)
- The catastrophic games are a property of the SEED, not of the policy. On seeds 9223 / 9210 / 9218 five of the
  seven arms scored BIT-IDENTICALLY (338 / 560 / 564): lin_map, lineage6, disp, looseA and emerg produced the
  same game, because every mechanism they change (map routing t>=800, dispersion t>=800, late spawn bar t>=600,
  emergency rebuild at n<=4) is a LATE-GAME rule and the game was already over. Games that end before ~600 s are
  decided by the first minutes and no late knob can touch them.
- Games below 600 s: lin_map 3/48, lineage6 4/48, lin_tune 2/48, lin_tail 3/48. Games below 1000 s: 10-14 of 48.
- Per-seed sd of the mean-over-policies is 303; the mean within-seed sd across the 7 policies is 239. Roughly half
  of all variance is the seed and half is the chaotic re-draw a code change causes. This is why nothing in this
  family has ever confirmed: the signal we are hunting (+50) is a fifth of the noise a code change injects.
- Only the two arms that change EARLY behaviour (young_gap 30 = a birth is due 30 s after the last one instead of
  40) ever rescued a wipe-out seed: lin_tune scored 1339 on seed 9223 and 1295 on 9218 where every late-game arm
  scored 338 / 564. That is the right mechanism for the weak quartile - but it did not replicate on 9248-9280.

## VERDICT PER POLICY (48 fresh seeds, paired; ledger evidence in brackets)
| policy   | mean   | diff vs lin_map | p25    | min   | call |
|----------|--------|-----------------|--------|-------|------|
| lin_map  | 1255.5 |   0.0           | 1091.3 | 337.6 | LIVE - keep |
| lineage6 | 1261.7 |  +6.2 se 52.8   | 1065.6 | 337.6 | identical [ledger 192 seeds +7.4 se 33.3] - no reason to switch either way |
| lin_tune | 1312.3 | +56.8 se 58.3   | 1087.4 | 559.8 | NOT REAL: -12.2 se 44.2 over 81 seeds |
| lin_tail | 1213.2 | -42.3 se 51.0   |  993.1 | 367.7 | WORSE on mean AND p25 - do not serve |
| disp     | 1294.4 | +38.8 se 32.7   |  974.2 | 337.6 | mean up, p25 DOWN 117 - weak evidence by the brief's own rule |
| looseA   | 1268.5 | +12.9 se 47.3   | 1014.1 | 337.6 | noise; starvation deaths 1146 vs 793 |
| emerg    | 1244.6 | -11.0 se 38.5   |  925.3 | 337.6 | flat mean, p25 DOWN 166 - demography constants stay a dead end |
| maptune  | 1354.2 | +98.7 se 57.5   | 1109.9 | 559.8 | same block that flattered lin_tune; discount accordingly |

## RECOMMENDATION FOR THE FINAL
SERVE lin_map, UNCHANGED (survival/policies/lin_map.py, already live on serve :9052). Reasons:
1. Nothing beats it by 2 se. The two arms that looked best (lin_tune +56.8, maptune +98.7) both live on the same
   48-seed block, and extending that block by 33 seeds turned lin_tune into -12.2 se 44.2.
2. lin_tail is measurably the WORST of the four on both mean (-42.3) and p25 (-98): under the old tail objective
   it was the "lottery" candidate; under the 3-game-average objective it is the one policy to rule out.
3. lineage6 is statistically identical, so a rollback buys nothing and costs a redeploy before the final.
4. Every reliability knob the brief named was tried and none protects the weak quartile: herd_emerg 4 + herd_min 4
   LOWERS p25 by 166, late dispersion lowers it by 117, loose routing by 77.
IF the orchestrator wants a positive-EV bet anyway, the cheapest is LIN_PARAMS='{"young_gap":30,"spawn_bar_late":200}'
on the LIVE lin_map: env-var only, no code change, instant rollback by unsetting it. Point estimate +98.7 se 57.5 on
48 seeds, but the honest pooled estimate for these two constants across every block ever run is ~+20 with se ~40.
I do NOT recommend it without a 128-seed confirm on seeds nobody has looked at.

## THE ANSWER TO "WE NEED CONSISTENT 2k+"
lin_map over 81 fresh seeds: mean 1267.8 se 41.2, sd 370, p25 1073, min 163, 29.6% of games >= 1500, 1.2% >= 2000.
The final averages THREE games, so the quantity that is scored has mean 1268 and sd 370/sqrt(3) = 213:
400k bootstrap of the 3-game average over lin_map's own 85 harvested games (mean 1272, sd 363):
  mean 1272, sd 208, p05 918, p25 1136, p50 1278, p75 1415
  P(>= 1300) 45.8%   P(>= 1500) 13.7%   P(>= 1750) 0.71%   P(>= 2000) 0.00% (not one draw in 400,000).
A 2000 AVERAGE needs the per-game mean at ~2000, i.e. +730 s (+58%) on today's policy. For scale, the largest
confirmed gain this codebase has ever produced is herd_div 6 at +228, and every one of the ~40 arms since has
landed inside the null band. Tuning will not get there; only a different endgame economy will, and the measurement
budget to prove one is ~128 seeds per arm (~11 CPU-hours) on a Mac that today was shared with 200 other processes.
