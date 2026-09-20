# SURVIVAL endgame economy — end_econ.py (20 Sep, ~10:05-12:35)

Objective changed: the final evaluation AVERAGES THREE GAMES. Optimise mean / p25 / min, not the best draw.
Base: `survival/policies/lin_map.py` (LIVE) = lineage6 + exact shared dead-reckoned map + late fruit routing.
Candidate: `survival/policies/end_econ.py` (subclass of lin_map, `END_PARAMS='{"end_on":0}'` = plain lin_map).

## Simulator facts I re-derived from the upstream code (three of them contradict the brief)

Read from `nac-n3-tmp/upstream/survival-simulator/src/elements/{environment,fruit,tree,agent}.py`.

1. **A fruit ROTS 50 s after it spawns, not 80.** `fruit.grow(amount=2*dt)` advances `age` by 2/s and the
   environment removes a fruit at `age > 100`. Energy is `min(60, 20 + 2*age_seconds)`: 20 at birth, 60 after
   20 s, then 30 s of standing at 60. lin_map's `fr_ttl=80` therefore keeps entries ~30 s past the fruit's
   death; end_econ uses 48.
2. **Standing still costs 1 energy/s, not 0.1/s** (`agent.energy -= dt * biome_modifier` with dt=0.1 → 0.1 per
   step). Walking costs 0.05 per *requested* px on top (charged before the biome penalty), so walking at 10
   px/step is 6 energy/s: ~6x the cost of standing, and ~17 px per energy.
3. **A tree effectively dies between age 55 and 65, not 50-100.** The death test runs EVERY step with a fresh
   draw: `age > 50 + 50*sqrt(U)`, i.e. a per-step hazard of `((age-50)/50)^2`. Cumulative hazard from age 50 to
   A is `(A-50)^3/750`: 26% of trees reach age 60, 1.1% reach 65, none reach 70. Mean lifetime ≈ 58 s.
   With fruit only from age 20, a tree's whole productive life is ~37 s ≈ 3.7 fruits.
4. Fruit spawns uniform(r, 3r) from the trunk, r = 20 for a mature tree → **20-60 px from the trunk**, and
   everything inside `hearing_radius` is perceived every step regardless of the vision cone.
5. Equilibrium tree count (my derivation, matches lineage6's `trees_est`): spawn 200/N * 0.5^(t/300) attempts/s,
   ~0.6 accepted, lifetime 58 s → **N ≈ 83 * 0.5^(t/600)**: 42 at 600 s, 21 at 1200 s, 10 at 1800 s, 5 at 2400 s.
6. **A tree that is in view is alive by definition.** So "my tree is barren" is almost always variance
   (P(no fruit in 18 s) = e^-1.8 = 17%), and the only reliable signal to relocate is *no tree in view at all*.

**The economic consequence that drives this arm:** one camped tree yields 0.1 fruit/s. Harvested on sight that
is 0.1*20 = **2 energy/s**; harvested ripe it is 0.1*60 = **6 energy/s**. An agent standing costs 1 energy/s and
a birth costs the parent 100. So harvest discipline alone is the difference between one tree supporting ~1 agent
and one tree supporting ~5. That is the biggest single lever I could find in the economy, and it is free: the
standing cost is paid whether you wait or not.

## Mechanisms (all switchable through END_PARAMS)

* **M1 `ripe_on` — ripeness harvest.** The map now marks every fruit entry that was seen BORN: the entry is new
  this step, the observing agent did not move last step, and the spot was already inside its hearing disc (or
  well inside its vision cone, accounting for the scan turn) the step before — so the fruit must have spawned
  now and its age, energy and rot deadline are exact. A fruit whose estimated energy is still under `ripe_e=56`
  is left to grow while the agent can afford the wait (`energy - 1.2*seconds_to_ripe > ripe_floor=70`, and never
  below `ripe_hard=60`); the agent stands and scans instead of collecting 20 energy.
* **M2 `circ_on` — site circuit.** From `circ_t=400` s, an agent with no fruit in view *and no tree in view*
  walks to the best remembered site in the shared map: a remembered fruit cluster valued at what it will be
  worth ON ARRIVAL (and skipped if it will have rotted), or a remembered tree young enough to still be alive
  (`tree_life=45` s since first sighting). One agent per site (`tclaims`). Rich agents crowded at one tree
  (`disp_on`) split off to a free remembered tree so two agents never share one tree's 0.1 fruit/s.
* **M3 `srch_on` — search legs.** No tree in view, nothing in the map, energy above `srch_min_e=45`: standing
  is certain death at income 0, so commit to a straight leg at full walking speed, facing the direction of
  travel (maximal new area per energy), turning when an edge comes within 70 px.
* **M4 `camp_close` — camp on the trunk** (`tree_stay` 35 → 14). Shortens every walk to a fruit and puts most of
  the tree's fruit inside the hearing disc, which is what makes M1's birth detection fire.

## Measurements

Frozen scorer, Mac only, `--procs 6`, `--max-time 3000`, paired against my own lin_map run on the same seeds.
(The Mac was heavily loaded by other tenants: ~124 s of CPU per game, ~3.4 cores available, so a 64-seed pair
is ~75 min of wall clock. That is why there is one screen and no separate confirm block.)

### Smoke (seeds 9000-9003, n=4, not a claim)

| policy | mean | p25 | min |
|---|---|---|---|
| lin_map | 1388.4 | 1347.9 | 1171.2 |
| end_econ | 1412.0 | 1173.4 | 1137.1 |

Mechanism counters from one 1800 s game: 784 fruit seen born, 21946/145828 agent-steps spent holding for
ripeness (15%), 1779 steps with no tree in view, 459 site routes started (99 arrivals, 314 cancelled because
food appeared on the way, 39 timeouts), 210 search steps. No errors.

### Screen block 1 (seeds 9000-9015, n=16, paired, both runs from the same command)

    lin_map    n=16 mean= 1306.3 se= 71.9 p25= 1227.4 p50= 1326.7 min= 625.1 max= 1759.8  >=1500 4 (25%)  >=2000 0  deaths p/s/o=623/310/1180
    end_econ   n=16 mean= 1225.7 se= 55.5 p25= 1137.1 p50= 1167.0 min= 801.3 max= 1595.7  >=1500 2 (12%)  >=2000 0  deaths p/s/o=562/348/1192
    PAIRED end_econ - lin_map: diff=-80.5 se=56.5 z=-1.42  wins=5/16

**The combined arm loses.** Mean -80, p25 -90, median -160, 5/16 wins; the only thing that improves is the
worst game (min 801 vs 625) and predator deaths (-10%), while starvation rises (+12%). Not conclusive at
z=-1.42, but it is the wrong direction and it is not worth spending the rest of the budget confirming a loss.

The Mac was carrying 10+ other survival evaluations from sibling agents all morning (load average 114-190 on 12
cores, ~124 s of CPU per game, my share ~2 cores), so a 16-seed paired block costs ~24 min of wall clock.
That is why the blocks below are 16 seeds and not the 64 the measurement rules ask for: at this contention a
64-seed pair is ~100 min and there was budget for exactly one.

### Ablation (same seeds 9000-9015, paired against the lin_map draw above)

Splitting the arm in two to find which half costs:
* **R = harvest discipline only** (M1 ripeness + M4 camp on the trunk): `END_PARAMS='{"circ_on":0,"srch_on":0,"disp_on":0}'`
* **W = walking only** (M2 circuit + M3 search, no ripeness, tree_stay back to 35): `END_PARAMS='{"ripe_on":0,"camp_close":0}'`

    lin_map    n=16 mean= 1306.3 se= 71.9 p25= 1227.4 p50= 1326.7 min= 625.1 max= 1759.8  >=1500 4 (25%)  deaths p/s/o=623/310/1180
    armR       n=16 mean= 1368.1 se= 93.6 p25= 1064.6 p50= 1520.8 min= 641.4 max= 1936.7  >=1500 8 (50%)  deaths p/s/o=561/297/1267
    PAIRED armR - lin_map: diff=+61.8 se=86.8 z=0.71  wins=8/16
    armW       n=16 mean= 1266.0 se= 83.1 p25= 1009.5 p50= 1202.5 min= 726.5 max= 1788.4  >=1500 5 (31%)  deaths p/s/o=594/406/1141
    PAIRED armW - lin_map: diff=-40.3 se=75.1 z=-0.54  wins=7/16

**The two halves pull in opposite directions and the combination is worse than either.**

* **W (walking) is the cost.** Starvation deaths +31% (406 vs 310), p25 -218, mean -40. This is the fourth
  independent time this lane has found that sending agents to remembered/unseen ground raises starvation more
  than it raises income (LEDGER: `surv3-map` arm A "starve 3024 vs 1531", `surv2-disperse` BARREN EXIT
  "starvation deaths still rise", LATE DISPERSION -17). The circuit only fires when NO tree is in view, and
  even that disciplined version loses. **Treat map-driven relocation as a closed dead end.**
* **R (harvest discipline) is the promising half.** Mean +62, median +194 (1520.8 vs 1326.7), games >= 1500
  doubled (8/16 vs 4/16), best game 1936.7 vs 1759.8, predator deaths -10% and starvation -4%, old-age deaths
  +7% (agents live long enough to age out instead of starving - the right direction). But p25 FALLS
  (1064.6 vs 1227.4): it raises the top half and drags the bottom half, i.e. it adds variance. On 16 seeds
  z=0.71, which is not a result.

### Empirical confirmation of the facts above (scratchpad/end/lifecheck.py, seed 4242, 400 s, no agents)

    trees died n=453  mean_age=57.9  p05=53.4 p50=57.8 p95=62.5 max=78.8
    fruit removed n=1128  mean_age=99.9 (= exactly 50.0 s of life, every single one)
    trees alive at t=400: 51   (my equilibrium formula 83*0.5^(t/600) predicts 52)

So: a tree really does die at ~58 s (95% between 53 and 63), a fruit really does rot exactly 50 s after it
spawns, and the tree population really does halve every 600 s.

## How the orchestrator switches this

    POLICY=survival/policies/end_econ.py                      # candidate, all four mechanisms on
    END_PARAMS='{"end_on":0}'                                 # byte-equivalent behaviour to lin_map
    END_PARAMS='{"circ_on":0,"srch_on":0,"disp_on":0}'        # ripeness harvest only (M1 + M4)
    END_PARAMS='{"ripe_on":0}'                                # circuit + search only
    POLICY=survival/policies/lin_map.py                       # full rollback (current LIVE)

## Why this is not the ripeness idea that already failed in the ledger

`surv2-lin_sprint` arm R ("sr_ripe: agents >= 30% energy leave fruit seen born until 20 s old", -27.5 se 81.2
on 16 seeds) and `tfood_patience` ("fruit memory + wait at tree", -34) both tried to wait for ripeness on
lineage6, WITHOUT the shared map. Without it "seen born" is a guess: an agent cannot tell a fruit that just
spawned from one it has only just turned towards, so it waits on fruit that is about to rot and walks away from
fruit that is already ripe. end_econ gets the birth time from the map: an entry that is new this step, for a
motionless observer, at a spot that was inside the hearing disc (or well inside the cone, allowing for the scan
turn) the step before, can only have spawned this step. The wait is also budgeted (`energy - 1.2 * seconds to
ripe > 70`) instead of unconditional, so a poor agent still eats immediately.

### Confirm of arm R on FRESH seeds 9016-9031 (own lin_map draw on the same seeds)

    lin_map    n=16 mean= 1273.3 se= 91.4 p25= 1141.6 p50= 1312.5 min= 414.2 max= 2069.1  >=1500 4  >=2000 1  deaths p/s/o=610/252/1172
    armR       n=16 mean= 1233.4 se= 80.6 p25= 1067.7 p50= 1159.3 min= 519.8 max= 1771.3  >=1500 4  >=2000 0  deaths p/s/o=603/241/1171
    PAIRED armR - lin_map: diff=-39.9 se=76.9 z=-0.52  wins=9/16

The +62 of the selection block did not replicate. Pooled over both blocks:

    POOLED 32 seeds 9000-9031
    lin_map  n=32 mean= 1289.8 se= 57.3 p25= 1147.3 p50= 1312.5 min= 414.2 max= 2069.1 >=1500  8 (25%) >=1800 1 >=2000 1 deaths p/s/o=1233/562/2352
    armR     n=32 mean= 1300.8 se= 62.0 p25= 1064.6 p50= 1274.5 min= 519.8 max= 1936.7 >=1500 12 (38%) >=1800 1 >=2000 0 deaths p/s/o=1164/538/2438
    PAIRED armR - lin_map: diff=+11.0 se=57.8 z=0.19  wins=17/32

## VERDICT

**Do not deploy. lin_map stays LIVE.** Over 32 paired seeds the best variant of this angle is +11.0 (se 57.8,
z 0.19) on the mean, p25 is 83 LOWER, min is 106 higher and the >=1500 rate is 38% vs 25%. Under an objective
that averages three games, a flat mean with more spread is not an improvement, and the brief's own rule ("a
change that raises the mean but leaves p25 unchanged is weak evidence") rules this out twice over.

What the 96 paired games did establish:

1. **Map-driven relocation is closed.** Even the most disciplined version - move only when there is NO tree in
   view, only to a claimed site, valued at its worth on arrival - costs 40 points and 31% more starvation
   deaths. Four independent attempts in this lane now agree. Do not spend another arm on it.
2. **Harvest discipline is real but not yet worth points.** Waiting for fruit to ripen does what the economics
   say: predator deaths -6%, starvation -4%, old-age deaths +4% (agents age out instead of starving) and the
   >=1500 rate goes from 25% to 38%. It buys the top half of the distribution and sells the bottom half.
3. **The bottom half is where it leaks.** The deferral budget (`energy - 1.2*seconds_to_ripe > ripe_floor=70`)
   lets a nearly-poor agent wait up to 25 s for a fruit a sibling may take first. In a poor game that is the
   difference between living and starving, which is exactly where p25 fell.

### Next step actually worth taking (running as this is written)

`END_PARAMS='{"ripe_floor":140,"ripe_hard":110}'` — defer only while comfortably rich, so the mechanism keeps
the upside it demonstrably has and stops costing the poor games. Result below if it lands inside the time box.

### Refinement test: defer only while rich (`END_PARAMS='{"ripe_floor":140,"ripe_hard":110}'`, 32 seeds 9000-9031)

    lin_map  n=32 mean= 1289.8 se= 57.3 p25= 1147.3 p50= 1312.5 min= 414.2 max= 2069.1 >=1500  8 (25%) >=2000 1
    armR2    n=32 mean= 1241.4 se= 58.1 p25=  988.5 p50= 1224.6 min= 636.9 max= 2032.3 >=1500  7 (22%) >=2000 1
    PAIRED armR2 - lin_map: diff=-48.4 se=59.4 z=-0.82  wins=15/32

It does NOT rescue p25 — it makes it worse. That kills the "poor agents wait too long" explanation: raising the
deferral budget makes the mechanism rarer (behaviour moves back towards lin_map) and the score does not move
back towards 0 either. The three harvest-discipline variants land at +62, -40 and -48 with paired se 58-87,
i.e. all three sit inside one noise band. **Harvest discipline is not distinguishable from lin_map on 32-96
paired games, in any of the three settings tried.**

## Final summary of the 160 paired games run in this box

| arm | seeds | mean | paired diff | p25 | min | >=1500 | >=2000 |
|---|---|---|---|---|---|---|---|
| lin_map (base) | 32 | 1289.8 | — | 1147.3 | 414.2 | 25% | 1/32 |
| end_econ all four | 16 | 1225.7 | -80.5 se 56.5 | 1137.1 | 801.3 | 12% | 0 |
| M2+M3 walking only | 16 | 1266.0 | -40.3 se 75.1 | 1009.5 | 726.5 | 31% | 0 |
| M1+M4 harvest only (shipped default) | 32 | 1300.8 | +11.0 se 57.8 | 1064.6 | 519.8 | 38% | 0 |
| M1+M4, defer only while rich | 32 | 1241.4 | -48.4 se 59.4 | 988.5 | 636.9 | 22% | 1/32 |

**Recommendation: do not deploy. Keep lin_map live.** Nothing here beats it on mean, and nothing here beats it
on p25 at all.

### What the next agent on this lane should and should not do

* **Do not** build another arm on map-driven relocation, exploration or dispersion. Five attempts, five losses,
  and the failure mode is always the same: starvation deaths rise faster than income does.
* **Do** fix `fr_ttl` in lin_map from 80 to ~48 regardless of this arm — a fruit is provably gone 50 s after it
  spawns, so a third of the routing entries lin_map trusts are already rotten. That is a one-line correctness
  fix to the LIVE policy, not a behaviour bet.
* The economics say one camped tree can feed ~5 agents if its fruit is harvested ripe instead of ~1 if it is
  harvested on sight, and the mechanism does move the distribution (>=1500 from 25% to 38%). The reason that
  does not become score is NOT the deferral budget (tested, it makes it worse). The remaining suspects are
  (a) a sibling eating the deferred fruit unripe anyway - measure what fraction of deferred fruit this herd
  actually collects at >= 55 energy, which the diag counters do not yet report, and (b) the birth economy:
  deferral lowers instantaneous energy, and a birth needs the parent above spawn_bar_late 230, so the herd may
  simply be trading fruit energy for fewer births. Measure births per 1000 s before designing anything else.
