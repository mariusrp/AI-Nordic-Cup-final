# Drone tracks folded into drone-evolve (13:15). Each is its OWN island; keep its line of approach.
Islands share only their best result and condensed lessons, once per generation (at the meta-review).
Merge authority for drone: drone-evolve (drone-main folded in at 14:25).

| Island (folded track) | Thesis / line of approach | Champion | Score | Status |
|---|---|---|---|---|
| T-alt (drone-alt) | Classical detector (multi-scale, multi-rotation template matching of the 16 synthetic assets) + a frozen leave-block-out detector scorer | drone-alt-r1-1 (scorer WIP), drone-alt-r1-2 (template matcher WIP) | unscored | interrupted mid-round |
| T-main (drone-main) | End-to-end production line: YOLO v2 + tracker + coverage planner; precision-first reporting (track confirmation, L2 verification); domain-robust retrain (--robust negatives, LoveDA backgrounds, degradations) | drone-main-r1-2 (LIVE in production, validation 0.1317 vs 0.1008 same route; AUDITED PASS, keep main's tools/nac.py) | 0.1317 validation | r1-1 domain-robust retrain: 0.1102 vs 0.0941 (weaker than r1-2). Round 2 not started. Production and main differ: merge r1-2 via handoff |
