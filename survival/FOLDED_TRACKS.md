# Survival tracks folded into survival-evolve (13:15). Each is its OWN island; keep its line of approach.
Islands share only their best result and condensed lessons, once per generation (at the meta-review). Generators work from their island's champion + shared lessons only.

| Island (folded track) | Thesis / line of approach | Champion (branch: file) | Train score (16 seeds) | Status |
|---|---|---|---|---|
| T-behaviour (survival-main) | Predator handling and herd behaviour: awake/charging-aware fleeing, vigilance, population control | survival-r1-3: survival/policies/r1_3_pop_control.py | 939.5 +- 50.3 vs 895.8 | not significant; r1-1 predator-stamina 701.8 (worse: stand-off buffer lost); r1-2 fruit-ripen 946.8 +- 70.5 (noisy) |
| T-params (survival-opt) | Automated parameter search (Optuna/CMA) over policy constants, time-phased schedules | survival-opt-r1-1 / r1-2 (WIP, unscored) | - | interrupted; lesson from r1-4: single-seed-set search gains overfit (+154 vanished on paired check) -> always confirm on fresh seeds |
| T-food (survival-food) | Food economics + breeding: need-based fruit claiming with newborn priority, ripeness-aware harvesting, trait-selective breeding | survival-food-r2-2: survival/policies/food_r2_2.py | 973.2 +- 34.1 vs 900.2 | UNAUDITED candidate (handed off); food-r1-1 1026.5 failed holdout audit (+37.6, z~0.8); absolute dead-reckoned map (r1-2) failed badly (457) |
