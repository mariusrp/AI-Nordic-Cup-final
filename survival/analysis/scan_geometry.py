#!/usr/bin/env python3
"""Understanding lane: detection geometry of a rotating vision cone (pure geometry from the sim constants, no sim).
A predator charges straight at the agent at 15 px/step from 250 px, from a uniformly random bearing. The agent
perceives it when d <= hearing (50) or (d <= 200 and |bearing - heading| <= 30 deg), sampled once per step.
The agent's heading turns omega rad/step (random phase); turning costs omega/(2 pi) energy per step.
Prints P(first perceived at >= 90 px, i.e. before the predator's unconditional-charge radius) and at >= 120 px,
and the energy cost per second. omega = 0 is a fixed heading (e.g. looking where it walks).
Usage: python3 scan_geometry.py
"""
import math, random

R = random.Random(1)
for hear in (50, 70, 90):
    print(f"hearing {hear}")
    for om in (0.0, 0.1, 0.2, 0.35, 0.5, 0.7, 0.9, 1.05):
        n90 = n120 = 0
        N = 20000
        for _ in range(N):
            b = R.uniform(-math.pi, math.pi)      # predator bearing (world)
            h = R.uniform(-math.pi, math.pi)      # agent heading at step 0
            d = 250.0 - R.uniform(0, 15)          # random phase of the approach
            first = None
            while d > 15:
                rel = (b - h + math.pi) % (2 * math.pi) - math.pi
                if d <= hear or (d <= 200 and abs(rel) <= math.pi / 6):
                    first = d; break
                d -= 15.0; h += om
            first = first or 0
            n90 += first >= 90 + 15   # +15: the observation is one predator move stale
            n120 += first >= 120 + 15
        print(f"  omega {om:4.2f} rad/step ({math.degrees(om):3.0f} deg)  P(seen >= 90 px) {n90/N:.2f}  P(>= 120 px) {n120/N:.2f}  "
              f"turn cost {10*om/(2*math.pi):.2f} energy/s")
