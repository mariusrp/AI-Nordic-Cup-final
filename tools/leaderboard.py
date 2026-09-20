#!/usr/bin/env python3
"""Print top validation scores per case from the public leaderboard."""
import json, urllib.request
d = json.load(urllib.request.urlopen("https://cases.nordicaicup.com/api/v1/leaderboard/validation", timeout=30))
print("teams:", len(d))
for case in ["survival-simulator", "drone-flyby", "medical-appointment"]:
    rows = sorted([(t["scores"][case]["real"], t["team_name"]) for t in d if t["scores"].get(case)], reverse=True)
    print(f"\n{case}: {len(rows)} entries")
    for i, (s, n) in enumerate(rows[:8]):
        print(f"  {i+1:2d}. {s:10.4f}  {n}")
    me = [(i+1, s) for i, (s, n) in enumerate(rows) if n == "Phillips"]
    print("  Phillips:", me)
