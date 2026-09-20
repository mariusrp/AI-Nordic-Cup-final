#!/usr/bin/env python3
"""Winner handoff queue. Only the case's merge authority merges/redeploys (see /home/claude/handoff/authority.json).
  handoff.py submit <case> <branch> "<score/se/baseline, audit verdict, what it is>"   # non-authority workflows
  handoff.py list <case>                                                              # authority: pending winners
  handoff.py done <case> <branch> merged|rejected "<reason>"                         # authority: after deciding
"""
import json, sys, time, os
Q = "/home/claude/handoff"
def load(case):
    p = f"{Q}/{case}.jsonl"
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []
cmd, case = sys.argv[1], sys.argv[2]
if cmd == "submit":
    with open(f"{Q}/{case}.jsonl", "a") as f:
        f.write(json.dumps({"t": time.strftime("%H:%M"), "branch": sys.argv[3], "note": sys.argv[4] if len(sys.argv) > 4 else "", "status": "pending"}) + "\n")
    print(f"handed off {sys.argv[3]} to the {case} merge authority ({json.load(open(Q + '/authority.json'))[case]}). Do NOT merge or redeploy yourself.")
elif cmd == "list":
    rows = load(case); done = {(r["branch"]) for r in rows if r["status"] != "pending"}
    for r in rows:
        if r["status"] == "pending" and r["branch"] not in done: print(json.dumps(r))
elif cmd == "done":
    with open(f"{Q}/{case}.jsonl", "a") as f:
        f.write(json.dumps({"t": time.strftime("%H:%M"), "branch": sys.argv[3], "status": sys.argv[4], "note": sys.argv[5] if len(sys.argv) > 5 else ""}) + "\n")
    print("recorded")
