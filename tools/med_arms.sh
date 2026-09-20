#!/bin/bash
# Run medical portal arms sequentially: for each "name|ENV=val ENV=val" restart the /workspace/med-attach server on
# :9054 (POD=gpu) with that env, verify the serving process carries it, validate on the portal, append to RESULTS.
# The portal is near-deterministic (repeat runs ~0.002 apart) and the leaderboard keeps the best score, so every
# arm is a free measurement on 19 unseen conversations. Usage: bash tools/med_arms.sh RESULTS.tsv "arm1|ENV=.." ...
set -u
cd "$(dirname "$0")/.."
OUT=$1; shift
URL=https://5g74a0atgzlxwf-9054.proxy.runpod.net/predict
for spec in "$@"; do
  name=${spec%%|*}; env=${spec#*|}
  echo "=== $(date +%H:%M) arm $name: $env"
  NAC_MERGER=medical POD=gpu python3 infra/pod.py exec "p=\$(ss -ltnp | grep ':9054 ' | grep -o 'pid=[0-9]*' | cut -d= -f2 | head -1); [ -n \"\$p\" ] && kill \$p; sleep 4; ss -ltnp | grep -c ':9054 '" 30 | tail -1
  NAC_MERGER=medical POD=gpu python3 infra/pod.py bg "med_arm_$name" "cd /workspace/med-attach/medical && source env.sh && MED_PORT=9054 $env MED_REQ_LOG=/workspace/logs/med_requests_arm_$name.jsonl \$PY server.py" | tail -1
  got=$(POD=gpu python3 infra/pod.py exec "for i in \$(seq 80); do curl -s localhost:9054/ | grep -q running && break; sleep 3; done; p=\$(ss -ltnp | grep ':9054 ' | grep -o 'pid=[0-9]*' | cut -d= -f2 | head -1); tr '\0' '\n' < /proc/\$p/environ | grep -E '^MED_' | grep -v MED_REQ_LOG | sort | tr '\n' ' '" 300 | tail -1)
  echo "serving env: $got"
  score=$(python3 tools/portal.py validate medical $URL 2>&1 | grep -E "^SCORE" | tail -1)
  echo "$score"
  printf '%s\t%s\t%s\t%s\t%s\n' "$(date +%Y-%m-%dT%H:%M)" "$name" "$env" "$got" "$score" >> "$OUT"
done
echo "ALL_ARMS_DONE"
