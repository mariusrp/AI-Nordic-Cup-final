#!/bin/bash
# Run drone portal arms sequentially on POD=gpu's public port 22 (194.68.245.26:22174): for each "name|VAR=val ..."
# restart the server from /workspace/ft/code/drone via bb3/serve_ft5.sh with those variables (W, W2, WB, WB2, SPLIT,
# VERIFY, plus any DRONE_* env), validate on the portal, append to RESULTS. The LAST spec should be the production
# configuration so the pod ends up serving production again. Usage: bash tools/drone_arms.sh RESULTS.tsv "arm|W=.." ...
set -u
cd "$(dirname "$0")/.."
OUT=$1; shift
URL=http://194.68.245.26:22174/predict
BASE="CODE=/workspace/ft/code/drone UPSTREAM=/workspace/upstream PORTS=9053,22 P=22 BANK=/workspace/drone_evolve_prod_g22/drone/bankAll.pt R11=/workspace/drone_weights/r11.pt PYPATH=/workspace/i4/pylib"
for spec in "$@"; do
  name=${spec%%|*}; vars=${spec#*|}
  echo "=== $(date +%H:%M) drone arm $name: $vars"
  up=$(NAC_MERGER=drone POD=gpu python3 infra/pod.py exec "$BASE ARM=$name $vars bash /workspace/ft/code/drone/bb3/serve_ft5.sh" 300 | tail -1)
  echo "$up"
  # warm-up + latency check on the pod itself (the first real request must not be the cold one)
  NAC_MERGER=drone POD=gpu python3 infra/pod.py exec "for i in \$(seq 40); do curl -s localhost:22/ | grep -q running && break; sleep 3; done; tail -2 /workspace/logs/ft_arm_$name.log | cut -c1-160" 150 | tail -2
  score=$(python3 tools/portal.py validate drone $URL 2>&1 | grep -E "^SCORE" | tail -1)
  echo "$score"
  printf '%s\t%s\t%s\t%s\t%s\n' "$(date +%Y-%m-%dT%H:%M)" "$name" "$vars" "$up" "$score" >> "$OUT"
done
echo "ALL_DRONE_ARMS_DONE"
