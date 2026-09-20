#!/bin/bash
# n3 medical portal arms on POD=gpu :9052 from /workspace/n3-med (does not touch the :9054 production server).
# Usage: bash tools/med_arms_n3.sh RESULTS.tsv "arm1|ENV=val ENV=val" ...   Base env = AEQ production.
set -u
cd "$(dirname "$0")/.."
OUT=$1; shift
URL=https://5g74a0atgzlxwf-9052.proxy.runpod.net/predict
BASE="MED_LLM_URL=http://127.0.0.1:8001/v1 MED_ORDER=aeq MED_SPAN_COVERAGE_NEXT=1 MED_USE_QUOTE=0"
wait_free() {  # block while any medical validation of ours is unfinished (portal runs one at a time; a restart mid-run mixes configs)
  for i in $(seq 120); do
    busy=$(python3 tools/nac.py status medical 2>/dev/null | python3 -c "
import sys,ast; s=sys.stdin.read(); i=s.find('{'); d=ast.literal_eval(s[i:s.rfind('}')+1]); v=d['validations'][0]; print(0 if v.get('finished_at') else 1)" 2>/dev/null || echo 0)
    [ "$busy" = "0" ] && return 0
    sleep 10
  done
}
for spec in "$@"; do
  name=${spec%%|*}; env=${spec#*|}
  echo "=== $(date +%H:%M) arm $name: $env"
  wait_free
  POD=gpu python3 infra/pod.py exec "p=\$(ss -ltnp | grep ':9052 ' | grep -o 'pid=[0-9]*' | cut -d= -f2 | head -1); [ -n \"\$p\" ] && kill \$p; sleep 4; ss -ltnp | grep -c ':9052 '" 30 | tail -1
  POD=gpu python3 infra/pod.py bg "n3_med_$name" "cd /workspace/n3-med/medical && source env.sh && MED_PORT=9052 $BASE $env MED_REQ_LOG=/workspace/logs/med_requests_n3_$name.jsonl \$PY server.py" | tail -1
  got=$(POD=gpu python3 infra/pod.py exec "for i in \$(seq 80); do curl -s localhost:9052/ | grep -q running && break; sleep 3; done; p=\$(ss -ltnp | grep ':9052 ' | grep -o 'pid=[0-9]*' | cut -d= -f2 | head -1); tr '\0' '\n' < /proc/\$p/environ | grep -E '^MED_' | grep -v MED_REQ_LOG | sort | tr '\n' ' '" 300 | tail -1)
  echo "serving env: $got"
  score=$(python3 tools/portal.py validate medical $URL 2>&1 | grep -E "^SCORE|not queued" | tail -1)
  echo "$score"
  printf '%s\t%s\t%s\t%s\t%s\n' "$(date +%Y-%m-%dT%H:%M)" "$name" "$env" "$got" "$score" >> "$OUT"
done
echo ALL_ARMS_DONE
