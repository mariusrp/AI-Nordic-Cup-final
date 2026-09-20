#!/usr/bin/env bash
# Start the /predict server on :9054 (after start_llm.sh). Logs: /workspace/logs/med_server.log
# Env knobs: MED_ASR_MODEL (large-v3-turbo), MED_YES_BIAS, MED_DEADLINE, MED_BACKEND (llm|heuristic)
set -uo pipefail
source "$(dirname "$0")/env.sh"
pkill -f "python.*medical/[s]erver.py" 2>/dev/null; sleep 2
cd "$MED_DIR"
nohup $PY "$MED_DIR/server.py" > /workspace/logs/med_server.log 2>&1 &
for i in $(seq 1 120); do
  if curl -s http://127.0.0.1:9054/ | grep -q running; then echo "server up after ${i}x3s"; curl -s http://127.0.0.1:9054/api; echo; exit 0; fi
  if ! pgrep -f "python.*medical/[s]erver.py" >/dev/null; then echo "server died:"; tail -40 /workspace/logs/med_server.log; exit 1; fi
  sleep 3
done
echo "server not up"; tail -40 /workspace/logs/med_server.log; exit 1
