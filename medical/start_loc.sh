#!/usr/bin/env bash
# Start the CPU locator sidecar (loc_server.py, :9061) used by pipeline.py as the per-question
# LLM-failure fallback. CPU only (CUDA hidden), fp32, 4 threads. Log: /workspace/logs/med_loc.log
# The /predict server works without it (heuristic fallback) and notices it within 10 s once up.
set -uo pipefail
D="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$D"
CUDA_VISIBLE_DEVICES= HF_HOME=/workspace/hf HF_HUB_OFFLINE=1 MED_LOC_PORT=${MED_LOC_PORT:-9061} MED_LOC_THREADS=${MED_LOC_THREADS:-4} \
  nohup /workspace/venv-vllm/bin/python "$D/loc_server.py" > /workspace/logs/med_loc.log 2>&1 &
for i in $(seq 1 60); do
  if curl -s "http://127.0.0.1:${MED_LOC_PORT:-9061}/health" | grep -q '"ok"'; then echo "locator up after ${i}x5s"; exit 0; fi
  sleep 5
done
echo "locator not up"; tail -20 /workspace/logs/med_loc.log; exit 1
