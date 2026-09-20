#!/usr/bin/env bash
# Start the local vLLM OpenAI server used by the medical pipeline.
#   bash start_llm.sh <hf_model> [gpu_mem_fraction=0.55] [port=8001]
# Stops any previous medical vLLM first. Logs: /workspace/logs/vllm_med.log
set -uo pipefail
source "$(dirname "$0")/env.sh"
MODEL=${1:-cyankiwi/Qwen3.8-27B-AWQ-INT4}
MEM=${2:-0.55}
PORT=${3:-8001}
# ninja (flashinfer JIT) lives in the venv; vllm is started by absolute path so put the venv on PATH.
export PATH=/workspace/venv-vllm/bin:$PATH
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}   # pod has a 5120 pid/thread cap shared by all jobs
export VLLM_USE_FLASHINFER_SAMPLER=${VLLM_USE_FLASHINFER_SAMPLER:-0}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}   # weights are pre-downloaded; skip slow hub lookups
# A local directory can be served under a stable name: MED_SERVED_NAME=Qwen/... bash start_llm.sh /root/dir
NAME_ARGS=""; [ -n "${MED_SERVED_NAME:-}" ] && NAME_ARGS="--served-model-name $MED_SERVED_NAME"
pkill -f "[v]llm serve.*--port $PORT" 2>/dev/null; sleep 3
nohup /workspace/venv-vllm/bin/vllm serve "$MODEL" --port "$PORT" --host 127.0.0.1 \
  --max-model-len ${MED_MAX_LEN:-8192} --gpu-memory-utilization "$MEM" --max-num-seqs 16 \
  --enable-prefix-caching --limit-mm-per-prompt '{"image":0,"video":0}' \
  $NAME_ARGS ${MED_VLLM_EXTRA:-} > /workspace/logs/vllm_med.log 2>&1 &
for i in $(seq 1 180); do
  if curl -s "http://127.0.0.1:$PORT/v1/models" | grep -q '"id"'; then echo "vLLM ready after ${i}x5s: $MODEL"; exit 0; fi
  if ! pgrep -f "[v]llm serve.*--port $PORT" >/dev/null; then echo "vLLM died:"; tail -40 /workspace/logs/vllm_med.log; exit 1; fi
  sleep 5
done
echo "vLLM not ready after 15 min"; tail -40 /workspace/logs/vllm_med.log; exit 1
