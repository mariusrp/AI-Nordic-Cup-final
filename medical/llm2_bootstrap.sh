#!/bin/bash
# Bootstrap a fresh RunPod GPU pod as a second medical LLM server (dense Qwen3.8-27B-AWQ-INT4) reachable over the
# pod's public TCP port 9052. Mirrors medical/start_llm.sh flags. Logs: /workspace/logs/llm2_*.log
set -uo pipefail
MODEL=${MODEL:-cyankiwi/Qwen3.8-27B-AWQ-INT4}
PORT=${PORT:-9052}
export HF_HOME=/workspace/hf
mkdir -p /workspace/logs /workspace/hf
if [ ! -x /workspace/venv-vllm/bin/vllm ]; then
  python3 -m venv /workspace/venv-vllm
  /workspace/venv-vllm/bin/pip install -q --upgrade pip
  /workspace/venv-vllm/bin/pip install -q "vllm==0.29.0" "huggingface_hub" > /workspace/logs/llm2_pip.log 2>&1 || { echo PIP_FAILED; tail -20 /workspace/logs/llm2_pip.log; exit 1; }
fi
echo "vllm $(/workspace/venv-vllm/bin/pip show vllm 2>/dev/null | grep -i ^version)"
/workspace/venv-vllm/bin/python -c "from huggingface_hub import snapshot_download; print(snapshot_download('$MODEL'))" > /workspace/logs/llm2_download.log 2>&1 || { echo DOWNLOAD_FAILED; tail -5 /workspace/logs/llm2_download.log; exit 1; }
echo "downloaded: $(tail -1 /workspace/logs/llm2_download.log)"
export PATH=/workspace/venv-vllm/bin:$PATH OMP_NUM_THREADS=8 VLLM_USE_FLASHINFER_SAMPLER=0 HF_HUB_OFFLINE=1
nohup /workspace/venv-vllm/bin/vllm serve "$MODEL" --port "$PORT" --host 0.0.0.0 \
  --max-model-len ${MAX_LEN:-8192} --gpu-memory-utilization ${MEM:-0.85} --max-num-seqs 16 \
  --enable-prefix-caching --limit-mm-per-prompt '{"image":0,"video":0}' > /workspace/logs/llm2_vllm.log 2>&1 &
for i in $(seq 1 240); do
  if curl -s "http://127.0.0.1:$PORT/v1/models" | grep -q '"id"'; then echo "LLM2_READY after ${i}x5s"; exit 0; fi
  if ! pgrep -f "[v]llm serve" >/dev/null; then echo "LLM2_DIED"; tail -30 /workspace/logs/llm2_vllm.log; exit 1; fi
  sleep 5
done
echo "LLM2_TIMEOUT"; tail -30 /workspace/logs/llm2_vllm.log; exit 1
