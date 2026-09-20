#!/usr/bin/env bash
# One-time pod setup for the medical case (runpod/pytorch 2.4, CUDA 12.4, py3.11).
# Idempotent; safe to re-run. Everything heavy lives on the /workspace volume.
#   bash /workspace/<repo>/medical/pod_setup.sh 2>&1 | tee /workspace/logs/med_setup.log
set -euo pipefail
mkdir -p /workspace/logs /workspace/hf /workspace/tx /workspace/medical_seen
export HF_HOME=/workspace/hf
nvidia-smi --query-gpu=name,driver_version,memory.used,memory.total --format=csv || true

# 1) upstream competition repo (dtos.py, utils.py, local_evaluator.py, data/)
if [ ! -d /workspace/upstream/medical-appointment ]; then
  git clone --depth 1 https://github.com/amboltio/Nordic-AI-Cup-2026 /workspace/upstream
fi

# 2) ASR + server venv (system site packages -> reuses the image's torch/cuDNN 9 libs)
if [ ! -x /workspace/venv-med/bin/python ]; then
  python3 -m venv --system-site-packages /workspace/venv-med
fi
/workspace/venv-med/bin/pip install -q -U pip
/workspace/venv-med/bin/pip install -q "faster-whisper>=1.1" "fastapi>=0.115" "uvicorn>=0.30" \
    "pydantic>=2.7,<3" "requests>=2.31" nvidia-cublas-cu12 "nvidia-cudnn-cu12>=9,<10"

# 3) vLLM in its own venv (it pins its own torch; must support the qwen3_5 architecture)
if [ ! -x /workspace/venv-vllm/bin/python ]; then
  python3 -m venv /workspace/venv-vllm
  /workspace/venv-vllm/bin/pip install -q -U pip uv
fi
/workspace/venv-vllm/bin/uv pip install --python /workspace/venv-vllm/bin/python -U vllm \
    2>&1 | tail -3
/workspace/venv-vllm/bin/python -c "import vllm, torch; print('vllm', vllm.__version__, 'torch', torch.__version__, 'cuda ok', torch.cuda.is_available())"

# 4) pre-download models (so the first server start is not a download)
/workspace/venv-med/bin/python - <<'EOF'
import os
from faster_whisper import download_model
for m in ("large-v3-turbo", "large-v3"):
    print(m, download_model(m))
EOF
for M in ${MED_LLM_CANDIDATES:-cyankiwi/Qwen3.8-27B-AWQ-INT4 Qwen/Qwen3.5-35B-A3B-GPTQ-Int4}; do
  /workspace/venv-vllm/bin/huggingface-cli download "$M" --exclude "*.pth" >/dev/null 2>&1 \
    || /workspace/venv-vllm/bin/hf download "$M" >/dev/null 2>&1 || echo "download failed: $M"
  echo "downloaded $M"
done
echo "medical setup done"
