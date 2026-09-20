#!/bin/bash
# One-time (idempotent) setup of the drone workstream on the RunPod pod.
# Code lives in /workspace/drone (bash push.sh).  Run:  bash /workspace/drone/pod_setup.sh
# Everything goes into /workspace/venv-drone (--system-site-packages, reuses the image torch).
set -e
V=/workspace/venv-drone
[ -x $V/bin/python ] || python3 -m venv --system-site-packages $V
$V/bin/pip install -q "ultralytics>=8.3" faster-coco-eval fastapi uvicorn requests scipy pydantic 2>&1 | tail -2
# ultralytics pulls opencv-python (needs libGL, absent in the runpod image) -> headless; torch 2.4 needs numpy<2
$V/bin/pip uninstall -y -q opencv-python >/dev/null 2>&1 || true
$V/bin/pip install -q "numpy<2" "opencv-python-headless<4.11" 2>&1 | tail -1
export YOLO_CONFIG_DIR=/workspace/.ultralytics
[ -d /workspace/upstream/.git ] || git clone -q https://github.com/amboltio/Nordic-AI-Cup-2026 /workspace/upstream
mkdir -p /workspace/logs /workspace/drone_seen
$V/bin/python -c "import cv2, ultralytics, torch; print('ok', cv2.__version__, ultralytics.__version__, torch.__version__, torch.cuda.is_available())"
$V/bin/pip install -q huggingface_hub 2>&1 | tail -1
