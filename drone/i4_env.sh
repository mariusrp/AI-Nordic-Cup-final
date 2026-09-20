# I4 (foundation features) pod env on POD=gpu3: shared venv-drone + timm/DINOv2 in /workspace/i4/pylib (pip --target, no deps)
export PYTHONPATH=/workspace/i4/pylib HF_HOME=/workspace/i4/hf UPSTREAM=/workspace/upstream YOLO_CONFIG_DIR=/workspace/i4/.ultra
PY=/workspace/venv-drone/bin/python
# setup (once): mkdir -p /workspace/i4 && /workspace/venv-drone/bin/pip install --no-deps --target /workspace/i4/pylib timm huggingface_hub safetensors
