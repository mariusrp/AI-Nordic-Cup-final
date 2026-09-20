# Source me on the pod: common environment for the medical case.
export HF_HOME=/workspace/hf
export UPSTREAM=${UPSTREAM:-/workspace/upstream}
MED_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export MED_DIR
PY=/workspace/venv-med/bin/python
[ -x "$PY" ] || PY=python3
export PY
# cuBLAS/cuDNN shipped as pip wheels -> visible to ctranslate2 (faster-whisper)
NVLIBS=$($PY - <<'P' 2>/dev/null
import os, importlib
out = []
for m in ("nvidia.cublas.lib", "nvidia.cudnn.lib"):
    try:
        mod = importlib.import_module(m)
        out.append(os.path.dirname(mod.__file__) if getattr(mod, "__file__", None) else list(mod.__path__)[0])
    except Exception:
        pass
print(":".join(out))
P
)
export LD_LIBRARY_PATH="${NVLIBS}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export MED_LLM_URL=${MED_LLM_URL:-http://127.0.0.1:8001/v1}
# PRODUCTION (holdout-audited): onset start rule (MED_SPAN_ONSET_SENTSTART=1) + quote gate 'any' are the
# pipeline.py CODE DEFAULTS since medical-evolve-g1-1-onset; env.sh deliberately exports neither.
# Do NOT set MED_SPAN_SHIFT_S here: the code uses shift 0 whenever onsets exist and keeps the legacy
# +0.2 s only when onsets are missing (decode failure / sparse onsets).
# MED_ORDER=aeq was rolled back 22:10: local +0.011 and holdout +0.010, but organiser validation 0.7338 vs 0.744 (portal is near-deterministic).
