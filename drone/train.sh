#!/bin/bash
# Build synthetic data and train the YOLO detector.
#   bash train.sh                       # defaults below
#   MODEL=yolo11m.pt EPOCHS=80 NAME=v2 CANVASES=3000 EXTRA_BG=/workspace/drone_bg bash train.sh
# Output: runs/detect/$NAME/weights/best.pt  (copied to weights/$NAME.pt and weights/best.pt)
set -euo pipefail
PY=${PY:-/workspace/venv-drone/bin/python}
cd "$(dirname "$0")"
MODEL=${MODEL:-yolo11s.pt}
EPOCHS=${EPOCHS:-60}
IMGSZ=${IMGSZ:-1280}
BATCH=${BATCH:-12}
NAME=${NAME:-v1}
CANVASES=${CANVASES:-2000}
WORKERS=${WORKERS:-$(( $(nproc) - 1 ))}
DATA=${DATA:-data/yolo_$NAME}
DEVICE=${DEVICE:-0}

[ -f data/synth_src/cutouts.json ] || $PY make_synth.py prep --src data/synth_src
if [ ! -f "$DATA/data.yaml" ]; then
  $PY make_synth.py gen --src data/synth_src --out "$DATA" --canvases "$CANVASES" --val-canvases 120 \
      --workers "$WORKERS" --seed "${SEED:-0}" ${EXTRA_BG:+--extra-bg $EXTRA_BG} ${GENERIC_BG:+--generic-bg $GENERIC_BG}
fi
YAML="$DATA/data.yaml"
if [ -n "${PSEUDO:-}" ]; then   # extra real (pseudo-labelled) data: PSEUDO="/abs/dir1 /abs/dir2" each with images/ labels/
  YAML="$DATA/data_$NAME.yaml"
  { echo "path: $(cd "$DATA" && pwd)"; echo "train:"; echo "  - images/train"
    for p in $PSEUDO; do for r in $(seq 1 ${PSEUDO_REPEAT:-2}); do echo "  - $p/images"; done; done
    echo "val: images/val"; sed -n '/^names:/,$p' "$DATA/data.yaml"; } > "$YAML"
fi
export YOLO_CONFIG_DIR=${YOLO_CONFIG_DIR:-/workspace/.ultralytics}
PY=${PY:-/workspace/venv-drone/bin/python}
$(dirname $PY)/yolo detect train data="$YAML" model="$MODEL" epochs="$EPOCHS" imgsz="$IMGSZ" batch="$BATCH" \
  device="$DEVICE" workers="$WORKERS" name="$NAME" exist_ok=True project="$(pwd)/runs/detect" \
  mosaic=1.0 close_mosaic=10 scale=0.3 degrees=0 translate=0.1 fliplr=0.5 flipud=0.5 \
  hsv_h=0.015 hsv_s=0.5 hsv_v=0.35 mixup=0.0 cos_lr=True patience=100 plots=False ${EXTRA_ARGS:-}
mkdir -p weights
cp "runs/detect/$NAME/weights/best.pt" "weights/$NAME.pt"
cp "runs/detect/$NAME/weights/best.pt" weights/best.pt
echo "trained -> weights/$NAME.pt"
