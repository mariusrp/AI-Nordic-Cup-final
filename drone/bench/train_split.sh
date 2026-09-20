#!/bin/bash
# Generalisation PROBE of the ft5r recipe (not a serve candidate): the same data and recipe as ft5r, but only the real
# views of frames <= 150 (+ all Helsinki tiles + the 9000 synthetic views), so frames 181-249 stay a block of NEW
# instances the model never saw. Scored afterwards with bench/det_views.py and bench/run_hold.sh against the table GT
# next to ft_all (same protocol, 0.195 L1 recall) and ft5r (trained on those frames, 0.917 = memorisation ceiling).
#   ALLOW_GPU_MAIN=1 POD=gpu python3 infra/pod.py bg nacb_train "bash /workspace/nacb/drone/bench/train_split.sh"
set -u
W=/workspace/ft6s
mkdir -p $W
cd $W
ls /workspace/ft5/real_all/images/train/*.png | awk -F/ '{n=$NF; if (n ~ /^hel/) print; else if (n ~ /^vc/) {f=substr(n,3,3)+0; if (f<=150) print}}' > $W/train_real_le150.txt
echo "real views <= 150 + helsinki: $(wc -l < $W/train_real_le150.txt)"
cat > $W/data_ft6s.yaml <<EOF
path: /
train:
  - /root/dd/data/yolo_v2/images/train
  - $W/train_real_le150.txt
val: /root/dd/data/yolo_v2/images/val
names:
$(sed -n '/^names:/,$p' /workspace/ft5/runs/data_ft5r.yaml | tail -n +2)
EOF
cat $W/data_ft6s.yaml | head -8
export YOLO_CONFIG_DIR=/workspace/ft/.ultra OMP_NUM_THREADS=4
/workspace/venv-drone/bin/python - <<'EOF'
from ultralytics import YOLO
m = YOLO('/workspace/drone_weights/v2.pt')
m.train(data='/workspace/ft6s/data_ft6s.yaml', epochs=int(__import__('os').environ.get('EP', '12')), imgsz=1280, batch=8,
        lr0=0.001, lrf=0.2, warmup_epochs=1, optimizer='AdamW', momentum=0.9, freeze=10, cos_lr=True,
        mosaic=0.5, close_mosaic=3, mixup=0.1, degrees=0, fliplr=0.5, flipud=0.5, scale=0.3, hsv_h=0.01, hsv_s=0.4, hsv_v=0.3,
        project='/workspace/ft6s', name='ft6s', exist_ok=True, workers=5, device=0, plots=False, patience=100)
print('TRAIN_DONE /workspace/ft6s/ft6s/weights/last.pt')
EOF
