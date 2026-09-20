#!/bin/bash
# Waits for the split-training probe (train_split.sh -> /workspace/ft6s/ft6s/weights/last.pt, TRAIN_DONE in its log),
# then measures it exactly like ft_all / ft5r: detector-only recall on the recorded L1 and L2 views of frames 181-249
# (det_views.py) and the whole-stack hold181 replay + score next to fa and f5 (run_hold.sh ARMS="fa f5 f6").
#   ALLOW_GPU_MAIN=1 POD=gpu python3 infra/pod.py bg nacb_eval6 "bash /workspace/nacb/drone/bench/eval_split.sh"
set -u
W=/workspace/nacb
PY=/workspace/venv-drone/bin/python
export UPSTREAM=/workspace/upstream YOLO_CONFIG_DIR=/workspace/ft/.ultra OMP_NUM_THREADS=2 PYTHONPATH=/workspace/i4/pylib
export DRONE_RECORD_DIR= DRONE_REPLAY_TABLE=
F6=/workspace/ft6s/ft6s/weights/last.pt
FA=/workspace/ft/runs/ft_all/weights/last.pt
F5=/workspace/ft5/runs/ft5r/weights/last.pt
for i in $(seq 200); do grep -q TRAIN_DONE /workspace/logs/nacb_train.log && break; sleep 30; done
grep -q TRAIN_DONE /workspace/logs/nacb_train.log || { echo "training not done"; exit 1; }
echo "=== $(date +%T) train done; synthetic-val mAP50 per epoch:"; cut -d, -f1,8,9 /workspace/ft6s/ft6s/results.csv | tail -4
cd $W/drone
echo "=== $(date +%T) det_views L1 (recorded L1 views, 181-249)"
$PY bench/det_views.py $W/gt_hold.json $W/det_f6_L1.json --imgsz 1280 --thr 0.25 --levels 1 f6=$F6 fa=$FA
echo "=== $(date +%T) det_views L1 @0.05"
$PY bench/det_views.py $W/gt_hold.json $W/det_f6_L1_05.json --imgsz 1280 --thr 0.05 --levels 1 f6=$F6
echo "=== $(date +%T) det_views L2 survey sweeps"
$PY bench/det_views.py $W/gt_hold.json $W/det_f6_L2.json --imgsz 1280 --thr 0.25 --levels 2 --root /workspace/drone_seen_full f6=$F6
echo "=== $(date +%T) hold181 whole stack"
ARMS="fa f6 f5 fa_t15" NBOOT=100 bash bench/run_hold.sh
echo EVAL6_DONE
