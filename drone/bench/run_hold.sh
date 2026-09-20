#!/bin/bash
# hold181 bench, step 2: open-loop causal replay of the 9 held-out recordings (frames 1-249; >= 151 read from
# /workspace/.holdout, evaluation only) through the PRODUCTION server code with several detector arms, then score
# frames 181-249 and 151-249 against the table GT. Runs on POD=gpu (contended: GPU shared with the medical vLLM), one
# arm after another so only one YOLO is resident at a time.
#   bash run_hold.sh            (all arms)     ARMS="fa fav" bash run_hold.sh   (subset)
set -u
W=/workspace/nacb
CODE=$W/drone
OUT=$W/out
mkdir -p $OUT $W/logs
export UPSTREAM=/workspace/upstream YOLO_CONFIG_DIR=/workspace/ft/.ultra OMP_NUM_THREADS=2 PYTHONPATH=/workspace/i4/pylib
export DRONE_RECORD_DIR= DRONE_REPLAY_TABLE=
PY=/workspace/venv-drone/bin/python
BANK=/workspace/drone_evolve_prod_g22/drone/bankAll.pt
V2=/workspace/drone_weights/v2.pt
R11=/workspace/drone_weights/r11.pt
FA=/workspace/ft/runs/ft_all/weights/last.pt
F5=/workspace/ft5/runs/ft5r/weights/last.pt
SEQS=$(ls /workspace/.holdout/drone_seen_heldout/)
cd $CODE
[ -f $W/gt_hold.json ] || $PY bench/mk_gt.py /workspace/replay/tables/table_v3_spc.json $W/gt_hold.json --f0 151 --f1 249

run() {  # tag, then env assignments
  local tag=$1; shift
  if ls $OUT/${tag}_*.jsonl >/dev/null 2>&1; then echo "skip $tag (dumps exist)"; return; fi
  echo "=== $(date +%H:%M:%S) arm $tag: $*"
  env "$@" $PY replay_multi.py $OUT $tag $SEQS > $W/logs/replay_$tag.log 2>&1
  grep -E "REPLAY_DONE|predict_ms" $W/logs/replay_$tag.log | tail -3
}

for arm in ${ARMS:-fa fav v2 v2v f5 f5v fa_t15 fa_t35 v2r v2rv}; do
  case $arm in
    fa)     run fa     DRONE_WEIGHTS=$FA DRONE_WEIGHTS2= ;;
    fav)    run fav    DRONE_WEIGHTS=$FA DRONE_WEIGHTS2= DRONE_VERIFY=$BANK ;;
    v2)     run v2     DRONE_WEIGHTS=$V2 DRONE_WEIGHTS2= ;;
    v2v)    run v2v    DRONE_WEIGHTS=$V2 DRONE_WEIGHTS2= DRONE_VERIFY=$BANK ;;
    v2r)    run v2r    DRONE_WEIGHTS=$V2 DRONE_WEIGHTS2=$R11 ;;
    v2rv)   run v2rv   DRONE_WEIGHTS=$V2 DRONE_WEIGHTS2=$R11 DRONE_VERIFY=$BANK ;;
    f5)     run f5     DRONE_WEIGHTS=$F5 DRONE_WEIGHTS2= ;;
    f5v)    run f5v    DRONE_WEIGHTS=$F5 DRONE_WEIGHTS2= DRONE_VERIFY=$BANK ;;
    fa_t15) run fa_t15 DRONE_WEIGHTS=$FA DRONE_WEIGHTS2= DRONE_NEW_THR=0.15 ;;
    fa_t35) run fa_t35 DRONE_WEIGHTS=$FA DRONE_WEIGHTS2= DRONE_NEW_THR=0.35 ;;
    f6)     run f6     DRONE_WEIGHTS=/workspace/ft6s/ft6s/weights/last.pt DRONE_WEIGHTS2= ;;   # split probe (train_split.sh)
    f6_t15) run f6_t15 DRONE_WEIGHTS=/workspace/ft6s/ft6s/weights/last.pt DRONE_WEIGHTS2= DRONE_NEW_THR=0.15 ;;
    *) echo "unknown arm $arm" ;;
  esac
done
echo "=== $(date +%H:%M:%S) scoring"
ARGS=""
for arm in ${ARMS:-fa fav v2 v2v f5 f5v fa_t15 fa_t35 v2r v2rv}; do ls $OUT/${arm}_*.jsonl >/dev/null 2>&1 && ARGS="$ARGS $arm=$arm"; done
echo "--- frames 181-249"; $PY bench/score_hold.py $W/gt_hold.json $OUT 181 249 ${NBOOT:-100} $ARGS
echo "--- frames 151-249"; $PY bench/score_hold.py $W/gt_hold.json $OUT 151 249 0 $ARGS
echo HOLD_DONE
