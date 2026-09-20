#!/bin/bash
# Whole-stack smoke test of FINAL serve candidates on a SPARE port (9402) of POD=gpu: the production code
# (/workspace/replay/code/drone) + bb3/serve_ft5.sh exactly as the final restart will use it (no replay table, no
# recording, no survey), then the organisers' local_evaluator in offline and realtime (3 fps clock) mode on the full
# 25-frame Helsinki scene (/workspace/.holdout/upstream-full) and the 5-frame synthetic holdout scene. Reports the
# score, frames answered/skipped and the server-side latency from the arm log. Never touches port 22 / 9053.
#   ARMS="F5 FA F5V" bash run_rt.sh
set -u
CODE=/workspace/replay/code/drone
EV=/workspace/.holdout/upstream-full/drone-flyby
PY=/workspace/venv-drone/bin/python
P=9402
BANK=/workspace/drone_evolve_prod_g22/drone/bankAll.pt
FA=/workspace/ft/runs/ft_all/weights/last.pt
F5=/workspace/ft5/runs/ft5r/weights/last.pt
V2=/workspace/drone_weights/v2.pt
R11=/workspace/drone_weights/r11.pt
export DRONE_RECORD_DIR= DRONE_REPLAY_TABLE=
run() {  # name W W2 VERIFY [extra env]
  local name=$1 w=$2 w2=$3 ver=$4; shift 4
  echo "=== $(date +%T) $name W=$w W2=$w2 VERIFY=$ver $* load=$(cut -d' ' -f1-3 /proc/loadavg)"
  env "$@" ARM=rt_$name P=$P PORTS=$P CODE=$CODE UPSTREAM=/workspace/upstream W=$w W2=$w2 VERIFY=$ver \
    PYPATH=/workspace/i4/pylib bash $CODE/bb3/serve_ft5.sh | tail -1 | cut -c1-200
  for scene in ${SCENES:-helsinki drone_holdout_scene}; do
    sc=$scene; [ $scene = drone_holdout_scene ] && sc=/workspace/.holdout/drone_holdout_scene
    for mode in ${MODES:-offline realtime}; do
      flag=""; [ $mode = realtime ] && flag=--realtime
      out=$(cd $EV && $PY local_evaluator.py --url http://localhost:$P/predict --scene $sc $flag 2>&1 | grep -v Warning)
      echo "-- $scene $mode: $(echo "$out" | grep -E 'COCO mAP' | tr '\n' ' ') | $(echo "$out" | grep -E 'frames sent|frames skipped|frames unanswered|invalid' | tr -s ' ' | tr '\n' ' ')"
    done
  done
  grep -o '| [0-9]*ms' /workspace/logs/ft_arm_rt_$name.log | tr -dc '0-9\n' | sort -n | awk '{a[NR]=$1} END{print "server ms p50", a[int(NR/2)], "p90", a[int(NR*0.9)], "max", a[NR], "n", NR}'
  kill $(cat /workspace/ft/arm_rt_$name.pid) 2>/dev/null; sleep 3
}
for arm in ${ARMS:-F5 FA F5V}; do
  case $arm in
    F5)  run F5  $F5 "" 0 ;;
    FA)  run FA  $FA "" 0 ;;
    F5V) run F5V $F5 "" 1 BANK=$BANK ;;
    FAV) run FAV $FA "" 1 BANK=$BANK ;;
    V2RV) run V2RV $V2 $R11 1 BANK=$BANK ;;
    FA15) run FA15 $FA "" 0 DRONE_NEW_THR=0.15 ;;   # birth threshold, closed loop on a city ft_all never saw
    F515) run F515 $F5 "" 0 DRONE_NEW_THR=0.15 ;;
    F6)   run F6 /workspace/ft6s/ft6s/weights/last.pt "" 0 ;;
    *) echo "unknown arm $arm" ;;
  esac
done
echo RT_DONE
