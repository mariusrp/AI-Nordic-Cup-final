#!/bin/bash
# FINALS: end-to-end per-view latency of the SINGLE vs the TWO-MODEL (DRONE_UNION) detector path, measured
# through the real server stack (server.py + tracker + planner) on a SPARE port, driven by the organisers'
# local_evaluator in --realtime mode. Production (9053 + 22, 9052, 9054) is never touched.
#   P=9055 W=/workspace/ft5/runs/ft5r/weights/last.pt U=/workspace/drone_weights/v2.pt bash lat_fuse.sh
set -u
P=${P:-9055}
W=${W:-/workspace/ft5/runs/ft5r/weights/last.pt}
U=${U:-/workspace/drone_weights/v2.pt}
PY=/workspace/venv-drone/bin/python
CODE=${CODE:-/workspace/replay/code/drone}
export YOLO_CONFIG_DIR=/workspace/.ultralytics UPSTREAM=/workspace/upstream PYTHONPATH=/workspace/i4/pylib
export DRONE_WEIGHTS=$W DRONE_WEIGHTS2= DRONE_NEW_THR=0.25 DRONE_MAX_REPORT=100 DRONE_SURVEY=0
export DRONE_RECORD_DIR= DRONE_REPLAY_TABLE= DRONE_REPLAY_FAST=0 PORTS=$P
for arm in single union; do
  if [ "$arm" = union ]; then export DRONE_UNION=$U; else export DRONE_UNION=; fi
  LOG=/workspace/logs/lat_$arm.log
  : > $LOG
  for pid in $(ss -ltnp 2>/dev/null | grep -E ":$P " | grep -o 'pid=[0-9]*' | cut -d= -f2); do kill $pid 2>/dev/null; done
  for i in $(seq 30); do ss -ltn | grep -qE ":$P " || break; sleep 2; done
  cd "$CODE" && nohup $PY server.py >> $LOG 2>&1 < /dev/null &
  SP=$!
  for i in $(seq 90); do curl -s localhost:$P/ >/dev/null 2>&1 && break; sleep 2; done
  echo "== arm $arm (pid $SP) union='${DRONE_UNION}' loaded: $(grep -c 'DRONE_UNION\|weights' $LOG)"
  # two back-to-back offline passes = ~2x20 requests of real 4K photogrammetry through the whole stack,
  # then one --realtime pass (333 ms budget, the evaluator drops what we answer late)
  cd /workspace/upstream/drone-flyby
  for r in 1 2; do $PY local_evaluator.py --url http://localhost:$P/predict 2>&1 | tail -3; done
  $PY local_evaluator.py --url http://localhost:$P/predict --realtime 2>&1 | tail -6
  for pid in $(ss -ltnp 2>/dev/null | grep -E ":$P " | grep -o 'pid=[0-9]*' | cut -d= -f2); do kill $pid 2>/dev/null; done
  kill $SP 2>/dev/null; sleep 5
  $PY - "$LOG" "$arm" <<'PY'
import re, sys, statistics as st
ms = [float(m) for m in re.findall(r'\| (\d+(?:\.\d+)?)ms', open(sys.argv[1]).read())]
ms = ms[3:] if len(ms) > 12 else ms  # drop warm-up views
if ms:
    q = st.quantiles(ms, n=100)
    print(f'LAT {sys.argv[2]} n={len(ms)} p50={st.median(ms):.0f} p90={q[89]:.0f} p99={q[98]:.0f} max={max(ms):.0f} mean={st.mean(ms):.0f} ms')
else:
    print('LAT', sys.argv[2], 'no timings in', sys.argv[1])
PY
done
