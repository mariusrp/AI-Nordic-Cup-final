#!/bin/bash
# Frozen local scorer run on the pod: start a throw-away server on $P with $W weights, run upstream
# local_evaluator.py (default + --realtime) and det_eval.py.   W=weights/q1.pt bash eval_local.sh
cd "$(dirname "$0")"
PY=/workspace/venv-drone/bin/python
export YOLO_CONFIG_DIR=/workspace/.ultralytics
P=${P:-9061}
W=${W:-weights/best.pt}
DRONE_RECORD_DIR=/workspace/drone_local_rec DRONE_WEIGHTS=$W PORTS=$P $PY server.py > /workspace/logs/drone_eval_srv_$P.log 2>&1 &
SP=$!
for i in $(seq 60); do curl -s localhost:$P/ >/dev/null && break; sleep 2; done
cd /workspace/upstream/drone-flyby
echo "== $W ${SCENE:-helsinki} offline";  $PY local_evaluator.py --url http://localhost:$P/predict ${SCENE:+--scene $SCENE} 2>&1 | tail -${TAILN:-8}
echo "== $W ${SCENE:-helsinki} realtime"; $PY local_evaluator.py --url http://localhost:$P/predict ${SCENE:+--scene $SCENE} --realtime ${LAT:+--simulate-latency-ms $LAT} 2>&1 | tail -${TAILN:-8}
kill $SP
cd - >/dev/null
[ -n "${NODET:-}${SCENE:-}" ] || $PY det_eval.py --weights $W 2>&1 | grep -E "^L|DET_EVAL"
