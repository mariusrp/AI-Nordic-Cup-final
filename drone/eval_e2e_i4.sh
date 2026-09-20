#!/bin/bash
# I4 end-to-end A/B in ONE run on gpu3: upstream local_evaluator (offline + --realtime) on helsinki frames 0-19
# (worker copy /workspace/g1b/upstream, holdout frames absent), v2 alone vs v2 + DINOv2 verifier.
#   bash eval_e2e_i4.sh /workspace/i4/bankAll.pt
cd "$(dirname "$0")"
source i4_env.sh
export UPSTREAM=/workspace/g1b/upstream
P=${P:-9361}
BANK=${1:-/workspace/i4/bankAll.pt}
for V in "" "$BANK"; do
  for rep in 1 2; do
    DRONE_RECORD_DIR= DRONE_WEIGHTS=/workspace/drone_weights/v2.pt DRONE_VERIFY=$V PORTS=$P $PY server.py > /workspace/logs/i4_srv_$P.log 2>&1 &
    SP=$!
    for i in $(seq 90); do curl -s localhost:$P/ >/dev/null && break; sleep 2; done
    (cd $UPSTREAM/drone-flyby &&
      echo "== verify=[$V] rep$rep offline" && $PY local_evaluator.py --url http://localhost:$P/predict 2>&1 | tail -4 &&
      echo "== verify=[$V] rep$rep realtime" && $PY local_evaluator.py --url http://localhost:$P/predict --realtime 2>&1 | tail -4)
    grep -iE "ms|latency" /workspace/logs/i4_srv_$P.log | tail -2
    kill $SP; sleep 4
  done
done
