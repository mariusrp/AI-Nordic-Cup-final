#!/bin/bash
# g2-2 no-harm + latency A/B in ONE run on gpu3: upstream local_evaluator (offline + --realtime) on helsinki frames 0-19
# (worker copy /workspace/g1b/upstream, holdout frames absent): champion (v2 + verifier) vs routed (v2 + r11 planes + verifier).
cd "$(dirname "$0")"
source i4_env.sh
export UPSTREAM=/workspace/g1b/upstream
P=${P:-9372}
for rep in 1 2; do
  for W2 in "" /workspace/drone_weights/r11.pt; do
    DRONE_RECORD_DIR= DRONE_WEIGHTS=/workspace/drone_weights/v2.pt DRONE_WEIGHTS2=$W2 DRONE_VERIFY=/workspace/i4/bankAll.pt PORTS=$P \
      $PY server.py > /workspace/logs/g22_srv_$P.log 2>&1 &
    SP=$!
    for i in $(seq 90); do curl -s localhost:$P/ >/dev/null && break; sleep 2; done
    (cd $UPSTREAM/drone-flyby &&
      echo "== w2=[$W2] rep$rep offline" && $PY local_evaluator.py --url http://localhost:$P/predict 2>&1 | tail -3 &&
      echo "== w2=[$W2] rep$rep realtime" && $PY local_evaluator.py --url http://localhost:$P/predict --realtime 2>&1 | tail -3)
    grep -o "| [0-9]*ms" /workspace/logs/g22_srv_$P.log | tr -dc '0-9\n' | sort -n | awk '{a[NR]=$1} END{print "server ms p50", a[int(NR/2)], "p90", a[int(NR*0.9)], "n", NR}'
    kill $SP; sleep 4
  done
done
echo DONE
