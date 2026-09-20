#!/bin/bash
# g2-2 platform A/B servers on POD=gpu3 (direct TCP): A = champion (v2 + SAFER + verifier) on 9053 -> :22080,
# B = routed (v2 + r11 for helicopter/jet_plane/small_plane + SAFER + verifier) on 9054 -> :22081.
cd "$(dirname "$0")"
source i4_env.sh
for P in 9053 9054; do for p in $(ss -ltnp | grep ":$P " | grep -o 'pid=[0-9]*' | cut -d= -f2); do kill $p; done; done
sleep 2
DRONE_RECORD_DIR=/workspace/g22/rec_A DRONE_WEIGHTS=/workspace/drone_weights/v2.pt DRONE_VERIFY=/workspace/i4/bankAll.pt PORTS=9053 \
  nohup $PY server.py > /workspace/logs/g22_A.log 2>&1 < /dev/null &
DRONE_RECORD_DIR=/workspace/g22/rec_B DRONE_WEIGHTS=/workspace/drone_weights/v2.pt DRONE_WEIGHTS2=/workspace/drone_weights/r11.pt \
  DRONE_VERIFY=/workspace/i4/bankAll.pt PORTS=9054 nohup $PY server.py > /workspace/logs/g22_B.log 2>&1 < /dev/null &
for i in $(seq 90); do curl -s localhost:9053/ >/dev/null && curl -s localhost:9054/ >/dev/null && break; sleep 2; done
echo "A http://$RUNPOD_PUBLIC_IP:$RUNPOD_TCP_PORT_9053/predict  B http://$RUNPOD_PUBLIC_IP:$RUNPOD_TCP_PORT_9054/predict"
