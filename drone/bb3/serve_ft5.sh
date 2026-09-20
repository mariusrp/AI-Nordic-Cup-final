#!/bin/bash
# Serve one drone A/B arm from the pushed production code (/workspace/ft/code/drone) on any pod.
#   ARM=<name> P=<port> [PORTS=9053,22] W=<weights.pt> [W2=<routed weights or "">] [WB=<split B weights>]
#   [WB2=<split B routed or "">] [SPLIT=181] [VERIFY=1] [BANK=<bankAll.pt>] [R11=<r11.pt>] [PYPATH=<extra PYTHONPATH>]
#   bash serve_ft5.sh
# gpu5: 9053 -> public 194.68.245.228:22161, 9054 -> :22162, 9052 -> :22163. gpu (production): PORTS=9053,22 -> 194.68.245.26:22174,
#   BANK=/workspace/drone_evolve_prod_g22/drone/bankAll.pt R11=/workspace/drone_weights/r11.pt PYPATH=/workspace/i4/pylib.
# Recording is OFF unless DRONE_RECORD_DIR is inherited from the caller (replay arms record the whole flight).
set -u
CODE=${CODE:-/workspace/ft/code/drone}
cd "$CODE"
P=${P:-9053}
for pid in $(ss -ltnp | grep -E ":$P " | grep -o 'pid=[0-9]*' | cut -d= -f2); do kill $pid 2>/dev/null; done
sleep 1
export UPSTREAM=${UPSTREAM:-/workspace/ft/upstream} YOLO_CONFIG_DIR=/workspace/ft/.ultra OMP_NUM_THREADS=2
R11=${R11:-/workspace/bb5d/xa/drone_weights/r11.pt}
export DRONE_RECORD_DIR=${DRONE_RECORD_DIR:-} DRONE_WEIGHTS=$W PORTS=${PORTS:-$P}
export DRONE_WEIGHTS2=${W2-$R11}
[ -n "${PYPATH:-}" ] && export PYTHONPATH=$PYPATH
[ -n "${UNION:-}" ] && export DRONE_UNION=$UNION
if [ "${VERIFY:-1}" = 1 ]; then export DRONE_VERIFY=${BANK:-/workspace/ft/serve/bankAll.pt}; fi
if [ -n "${WB:-}" ]; then export DRONE_SPLIT_FRAME=${SPLIT:-181} DRONE_WEIGHTS_B=$WB DRONE_WEIGHTS2_B=${WB2-$R11}; fi
nohup /workspace/venv-drone/bin/python server.py > /workspace/logs/ft_arm_${ARM}.log 2>&1 < /dev/null &
echo $! > /workspace/ft/arm_${ARM}.pid
for i in $(seq 90); do curl -s localhost:$P/ >/dev/null 2>&1 && break; sleep 2; done
echo "$ARM on $P (pid $(cat /workspace/ft/arm_${ARM}.pid)): $(curl -s localhost:$P/) W=$W W2=$DRONE_WEIGHTS2 WB=${WB:-} UNION=${UNION:-} VERIFY=${VERIFY:-1}"
