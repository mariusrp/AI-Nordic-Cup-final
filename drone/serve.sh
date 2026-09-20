#!/bin/bash
# (Re)start the production drone server on the pod: ports 9053 (RunPod https proxy) and 22 (direct public TCP).
#   W=weights/best.pt bash /workspace/drone/serve.sh
# Submit URL: http://$RUNPOD_PUBLIC_IP:$RUNPOD_TCP_PORT_22/predict   (fallback https://<podid>-9053.proxy.runpod.net/predict)
cd "$(dirname "$0")"
PIDF=/workspace/drone_server.pid
if [ -f $PIDF ]; then kill $(cat $PIDF) 2>/dev/null; sleep 2; fi
for p in $(ss -ltnp | grep -E ':(22|9053) ' | grep -o 'pid=[0-9]*' | cut -d= -f2); do kill $p 2>/dev/null; done
sleep 1
export YOLO_CONFIG_DIR=/workspace/.ultralytics
DRONE_WEIGHTS=${W:-/workspace/drone_weights/v2.pt} PORTS=${PORTS:-9053,22} nohup /workspace/venv-drone/bin/python server.py \
  >> /workspace/logs/drone_server.log 2>&1 < /dev/null &
echo $! > $PIDF
for i in $(seq 60); do curl -s localhost:9053/ >/dev/null && break; sleep 2; done
echo "server pid $(cat $PIDF) weights ${W:-/workspace/drone_weights/v2.pt}: $(curl -s localhost:9053/) direct: http://$RUNPOD_PUBLIC_IP:$RUNPOD_TCP_PORT_22/predict"
