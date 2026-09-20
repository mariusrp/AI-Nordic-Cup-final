#!/bin/bash
# Push drone code to the pod at /workspace/drone.  Usage: bash push.sh [with-weights] [with-src]
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
STAGE=$(mktemp -d)/drone
mkdir -p "$STAGE"
EX="--exclude=./data --exclude=./runs --exclude=__pycache__ --exclude=./yolo11n.pt"
[[ " $* " == *" with-weights "* ]] || EX="$EX --exclude=./weights"
(cd "$HERE" && tar cf - $EX .) | (cd "$STAGE" && tar xf -)
if [[ " $* " == *" with-src "* ]]; then mkdir -p "$STAGE/data"; cp -r "$HERE/data/synth_src" "$STAGE/data/"; fi
python3 "$HERE/../infra/pod.py" push "$STAGE"
