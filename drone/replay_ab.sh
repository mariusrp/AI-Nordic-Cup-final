#!/bin/bash
# Offline causal replay A/B (POD=gpu3): recorded validation flights (frames 1-150) through the full stack with
# the champion verifier, weights A vs B.  bash replay_ab.sh OUTDIR "v2 r11 route"
cd "$(dirname "$0")"
source i4_env.sh
OUT=${1:-/workspace/g22/out}; ARMS=${2:-"v2 r11"}
mkdir -p $OUT
for S in 5fd5807bfa504f53948ed293a2a3b4a5 1062106c402f43389f8e4beda556622b a2e63304e1b7414aa015a77a9e5783e3; do
  for A in $ARMS; do
    W=$A; W2=
    if [ "$A" = route ]; then W=v2; W2=/workspace/drone_weights/r11.pt; fi   # v2 + r11 for helicopter/jet_plane/small_plane
    DRONE_WEIGHTS2=$W2 DRONE_WEIGHTS=/workspace/drone_weights/$W.pt DRONE_VERIFY=${VERIFY-/workspace/i4/bankAll.pt} $PY replay_causal.py \
      /workspace/drone_seen/$S $OUT/${A}_$S.jsonl 150 2>&1 | grep -v "INFO" | tail -3
  done
done
echo DONE
