#!/bin/bash
# Rebuild the valcity_plus targets + recorded-view datasets and fine-tune v2 on POD=gpu (used when gpu5 died).
# Inputs on the pod: /workspace/drone_seen (recorded validation views, frames <= 150 only), the valcity label kit
# in /workspace/bb1it4/drone/valcity (Happrox.npy, labels/v1_core.json, cands.json), the synthetic set
# /root/dd/data/yolo_v2 and /workspace/ft/code/drone (pushed from the repo). Output: /workspace/ft/runs/ft_all.
set -euo pipefail
export UPSTREAM=/workspace/upstream YOLO_CONFIG_DIR=/workspace/.ultralytics
PY=/workspace/venv-drone/bin/python
V=/workspace/bb1it4/drone/valcity
SEEN=/workspace/drone_seen
FT=/workspace/ft
mkdir -p $FT/up/src $FT/seen
# recordings only (never the Helsinki 'local' replay dir, whose frame numbers would collide with valcity targets)
for d in $SEEN/*/; do b=$(basename $d); [ "$b" = local ] && continue; ln -sfn $d $FT/seen/$b; done
M6="$SEEN/1062106c402f43389f8e4beda556622b/meta.jsonl $SEEN/11b4714ef228449aaf9d258ba25da7b8/meta.jsonl $SEEN/19fd8f1e1c2c4f0bbcb54891c190f09a/meta.jsonl $SEEN/5fd5807bfa504f53948ed293a2a3b4a5/meta.jsonl $SEEN/a2e63304e1b7414aa015a77a9e5783e3/meta.jsonl $SEEN/e010b9079a4e4111b48c305651a401c7/meta.jsonl"
cd /workspace/bb1it4/drone
if [ ! -d $FT/up/src/valcity_v1_plus ]; then
  $PY valcity/build_labels.py $V/Happrox.npy $V/labels/v1_core.json $FT/up/src/valcity_v1_core $M6
  cp /workspace/upstream/drone-flyby/*.py $FT/up/   # valcity_plus.py imports local_evaluator from its upstream arg
  $PY /workspace/ft/code/drone/analysis/valcity_plus.py $FT/up $V/Happrox.npy
fi
echo "plus frames: $(ls $FT/up/src/valcity_v1_plus/annotations | wc -l)"
M8="$M6 $SEEN/c43360927b5645d79146095b269f1858/meta.jsonl $SEEN/08e87581674d4ce3a6c5d931226c2fbb/meta.jsonl"
$PY bb3/make_targets.py $V/Happrox.npy $FT/up/src/valcity_v1_plus $V/cands.json $V/labels/v1_core.json $FT/targets_plus.json $M8
$PY -c "import json; from common import CLASSES; json.dump(list(CLASSES), open('$FT/names.json','w'))"
$PY bb3/build_views.py $FT/seen $FT/targets_plus.json $FT/vc_all 1 150 0 0 $FT/names.json
$PY bb3/build_views.py $FT/seen $FT/targets_plus.json $FT/vc_fold 1 100 126 150 $FT/names.json
cd $FT
ALLOW_GPU_MAIN=1 $PY code/drone/bb3/train_ft5.py ft_all /workspace/drone_weights/v2.pt /root/dd/data/yolo_v2 $FT/vc_all 3 ${EPOCHS:-10} $FT/runs 0.002 workers=4
echo REBUILD_DONE
