#!/bin/bash
# Round 2 on POD=gpu: v1_plus2 labels (eye-verified spacecraft x7, small_plane #4, jammer) -> scene -> targets ->
# recorded-view datasets in /workspace/ft2, then fine-tune from the round-1 weights once they exist.
set -euo pipefail
export UPSTREAM=/workspace/upstream YOLO_CONFIG_DIR=/workspace/.ultralytics
PY=/workspace/venv-drone/bin/python
V=/workspace/bb1it4/drone/valcity
CODE=/workspace/ft/code/drone
FT=/workspace/ft2
mkdir -p $FT/up/src
SEEN=/workspace/drone_seen
M6="$SEEN/1062106c402f43389f8e4beda556622b/meta.jsonl $SEEN/11b4714ef228449aaf9d258ba25da7b8/meta.jsonl $SEEN/19fd8f1e1c2c4f0bbcb54891c190f09a/meta.jsonl $SEEN/5fd5807bfa504f53948ed293a2a3b4a5/meta.jsonl $SEEN/a2e63304e1b7414aa015a77a9e5783e3/meta.jsonl $SEEN/e010b9079a4e4111b48c305651a401c7/meta.jsonl"
M8="$M6 $SEEN/c43360927b5645d79146095b269f1858/meta.jsonl $SEEN/08e87581674d4ce3a6c5d931226c2fbb/meta.jsonl"
cd /workspace/bb1it4/drone
$PY $CODE/valcity/labels/make_plus2.py $V/labels/v1_full.json $FT/v1_plus2.json
$PY valcity/build_labels.py $V/Happrox.npy $FT/v1_plus2.json $FT/up/src/valcity_v1_core $M6
cp /workspace/upstream/drone-flyby/*.py $FT/up/
$PY $CODE/analysis/valcity_plus.py $FT/up $V/Happrox.npy
$PY - <<'EOF'
import json, glob, collections
c = collections.Counter()
for f in glob.glob('/workspace/ft2/up/src/valcity_v1_plus/annotations/*.json'):
    for a in json.load(open(f))['annotations']:
        if not a.get('unsure'):
            c[a['object_id']] += 1
print('plus2 sure boxes per class', dict(c))
EOF
$PY bb3/make_targets.py $V/Happrox.npy $FT/up/src/valcity_v1_plus $V/cands.json $FT/v1_plus2.json $FT/targets.json $M8
$PY bb3/build_views.py /workspace/ft/seen $FT/targets.json $FT/vc_all 1 150 0 0 /workspace/ft/names.json
echo DATA2_DONE
while ! grep -aq 'DONE /workspace' /workspace/logs/ft_train.log; do sleep 60; done
BASE=/workspace/ft/runs/ft_all/weights/last.pt
[ -f $BASE ] || BASE=/workspace/drone_weights/v2.pt
cd $FT
ALLOW_GPU_MAIN=1 $PY $CODE/bb3/train_ft5.py ft2 $BASE /root/dd/data/yolo_v2 $FT/vc_all 3 ${EPOCHS:-6} $FT/runs 0.0015 workers=4
echo REBUILD2_DONE
