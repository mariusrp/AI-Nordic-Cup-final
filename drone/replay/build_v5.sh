#!/bin/bash
# Table v5 (+ two one-choice alternatives) of the deterministic validation flight. Serving shape unchanged from the
# one the portal picked for v3/v3_spc (0.6514 / 0.6623 LIVE): 100 rows/frame, runner-up class copies ON, box-scale
# hedge 0.78/1.3 at conf x0.15, --framemap 239=238.
# v5 = v3 + a size-ranked SPACECRAFT SLATE (drone/replay/mk_v5.py) + the late-strip (t* > 252) eye pass.
#   v5_slate14 : the slate is 14 clusters wide instead of 6          (one choice: how many spacecraft candidates rank high)
#   v5_keepspc : no purge - every v1 spacecraft cluster keeps its support (one choice: whether sub-IoU-size candidates are dropped)
# Inputs from POD=gpu /workspace/replay: dets/ dets_n01/ dets_l2/ H_fit.npy; CLUSTERS_V1 = fuse.py run with
# overrides_v1.json (natural, un-overridden supports), which is what mk_v5.py ranks on.
# Usage: bash build_v5.sh PY H.npy DETS DETS_N01 DETS_L2 OUTDIR CLUSTERS_V1.json [REPLAY_DIR]
set -e
PY=$1; H=$2; D="$3:$4:$5"; O=$6; C1=$7; R=${8:-$(dirname "$0")}
FUSE="--levelw 0.5,0.85,1.5 --maxbox 100 --box med --framemap 239=238"
HEDGE="--scales 0.78,1.3 --conf 0.15 --maxbox 100"
mkdir -p "$O"
$PY $R/mk_v5.py $C1 $R/overrides_v3.json $O/overrides_v5.json          --slate 6  --net 12
$PY $R/mk_v5.py $C1 $R/overrides_v3.json $O/overrides_v5_slate14.json  --slate 14 --net 12
$PY $R/mk_v5.py $C1 $R/overrides_v3.json $O/overrides_v5_keepspc.json  --slate 6  --net 12 --keepspc 1
for V in v5 v5_slate14 v5_keepspc; do
  $PY $R/fuse.py $H $D $O/pre_$V.json $O/clusters_$V.json --overrides $O/overrides_$V.json $FUSE --ids $O/ids_$V.json | tail -2
  $PY $R/hedge.py $O/pre_$V.json $O/ids_$V.json $O/clusters_$V.json $O/table_$V.json $HEDGE | tail -1
done
echo BUILD_V5_DONE
