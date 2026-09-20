#!/bin/bash
# Table v4 (+ two one-choice alternatives) of the deterministic validation flight. Same serving shape the portal picked
# for v3 (0.6514): 100 rows/frame, runner-up class copies ON, box-scale hedge 0.78/1.3 at conf x0.15.
# v4 = v3 + a crisp-calibrated auto-demotion of the eye-unjudged clusters that reach the top 40 of some frame
# (drone/replay/crisp.py on native-L2 views, threshold calibrated on the eye labels already in overrides_v3.json).
# Inputs from POD=gpu /workspace/replay: dets/ dets_n01/ dets_l2/ H_fit.npy; crisp.json from crisp.py (run on the pod,
# it needs the recorded PNG views).  Usage: bash build_v4.sh PY H.npy DETS DETS_N01 DETS_L2 OUTDIR CRISP.json REPLAY_DIR
set -e
PY=$1; H=$2; D="$3:$4:$5"; O=$6; CRISP=$7; R=${8:-$(dirname "$0")}
FUSE="--levelw 0.5,0.85,1.5 --maxbox 100 --box med --framemap 239=238"
HEDGE="--scales 0.78,1.3 --conf 0.15 --maxbox 100"
mkdir -p "$O"
# 0. reproduce the served v3 clusters/ids (the crisp measurement is keyed to them)
$PY $R/fuse.py $H $D $O/pre_v3.json $O/clusters_v3.json --overrides $R/overrides_v3.json $FUSE --ids $O/ids_v3.json | tail -1
# 1. overrides
$PY $R/mk_overrides.py $R/overrides_v3.json $O/clusters_v3.json $O/ids_v3.json $CRISP $O/overrides_v4.json --thr 1.8 --fac 0.3
$PY $R/mk_overrides.py $R/overrides_v3.json $O/clusters_v3.json $O/ids_v3.json $CRISP $O/overrides_v4_crisp22.json --thr 2.2 --fac 0.3
# 2. tables
for V in v4 v4_crisp22; do
  $PY $R/fuse.py $H $D $O/pre_$V.json $O/clusters_$V.json --overrides $O/overrides_$V.json $FUSE --ids $O/ids_$V.json | tail -1
  $PY $R/hedge.py $O/pre_$V.json $O/ids_$V.json $O/clusters_$V.json $O/table_$V.json $HEDGE | tail -1
done
# 3. alternative: condor/spacecraft class lottery on top of v4 (both classes have NO cluster above support 0.23)
$PY $R/hedge.py $O/pre_v4.json $O/ids_v4.json $O/clusters_v4.json $O/table_v4_lottery.json $HEDGE \
    --addclass "condor:hangar,helicopter,large_launcher:0.12;spacecraft:small_tower,medium_plane,small_plane:0.12" | tail -1
echo BUILD_V4_DONE
