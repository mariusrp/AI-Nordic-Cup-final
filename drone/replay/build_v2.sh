#!/bin/bash
# Table v2 (+ one-choice alternatives) of the deterministic validation flight. Inputs (fetched from POD=gpu /workspace/replay):
#   dets/ (old recorded views, 7 models x 1280/1920), dets_n01/ (new L0/L1 views of run 3b1ce119), dets_l2/ (native L2 survey
#   views of runs bb012122/fba5ec20/c805d021: 7 models x 960/1280/1920 + ft_all/ft_m/wise085 x 2560).
# Usage: bash build_v2.sh PY H.npy DETS_OLD DETS_N01 DETS_L2 OUTDIR [OVERRIDES (default overrides_v2.json)] [SUFFIX]
set -e
PY=$1; H=$2; D="$3:$4:$5"; O=$6; R=$(dirname "$0"); OV=${7:-$R/overrides_v2.json}; X=${8:-}; KS=${OV%.json}_keepspc.json
mkdir -p "$O"
COMMON="--levelw 0.5,0.85,1.5 --maxbox 100"
$PY $R/fuse.py $H $D $O/table_v2$X.json $O/clusters_v2.json --overrides $OV $COMMON --box med --ids $O/ids_v2.json | tail -2
$PY $R/fuse.py $H $D $O/table_v2_boxL2$X.json $O/cl_boxL2.json --overrides $OV $COMMON --box l2 | tail -1
$PY $R/fuse.py $H $D $O/table_v2_norun$X.json $O/cl_norun.json --overrides $OV $COMMON --box med --runnerup 1 | tail -1
$PY $R/fuse.py $H $D $O/table_v2_keepspc$X.json $O/cl_keepspc.json --overrides $KS $COMMON --box med | tail -1
$PY $R/fuse.py $H $D $O/table_v2_strict$X.json $O/cl_strict.json --overrides $OV $COMMON --box med --unjudged 0.3 | tail -1
echo BUILD_V2_DONE
