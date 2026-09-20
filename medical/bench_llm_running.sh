#!/usr/bin/env bash
# Benchmark the answerer against the vLLM server ALREADY running on :8001 (no restart).
#   bash medical/bench_llm_running.sh <tag> [tx_dir]
# Writes /workspace/runs/med/llm_<tag>.json (+ .detail.json) and appends to SUMMARY.txt.
set -uo pipefail
source "$(dirname "$0")/env.sh"
TAG=${1:?tag}
TX=${2:-/workspace/tx/large-v3-turbo}
OUT=/workspace/runs/med; mkdir -p "$OUT"; SUM=$OUT/SUMMARY.txt
cd "$MED_DIR"
M=$(curl -s http://127.0.0.1:8001/v1/models | $PY -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])')
echo "== LLM(bench_running) $M tag=$TAG on $TX (think=${MED_LLM_THINK:-0})" | tee -a "$SUM"
$PY pipeline.py --tx-dir "$TX" --out "$OUT/llm_$TAG.json" 2>"$OUT/llm_$TAG.log" | tee -a "$SUM"
for s in dev test all; do echo "$s: $($PY offline_eval.py "$OUT/llm_$TAG.json" --split $s --json)" | tee -a "$SUM"; done
$PY tune_bias.py "$OUT/llm_$TAG.detail.json" | tee -a "$SUM"
grep -c '"src": "heur"' "$OUT/llm_$TAG.detail.json" | sed 's/^/heuristic fallbacks: /' | tee -a "$SUM"
