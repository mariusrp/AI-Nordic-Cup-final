#!/usr/bin/env bash
# Benchmarks for the medical case on the pod. Run from anywhere:
#   bash medical/pod_bench.sh asr            # transcribe 39 train files with turbo + large-v3 (timing, span oracle)
#   bash medical/pod_bench.sh llm [models]   # each LLM on cached turbo transcripts: score + timing + bias sweep
#   bash medical/pod_bench.sh e2e [model]    # vLLM + server + upstream local_evaluator (true end-to-end)
# Results: /workspace/runs/med/*.json and a summary appended to /workspace/runs/med/SUMMARY.txt
set -uo pipefail
source "$(dirname "$0")/env.sh"
OUT=/workspace/runs/med; mkdir -p "$OUT"
AUDIO=$UPSTREAM/medical-appointment/data/audio
SUM=$OUT/SUMMARY.txt
cd "$MED_DIR"
stage=${1:-all}; shift || true

run_asr() {
  for M in ${MED_ASR_CANDIDATES:-large-v3-turbo large-v3}; do
    echo "== ASR $M" | tee -a "$SUM"
    MED_ASR_MODEL=$M $PY pipeline.py --audio-dir "$AUDIO" --backend heuristic \
        --save-tx /workspace/tx/$M --out "$OUT/asr_$M.json" 2>"$OUT/asr_$M.log" | tee -a "$SUM"
    $PY spans.py --tx /workspace/tx/$M | tail -9 | tee -a "$SUM"
    echo "heuristic on $M: $($PY offline_eval.py "$OUT/asr_$M.json" --json)" | tee -a "$SUM"
  done
}

run_llm() {
  TX=${MED_BENCH_TX:-/workspace/tx/large-v3-turbo}
  for M in ${@:-cyankiwi/Qwen3.8-27B-AWQ-INT4 Qwen/Qwen3.5-35B-A3B-GPTQ-Int4}; do
    tag=$(echo "$M" | tr '/' '_')_t${MED_LLM_THINK:-0}
    echo "== LLM $M on $TX (think=${MED_LLM_THINK:-0})" | tee -a "$SUM"
    bash start_llm.sh "$M" "${MED_GPU_MEM:-0.6}" || { echo "start failed: $M" | tee -a "$SUM"; continue; }
    $PY pipeline.py --tx-dir "$TX" --out "$OUT/llm_$tag.json" 2>"$OUT/llm_$tag.log" | tee -a "$SUM"
    echo "default bias: $($PY offline_eval.py "$OUT/llm_$tag.json" --json)" | tee -a "$SUM"
    $PY tune_bias.py "$OUT/llm_$tag.detail.json" | tee -a "$SUM"
    $PY tune_bias.py "$OUT/llm_$tag.detail.json" --spans unit --biases 0,1,2 | sed 's/^/[unit spans] /' | tee -a "$SUM"
    grep -c '"src": "heur"' "$OUT/llm_$tag.detail.json" | sed 's/^/heuristic fallbacks: /' | tee -a "$SUM"
  done
}

run_e2e() {
  M=${1:-cyankiwi/Qwen3.8-27B-AWQ-INT4}
  echo "== E2E $M asr=${MED_ASR_MODEL:-large-v3-turbo}" | tee -a "$SUM"
  curl -s http://127.0.0.1:8001/v1/models | grep -q "$M" || bash start_llm.sh "$M" "${MED_GPU_MEM:-0.6}"
  bash start_server.sh || return 1
  (cd "$UPSTREAM/medical-appointment" && $PY local_evaluator.py --url http://127.0.0.1:9054/predict) | tail -30 | tee -a "$SUM"
}

case $stage in
  asr) run_asr ;;
  llm) run_llm "$@" ;;
  e2e) run_e2e "$@" ;;
  all) run_asr; run_llm "$@"; run_e2e ;;
esac
