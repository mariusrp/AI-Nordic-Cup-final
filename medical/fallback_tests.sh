#!/usr/bin/env bash
# Fallback tests for the CPU locator (Tllmfolded-g4-3). Needs loc_server.py on :9061 and
# the shared vLLM on :8001. Usage: fallback_tests.sh OUTDIR [which...]
#   dead_heur  LLM dead, sidecar unreachable -> heuristic (old failure path)
#   dead_loc   LLM dead, sidecar up          -> locator
#   err50      50% of LLM calls fail (HTTP 500) -> locator for those
#   llm_new    healthy LLM, new code (must equal llm_old up to LLM nondeterminism)
#   llm_old    healthy LLM, production code (/workspace/med-evolve-g2-1)
#   mock_pair  deterministic mock LLM: old vs new code must give identical answers/spans
#   hang50     50% of LLM calls hang -> deadline path (first 5 conversations)
set -uo pipefail
source "$(dirname "$0")/env.sh"
OUT=$1; shift
mkdir -p "$OUT"
TX=/workspace/tx/large-v3-turbo
cd "$MED_DIR"
for w in "$@"; do
  case $w in
    dead_heur) MED_LLM_URL=http://127.0.0.1:9/v1 MED_LOC_URL=http://127.0.0.1:9 $PY pipeline.py --tx-dir $TX --out $OUT/dead_heur.json ;;
    dead_loc)  MED_LLM_URL=http://127.0.0.1:9/v1 $PY pipeline.py --tx-dir $TX --out $OUT/dead_loc.json ;;
    err50)     $PY flaky_llm.py 18011 error 0.5 & FP=$!; sleep 1
               MED_LLM_URL=http://127.0.0.1:18011/v1 $PY pipeline.py --tx-dir $TX --out $OUT/err50.json; kill $FP ;;
    err50_heur) $PY flaky_llm.py 18012 error 0.5 & FP=$!; sleep 1
               MED_LLM_URL=http://127.0.0.1:18012/v1 MED_LOC_URL=http://127.0.0.1:9 $PY pipeline.py --tx-dir $TX --out $OUT/err50_heur.json; kill $FP ;;
    hang50)    $PY flaky_llm.py 18013 hang 0.5 & FP=$!; sleep 1
               MED_LLM_URL=http://127.0.0.1:18013/v1 $PY pipeline.py --tx-dir $TX --limit 5 --out $OUT/hang50.json; kill $FP ;;
    llm_new)   $PY pipeline.py --tx-dir $TX --out $OUT/llm_new.json ;;
    llm_old)   (cd /workspace/med-evolve-g2-1/medical && $PY pipeline.py --tx-dir $TX --out $OUT/llm_old.json) ;;
    mock_pair) (cd /workspace/med-evolve-g2-1/medical && exec $PY mock_llm.py 18014) & MP=$!; sleep 3
               MED_LLM_URL=http://127.0.0.1:18014/v1 $PY pipeline.py --tx-dir $TX --out $OUT/mock_new.json
               (cd /workspace/med-evolve-g2-1/medical && MED_LLM_URL=http://127.0.0.1:18014/v1 $PY pipeline.py --tx-dir $TX --out $OUT/mock_old.json)
               kill $MP; cmp <(grep -v latency_ms $OUT/mock_old.json | grep -v -E '_s\"') <(grep -v latency_ms $OUT/mock_new.json | grep -v -E '_s\"') && echo MOCK_IDENTICAL ;;
    llm_old2)  (cd /workspace/med-evolve-g2-1/medical && $PY pipeline.py --tx-dir $TX --out $OUT/llm_old2.json) ;;
  esac 2>&1 | grep -v -E 'Warn|warn' | tail -3
  echo "== $w done"
done
