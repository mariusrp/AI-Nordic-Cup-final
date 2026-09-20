#!/bin/bash
# Switch medical PRODUCTION (:9054 on POD=gpu, /workspace/med-attach/medical) to island units.
# Usage: bash tools/med_switch_island.sh "MED_ISLAND_GAP=0.5 MED_ISLAND_THR=-40"
# Steps: back up spans.py/pipeline.py, upload the n3 versions (island is the code default), restart :9054 with the
# AEQ env + the given island params, verify the serving env and LLM binding, then one portal validation on :9054.
set -u
cd "$(dirname "$0")/.."
PARAMS=${1:-}
STAGE=/Users/adrian/blinq/projects/nac-n3-tmp/push_prod; rm -rf $STAGE; mkdir -p $STAGE/medical
cp medical/spans.py medical/pipeline.py $STAGE/medical/
echo "== $(date +%H:%M) backup + upload"
NAC_MERGER=medical POD=gpu python3 infra/pod.py exec "cd /workspace/med-attach/medical && cp -n spans.py spans.py.bak_sentence && cp -n pipeline.py pipeline.py.bak_sentence && ls -la *.bak_sentence | cut -c1-70" 30 | tail -2
NAC_MERGER=medical POD=gpu python3 infra/pod.py push $STAGE/medical /workspace/med-attach | tail -1
NAC_MERGER=medical POD=gpu python3 infra/pod.py exec "cd /workspace/med-attach/medical && md5sum spans.py pipeline.py | cut -c1-12 && grep -c 'def speech_islands' spans.py" 30 | tail -3
md5 -q medical/spans.py | cut -c1-12; md5 -q medical/pipeline.py | cut -c1-12
echo "== $(date +%H:%M) restart :9054"
NAC_MERGER=medical POD=gpu python3 infra/pod.py exec "p=\$(ss -ltnp | grep ':9054 ' | grep -o 'pid=[0-9]*' | cut -d= -f2 | head -1); [ -n \"\$p\" ] && kill \$p; sleep 4; ss -ltnp | grep -c ':9054 '" 30 | tail -1
NAC_MERGER=medical POD=gpu python3 infra/pod.py bg "med_prod_island" "cd /workspace/med-attach/medical && source env.sh && MED_PORT=9054 MED_LLM_URL=http://127.0.0.1:8001/v1 MED_ORDER=aeq MED_SPAN_COVERAGE_NEXT=1 MED_USE_QUOTE=0 MED_UNIT_MODE=island $PARAMS MED_REQ_LOG=/workspace/logs/med_requests_prod_island.jsonl \$PY server.py" | tail -1
got=$(NAC_MERGER=medical POD=gpu python3 infra/pod.py exec "for i in \$(seq 80); do curl -s localhost:9054/ | grep -q running && break; sleep 3; done; p=\$(ss -ltnp | grep ':9054 ' | grep -o 'pid=[0-9]*' | cut -d= -f2 | head -1); tr '\0' '\n' < /proc/\$p/environ | grep -E '^MED_' | grep -v MED_REQ_LOG | sort | tr '\n' ' '; echo; curl -s localhost:9054/api | python3 -c 'import sys,json;d=json.load(sys.stdin);print(\"llm\",d[\"llm\"],\"asr\",d[\"asr\"],\"unit_mode\",d.get(\"span_cfg\",{}).get(\"unit_mode\"))'" 300 | tail -2)
echo "serving: $got"
echo "== $(date +%H:%M) production validation on :9054"
python3 tools/portal.py validate medical https://5g74a0atgzlxwf-9054.proxy.runpod.net/predict 2>&1 | grep -E "^SCORE|not queued" | tail -1
echo "SWITCH_DONE (rollback: cd /workspace/med-attach/medical && cp spans.py.bak_sentence spans.py && cp pipeline.py.bak_sentence pipeline.py, restart :9054 with the AEQ env)"
