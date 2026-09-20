#!/usr/bin/env bash
# Refresh the shared drop point on POD=gpu (/workspace/share) and import inbox bundles from other sessions.
set -e
cd /home/claude/nac
mkdir -p /tmp/share_out && git bundle create /tmp/share_out/nac.bundle --all >/dev/null 2>&1 && cp LESSONS.md /tmp/share_out/
python3 infra/pod.py push /tmp/share_out /workspace >/dev/null
python3 infra/pod.py exec "cp /workspace/share_out/nac.bundle /workspace/share_out/LESSONS.md /workspace/share/ && rm -rf /workspace/share_out; mkdir -p /workspace/share/inbox/done" 60 >/dev/null
for f in $(python3 infra/pod.py exec "ls /workspace/share/inbox/*.bundle 2>/dev/null" 30); do
  b=$(basename "$f" .bundle); python3 infra/pod.py get "$f" "/tmp/$b.bundle" >/dev/null
  note=$(python3 infra/pod.py exec "cat /workspace/share/inbox/$b.txt 2>/dev/null" 30 | tr '\n' ' ' | cut -c1-400)
  for ref in $(git bundle list-heads "/tmp/$b.bundle" | awk '{print $2}' | grep refs/heads/); do
    br="ext-$b-${ref#refs/heads/}"; git fetch -q "/tmp/$b.bundle" "$ref:$br" && python3 tools/handoff.py submit "$(echo $b | grep -oE 'survival|medical|drone' | head -1)" "$br" "EXTERNAL (from other session, UNAUDITED): $note" >/dev/null && echo "imported $br"
  done
  python3 infra/pod.py exec "mv /workspace/share/inbox/$b.* /workspace/share/inbox/done/" 30 >/dev/null
done
echo "share synced $(date +%H:%M)"
