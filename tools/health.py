#!/usr/bin/env python3
"""Health check based on REAL PROGRESS in the last WINDOW minutes (default 15), not agent idle time.

Per workflow (registry: tools/workflows.json {run_id: "<case>-<tag>"}), progress signals:
  commits  - commits in the last WINDOW min on the workflow's worktree branches, plus main commits that mention it
  ledger   - new/changed ledger lines (<case>/LEDGER.tsv in main or in its worktrees, LEDGER.tsv rows with its case)
  pod      - files written in the last WINDOW min on any pod whose path mentions its branches/worktrees/tag
  (edits   - files edited in its worktrees: shown, but NOT counted as progress on their own)
Verdict: PROGRESS if commits, ledger or pod activity; else STALLED (no progress for WINDOW min) -> investigate/fix.
Also prints pod load. Usage: health.py [--window 15] [--no-pods]
"""
import json, os, re, subprocess, sys, time, glob

WINDOW = int(sys.argv[sys.argv.index("--window") + 1]) if "--window" in sys.argv else 15
NOPODS = "--no-pods" in sys.argv
REPO = "/home/claude/nac"
D = "/root/.claude/projects/-home-claude/323fa704-a9b4-5fa1-bde8-352ff3b28cce/subagents/workflows"
WF = json.load(open(f"{REPO}/tools/workflows.json"))
now = time.time()
cut = now - WINDOW * 60


def sh(cmd, cwd=REPO, timeout=60, env=None):
    try:
        return subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True, timeout=timeout, env=env).stdout
    except Exception:
        return ""


def recent_files(root, skip=(".git",)):
    out = []
    for dp, dns, fns in os.walk(root):
        dns[:] = [d for d in dns if d not in skip and not d.startswith(".venv") and d != "__pycache__"]
        for f in fns:
            p = os.path.join(dp, f)
            try:
                if os.path.getmtime(p) >= cut:
                    out.append(os.path.relpath(p, root))
            except OSError:
                pass
    return out


# ---- pod activity: one listing per pod ----
state = json.load(open(os.path.expanduser("~/.secrets/pod_state.json")))
pods = ["gpu"] + [k[len("pod_id_"):] for k in state if k.startswith("pod_id_")]
pod_recent, pod_load = {}, {}
if not NOPODS:
    for pod in pods:
        cmd = (f"find /workspace /root/dd -xdev \\( -name venv\\* -o -name .cache -o -name hf -o -name .holdout \\) -prune -o "
               f"-type f -newermt '-{WINDOW} minutes' -printf '%h\\n' 2>/dev/null | sort | uniq -c | sort -rn | head -80; "
               f"find /workspace/logs -type f -newermt '-{WINDOW} minutes' 2>/dev/null; "
               "ps -eo pcpu,args --no-headers 2>/dev/null | awk '$1>5' | cut -c1-200 | sed 's/^/PROC /'; echo @@LOAD; "
               "nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null; cat /proc/loadavg")
        out = sh(f'python3 {REPO}/infra/pod.py exec "{cmd}" 60', timeout=100, env={**os.environ, "POD": pod})
        files, _, load = out.partition("@@LOAD")
        pod_recent[pod] = [l for l in files.splitlines() if l.strip()]
        pod_load[pod] = " | ".join(l.strip() for l in load.splitlines() if l.strip()) or "UNREACHABLE"

# ---- main commits in window ----
main_commits = sh(f"git log main --since='{WINDOW} minutes ago' --format='%h %s'").splitlines()

print(f"== workflows (progress in last {WINDOW} min)  {time.strftime('%H:%M')}")
for wid, name in WF.items():
    case, _, tag = name.partition("-")
    d = f"{D}/{wid}"
    ended = False
    running = []
    if os.path.exists(d + "/journal.jsonl"):
        started, done = {}, set()
        for l in open(d + "/journal.jsonl"):
            j = json.loads(l)
            if j.get("type") == "started":
                started[j["key"]] = j.get("label")
            elif "result" in j:
                done.add(j.get("key"))
            if j.get("type") in ("finished", "completed", "stopped", "error"):
                ended = True
        running = [lab for k, lab in started.items() if k not in done]
    wts = sorted(glob.glob(f"{REPO}/.claude/worktrees/{wid}-*"))
    branches = [sh("git rev-parse --abbrev-ref HEAD", cwd=w).strip() for w in wts]
    branches = [b for b in branches if b and b != "HEAD"]
    # commits
    commits = []
    for b in branches:
        commits += [f"{b}: {c}" for c in sh(f"git log {b} --not main --since='{WINDOW} minutes ago' --format='%h %s'").splitlines() if "saved by orchestrator" not in c]
    key = re.compile(rf"\b{case}\b.*\b({tag}|g\d|meta-review|scribe|island)", re.I) if tag == "evolve" else re.compile(rf"{case}.*{tag}|{tag}.*{case}", re.I)
    commits += [f"main: {c}" for c in main_commits if key.search(c)]
    # ledger
    ledger = []
    for p in [f"{w}/{case}/LEDGER.tsv" for w in wts]:
        if os.path.exists(p) and os.path.getmtime(p) >= cut:
            ledger.append(os.path.relpath(p, "/home/claude"))
    mp = f"{REPO}/{case}/LEDGER.tsv"   # main ledger: only if recent lines mention this workflow
    if os.path.exists(mp) and os.path.getmtime(mp) >= cut and any(tag in l for l in open(mp).read().splitlines()[-20:]):
        ledger.append(os.path.relpath(mp, "/home/claude"))
    # pod activity attributable to this workflow
    ab0 = {"medical": "(?:medical|med)", "survival": "(?:survival|surv)", "drone": "drone"}[case]
    pat = [re.escape(b) for b in branches] + [re.escape(os.path.basename(w)) for w in wts] + [rf"(?<![a-z]){ab0}[-_]{tag}\b"]
    if tag == "understand":   # understanding lanes name their pod dirs und1, und_v1, ...
        pat += [rf"/und(?:\d|_)", rf"{ab0}.*analysis"]
    for b in branches:
        m = re.search(r"-(r|g)(\d+)-(\d+)$", b)
        if m:
            ab = {"medical": "(?:medical|med)", "survival": "(?:survival|surv)", "drone": "drone"}[case]
            pat += [rf"(?<![a-z]){ab}[-_](?:{tag}[-_])?{m.group(1)}{m.group(2)}[-_]?{m.group(3)}\b"]
    rx = re.compile("|".join(pat), re.I) if pat else None
    owners = json.load(open(f"{REPO}/tools/pod_owners.json"))
    podact = [f"{pod}:{f.strip()}" for pod, fs in pod_recent.items() for f in fs
              if (rx and rx.search(f)) or (owners.get(pod) == name and not f.startswith("PROC python /podagent"))]
    # edits (informational)
    edits = sum(len(recent_files(w)) for w in wts)
    progress = bool(commits or ledger or podact)
    # last real progress (any time): newest commit on its branches / attributable main commit / ledger write
    lastc = []
    for b in branches:
        o = sh(f"git log -1 {b} --not main --format='%ct|%s'").strip()
        if o and "saved by orchestrator" not in o:
            lastc.append(o)
    for c in sh("git log main -40 --format='%ct|%s'").splitlines():
        if key.search(c.split("|", 1)[-1]):
            lastc.append(c); break
    for pth in [f"{w}/{case}/LEDGER.tsv" for w in wts]:
        if os.path.exists(pth):
            lastc.append(f"{int(os.path.getmtime(pth))}|ledger lines written")
    last = max(lastc, key=lambda x: int(x.split("|")[0])) if lastc else None
    lastdesc = (time.strftime("%H:%M", time.localtime(int(last.split("|")[0]))) + " " + last.split("|", 1)[1][:70]) if last else "none yet"
    if podact and not commits and not ledger:
        lastdesc = "now: pod job running (" + podact[0].split(":", 1)[1].strip()[:50] + ")"
    young = os.path.exists(d) and (now - os.path.getctime(d)) < WINDOW * 60
    verdict = "ENDED" if ended and not running else ("PROGRESS" if progress else ("STARTING (launched < %d min ago)" % WINDOW if young else f"STALLED (no commits/ledger/pod activity for {WINDOW} min)"))
    print(f"{name:16s} {verdict.split(' (')[0]:9s} | last: {lastdesc} | doing: {', '.join(running[:2]) or '-'}")
    print(f"   commits={len(commits)} ledger={len(ledger)} pod_files={len(podact)} worktree_edits={edits} running={running[:4]}")
    for c in commits[:3]:
        print(f"     c {c[:110]}")
    for p in podact[:3]:
        print(f"     p {p[:110]}")

bal = sh(f"python3 {REPO}/infra/pod.py list 2>/dev/null")
try:
    jb = json.loads(bal); print(f"== budget: balance ${jb['clientBalance']:.2f}, spend ${jb['currentSpendPerHr']:.2f}/h, ~{jb['clientBalance']/max(jb['currentSpendPerHr'],0.01):.1f} h left" + ("   <<< BELOW $15: stop experiment pods, keep production (gpu, serve)" if jb['clientBalance'] < 15 else ""))
except Exception:
    pass
if not NOPODS:
    print("== pods (files written in window / load)")
    for pod in pods:
        print(f"-- {pod}: {len(pod_recent.get(pod, []))} files | {pod_load.get(pod)}")
