"""Pod self-stop watchdog. Stops THIS pod (runpodctl stop pod $RUNPOD_POD_ID) after IDLE_MIN minutes with no activity.
Activity = GPU util > 5%, 1-min load > 1.0, files written under /workspace in the last 5 min, or the control agent
(podagent.py) served a request (its CPU time grew). The orchestrator's 20-minute health checks count as activity, so
pods only stop if the orchestrating session has gone quiet for IDLE_MIN minutes."""
import os, subprocess, time
IDLE_MIN = float(os.environ.get("IDLE_MIN", "60"))
LOG = "/workspace/logs/watchdog.log"
os.makedirs("/workspace/logs", exist_ok=True)


def sh(c):
    try:
        return subprocess.run(c, shell=True, capture_output=True, text=True, timeout=30).stdout
    except Exception:
        return ""


def agent_cpu():
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            if b"podagent.py" in open(f"/proc/{pid}/cmdline", "rb").read():
                f = open(f"/proc/{pid}/stat").read().rsplit(")", 1)[1].split()
                return int(f[11]) + int(f[12])
        except Exception:
            pass
    return 0


last_active, last_cpu = time.time(), agent_cpu()
while True:
    time.sleep(60)
    reasons = []
    g = sh("nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null").strip()
    if g and max(int(x) for x in g.split()) > 5:
        reasons.append(f"gpu {g}%")
    load = float(open("/proc/loadavg").read().split()[0])
    if load > 1.0:
        reasons.append(f"load {load}")
    if sh("find /workspace -xdev -path /workspace/logs/watchdog.log -prune -o -type f -mmin -5 -print -quit 2>/dev/null").strip():
        reasons.append("files")
    c = agent_cpu()
    if c != last_cpu:
        reasons.append("agent request")
    last_cpu = c
    if reasons:
        last_active = time.time()
    idle = (time.time() - last_active) / 60
    with open(LOG, "a") as f:
        f.write(f"{time.strftime('%H:%M')} idle={idle:.0f}m {','.join(reasons)}\n")
    if idle >= IDLE_MIN:
        with open(LOG, "a") as f:
            f.write(f"{time.strftime('%H:%M')} STOPPING pod {os.environ.get('RUNPOD_POD_ID')} after {idle:.0f} idle minutes\n")
        sh(f"runpodctl stop pod {os.environ['RUNPOD_POD_ID']}")
        time.sleep(600)
