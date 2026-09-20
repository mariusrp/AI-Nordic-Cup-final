#!/usr/bin/env python3
"""RunPod manager: template, create, list, stop, terminate, exec, upload, download.
Usage: pod.py template | create [gpu] | list | stop ID | terminate ID | exec "cmd" [timeout] | bg name "cmd" |
       push [dir] | get remote local | wait"""
import base64, io, json, os, sys, tarfile, time, urllib.request, secrets as pysecrets

KEY = open(os.path.expanduser("~/.secrets/runpod")).read().strip()
STATE = os.path.expanduser("~/.secrets/pod_state.json")
HERE = os.path.dirname(os.path.abspath(__file__))
IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
PORTS = "8000/http,9052/http,9053/http,9054/http,22/tcp"
TEMPLATE_NAME = "nac2026-allcases"


def gql(q, variables=None):
    r = urllib.request.Request("https://api.runpod.io/graphql", data=json.dumps({"query": q, "variables": variables or {}}).encode(),
                               headers={"Content-Type": "application/json", "Authorization": f"Bearer {KEY}", "User-Agent": "curl/8.5.0"})
    d = json.load(urllib.request.urlopen(r, timeout=60))
    if d.get("errors"):
        raise SystemExit(json.dumps(d["errors"], indent=1))
    return d["data"]


def state():
    return json.load(open(STATE)) if os.path.exists(STATE) else {}


def save(s):
    json.dump(s, open(STATE, "w")); os.chmod(STATE, 0o600)


def start_cmd():
    agent = base64.b64encode(open(os.path.join(HERE, "podagent.py"), "rb").read()).decode()
    # boot: agent first (so we can reach the pod fast), everything else is installed by infra/setup.sh via the agent
    return ("bash -c 'mkdir -p /workspace/logs && echo " + agent + " | base64 -d > /podagent.py && "
            "pip install -q fastapi uvicorn >/tmp/pip.log 2>&1; python /podagent.py'")


def template():
    s = state()
    tok = s.get("token") or pysecrets.token_hex(20)
    s["token"] = tok
    q = """mutation($input: SaveTemplateInput) { saveTemplate(input: $input) { id name } }"""
    inp = {"name": TEMPLATE_NAME, "imageName": IMAGE, "containerDiskInGb": 40, "volumeInGb": 60, "volumeMountPath": "/workspace",
           "dockerArgs": start_cmd(), "ports": PORTS, "isServerless": False, "readme": "Nordic AI Cup 2026: all three cases on one pod",
           "env": [{"key": "AGENT_TOKEN", "value": tok}]}
    if s.get("template_id"):
        inp["id"] = s["template_id"]
    t = gql(q, {"input": inp})["saveTemplate"]
    s["template_id"] = t["id"]; save(s)
    print("template", t)


def create(gpu="NVIDIA A40", cloud="COMMUNITY"):
    """Try the requested GPU first, then fall back through cheap >=24GB options."""
    order = [gpu] + [g for g in ["NVIDIA A40", "NVIDIA RTX A6000", "NVIDIA GeForce RTX 4090", "NVIDIA L40S", "NVIDIA RTX 6000 Ada Generation", "NVIDIA L40"] if g != gpu]
    last = None
    for g in order:
        for c in ([cloud, "SECURE"] if cloud == "COMMUNITY" else [cloud]):
            try:
                return _create(g, c)
            except SystemExit as e:
                last = e; print("no capacity:", g, c, str(e)[:160])
    raise SystemExit(f"all options failed: {last}")


def _create(gpu, cloud):
    s = state()
    q = """mutation($input: PodFindAndDeployOnDemandInput) { podFindAndDeployOnDemand(input: $input) { id costPerHr machine { gpuDisplayName } } }"""
    inp = {"cloudType": cloud, "gpuCount": 1, "gpuTypeId": gpu, "name": "nac2026", "templateId": s["template_id"],
           "volumeInGb": 60, "containerDiskInGb": 40, "minVcpuCount": 8, "minMemoryInGb": 30,
           "ports": PORTS, "volumeMountPath": "/workspace", "dockerArgs": start_cmd(),
           "env": [{"key": "AGENT_TOKEN", "value": s["token"]}],
           "allowedCudaVersions": ["12.8", "12.9", "13.0"]}
    p = gql(q, {"input": inp})["podFindAndDeployOnDemand"]
    s["pod_id"] = p["id"]; save(s)
    print("created", p)


def lst():
    d = gql("query { myself { clientBalance currentSpendPerHr pods { id name desiredStatus costPerHr runtime { uptimeInSeconds ports { ip isIpPublic privatePort publicPort type } } machine { gpuDisplayName } } } }")
    print(json.dumps(d["myself"], indent=1))


def stop(pid):
    print(gql("mutation($i: PodStopInput!) { podStop(input: $i) { id desiredStatus } }", {"i": {"podId": pid}}))


def terminate(pid):
    print(gql("mutation($i: PodTerminateInput!) { podTerminate(input: $i) }", {"i": {"podId": pid}}))


POD = os.environ.get("POD", "gpu")  # "gpu" (A40, all cases) or "cpu" (survival simulations)


def pod_id():
    s = state()
    return s["pod_id"] if POD == "gpu" else s["pod_id_" + POD]


def url(port=8000):
    return f"https://{pod_id()}-{port}.proxy.runpod.net"


def create_serve(gpu="NVIDIA RTX A4500"):
    """Small low-latency serving pod in EU-SE-1 (organizer game server is Hetzner Helsinki) with direct TCP ports."""
    s = state()
    q = """mutation($input: PodFindAndDeployOnDemandInput) { podFindAndDeployOnDemand(input: $input) { id costPerHr machine { gpuDisplayName location } } }"""
    last = None
    for g in [gpu, "NVIDIA RTX 4000 Ada Generation", "NVIDIA RTX A5000", "NVIDIA GeForce RTX 3090", "NVIDIA RTX A4000", "NVIDIA GeForce RTX 4090", "NVIDIA A40"]:
        inp = {"cloudType": "SECURE", "gpuCount": 1, "gpuTypeId": g, "name": "nac2026-serve", "imageName": IMAGE,
               "dataCenterId": "EU-SE-1", "supportPublicIp": True, "containerDiskInGb": 20, "volumeInGb": 0,
               "ports": "8000/http,22/tcp,9052/tcp,9055/tcp", "dockerArgs": start_cmd(),
               "env": [{"key": "AGENT_TOKEN", "value": s["token"]}]}
        try:
            p = gql(q, {"input": inp})["podFindAndDeployOnDemand"]
            s["pod_id_serve"] = p["id"]; save(s); print("created serve pod", p); return
        except SystemExit as e:
            last = e; print("no capacity:", g, str(e)[:120])
    raise SystemExit(last)


def create_named(name, kind="gpu", spec="NVIDIA A40"):
    """Extra pods addressed with POD=<name>. kind=gpu (EU-SE-1, public TCP ports) or cpu (spec=instanceId)."""
    s = state()
    if kind == "cpu":
        q = """mutation($input: deployCpuPodInput!) { deployCpuPod(input: $input) { id costPerHr } }"""
        inp = {"instanceId": spec, "cloudType": "SECURE", "containerDiskInGb": 20, "name": f"nac2026-{name}",
               "imageName": "python:3.11-slim", "dockerArgs": start_cmd(), "ports": "8000/http",
               "env": [{"key": "AGENT_TOKEN", "value": s["token"]}]}
        p = gql(q, {"input": inp})["deployCpuPod"]
    else:
        q = """mutation($input: PodFindAndDeployOnDemandInput) { podFindAndDeployOnDemand(input: $input) { id costPerHr machine { gpuDisplayName location } } }"""
        p, last = None, None
        for g in [spec, "NVIDIA A40", "NVIDIA RTX A6000", "NVIDIA L40S", "NVIDIA GeForce RTX 4090", "NVIDIA RTX 6000 Ada Generation"]:
            inp = {"cloudType": "SECURE", "gpuCount": 1, "gpuTypeId": g, "name": f"nac2026-{name}", "imageName": IMAGE,
                   "dataCenterId": "EU-SE-1", "supportPublicIp": True, "containerDiskInGb": 40, "volumeInGb": 60,
                   "volumeMountPath": "/workspace", "ports": "8000/http,22/tcp,9053/tcp,9054/tcp,9052/tcp",
                   "dockerArgs": start_cmd(), "env": [{"key": "AGENT_TOKEN", "value": s["token"]}],
                   "allowedCudaVersions": ["12.8", "12.9", "13.0"]}
            try:
                p = gql(q, {"input": inp})["podFindAndDeployOnDemand"]; break
            except SystemExit as e:
                last = e; print("no capacity:", g, str(e)[:100])
        if p is None:
            raise SystemExit(last)
    s["pod_id_" + name] = p["id"]; save(s)
    print("created", name, p)


def create_cpu(instance="cpu5c-32-64"):
    s = state()
    q = """mutation($input: deployCpuPodInput!) { deployCpuPod(input: $input) { id costPerHr } }"""
    inp = {"instanceId": instance, "cloudType": "SECURE", "containerDiskInGb": 20, "name": "nac2026-cpu",
           "imageName": "python:3.11-slim", "dockerArgs": start_cmd(), "ports": "8000/http,9052/http",
           "env": [{"key": "AGENT_TOKEN", "value": s["token"]}]}
    p = gql(q, {"input": inp})["deployCpuPod"]
    s["pod_id_cpu"] = p["id"]; save(s)
    print("created cpu pod", p)


def call(path, body=None, raw=None, timeout=120, method=None):
    headers = {"x-token": state()["token"], "User-Agent": "curl/8.5.0"}
    data = None
    if body is not None:
        data = json.dumps(body).encode(); headers["Content-Type"] = "application/json"
    if raw is not None:
        data = raw; headers["Content-Type"] = "application/octet-stream"
    r = urllib.request.Request(url() + path, data=data, headers=headers, method=method or ("POST" if data else "GET"))
    return urllib.request.urlopen(r, timeout=timeout).read()


import re as _re


def _guard(cmd):
    """Production servers: only the case's merge authority (env NAC_MERGER=<case>) may restart them."""
    if os.environ.get("NAC_MERGER"):
        return
    prod = False
    if POD == "serve" and _re.search(r"server\.py|uvicorn|pkill|kill\s", cmd):
        prod = True
    if POD == "gpu" and _re.search(r"(server\.py|uvicorn)[^;&|]*(\b9054\b|\b22\b|RUNPOD_TCP_PORT_22)|PORT=(22|9054)\b|--port\s+(22|9054)\b|fuser\s+-k\s+(22|9054)", cmd):
        prod = True
    if prod:
        raise SystemExit("REDEPLOY BLOCKED: this command would (re)start a production server. Only the case's merge authority may redeploy "
                         "(see /home/claude/handoff/authority.json). If that's you, re-run with NAC_MERGER=<case>. Otherwise hand your winner over: "
                         "python3 /home/claude/nac/tools/handoff.py submit <case> <branch> \"<score+-se, audit verdict, summary>\" and use a spare port for tests.")


_HEAVY = _re.compile(r"yolo\S*\s+(detect\s+)?train|\btrain\.sh|vllm\s+serve|start_llm|transcrib|faster_whisper|whisperx|rerank|asr_bench|align\.py|make_synth|torchrun|\.train\(", _re.I)


def _gpu_main_guard(cmd):
    """POD=gpu is reserved for production servers (medical vLLM + server, drone server) and light clients of the
    shared LLM. New heavy GPU jobs must run on POD=gpu3 so the main GPU never runs out of memory."""
    if POD != "gpu" or os.environ.get("NAC_MERGER") or os.environ.get("ALLOW_GPU_MAIN"):
        return
    if _HEAVY.search(cmd):
        raise SystemExit("GPU JOB BLOCKED on POD=gpu: the main GPU is reserved for production (medical LLM ~29 GB, servers) and is near full. "
                         "Run training/transcription/reranking/extra LLMs on POD=gpu3 (A40, 46 GB, venv-drone already there): "
                         "POD=gpu3 python3 /home/claude/nac/infra/pod.py push/exec/bg ... ; copy needed data with get/push. "
                         "Light clients of the shared LLM (pipeline.py/offline_eval against localhost:8001) may still run on POD=gpu.")


def ex(cmd, timeout=90):
    _guard(cmd)
    _gpu_main_guard(cmd)
    r = json.loads(call("/exec", {"cmd": cmd, "timeout": timeout}, timeout=timeout + 30))
    sys.stdout.write(r.get("out", "")); sys.stderr.write(r.get("err", ""))
    return r


def bg(name, cmd):
    _guard(cmd)
    _gpu_main_guard(cmd)
    print(json.loads(call("/exec", {"cmd": cmd, "bg": True, "name": name})))


def push(src=None, dest="/workspace"):
    src = src or os.path.dirname(HERE)
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as t:
        def filt(ti):
            n = ti.name
            if any(x in n for x in ("/.git/", "__pycache__", ".venv")) or n.endswith("/.git"):
                return None
            return ti
        t.add(src, arcname=os.path.basename(src.rstrip("/")), filter=filt)
    print(json.loads(call(f"/upload?dest={dest}", raw=buf.getvalue(), timeout=600)))


def get(remote, local):
    open(local, "wb").write(call(f"/download?path={remote}", timeout=600))
    print("saved", local)


def wait():
    for i in range(90):
        try:
            print(json.loads(call("/", timeout=10))); return
        except Exception as e:
            time.sleep(10)
    raise SystemExit("pod agent not reachable")


if __name__ == "__main__":
    a = sys.argv[1:]
    {"template": lambda: template(), "create": lambda: create(*a[1:]), "list": lambda: lst(),
     "stop": lambda: stop(a[1]), "terminate": lambda: terminate(a[1]),
     "exec": lambda: ex(a[1], int(a[2]) if len(a) > 2 else 90), "bg": lambda: bg(a[1], a[2]),
     "push": lambda: push(*a[1:]), "get": lambda: get(a[1], a[2]), "wait": lambda: wait(),
     "url": lambda: print(url(int(a[1]) if len(a) > 1 else 8000)),
     "create_cpu": lambda: create_cpu(*a[1:]),
     "create_serve": lambda: create_serve(*a[1:]),
     "create_named": lambda: create_named(*a[1:])}[a[0]]()
