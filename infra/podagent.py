"""Tiny authenticated control agent that runs inside the RunPod pod (port 8000, via RunPod HTTPS proxy).
Lets the orchestrator (which only has HTTPS egress) run commands and move files. Token = env AGENT_TOKEN."""
import os, subprocess, io, tarfile, time, uuid
from fastapi import FastAPI, Request, HTTPException, Header
from fastapi.responses import PlainTextResponse, FileResponse
import uvicorn

TOKEN = os.environ["AGENT_TOKEN"]
app = FastAPI()
JOBS = {}


def auth(tok):
    if tok != TOKEN:
        raise HTTPException(401, "bad token")


@app.get("/")
def root():
    return {"ok": True, "time": time.time()}


@app.post("/exec")
async def exec_(req: Request, x_token: str = Header(None)):
    auth(x_token)
    b = await req.json()
    cmd, timeout = b["cmd"], b.get("timeout", 90)
    if b.get("bg"):
        jid = b.get("name") or uuid.uuid4().hex[:8]
        log = f"/workspace/logs/{jid}.log"
        os.makedirs("/workspace/logs", exist_ok=True)
        p = subprocess.Popen(["bash", "-lc", f"({cmd}) > {log} 2>&1"], start_new_session=True)
        JOBS[jid] = p
        return {"job": jid, "log": log, "pid": p.pid}
    try:
        r = subprocess.run(["bash", "-lc", cmd], capture_output=True, text=True, timeout=timeout)
        return {"code": r.returncode, "out": r.stdout[-20000:], "err": r.stderr[-8000:]}
    except subprocess.TimeoutExpired as e:
        return {"code": -1, "out": (e.stdout or b"")[-5000:] if isinstance(e.stdout, str) else "", "err": "timeout"}


@app.get("/jobs")
def jobs(x_token: str = Header(None)):
    auth(x_token)
    return {k: (p.poll()) for k, p in JOBS.items()}


@app.post("/upload")
async def upload(req: Request, dest: str = "/workspace", x_token: str = Header(None)):
    auth(x_token)
    data = await req.body()
    os.makedirs(dest, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as t:
        t.extractall(dest)
    return {"ok": True, "bytes": len(data)}


@app.get("/download")
def download(path: str, x_token: str = Header(None)):
    auth(x_token)
    return FileResponse(path)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")
