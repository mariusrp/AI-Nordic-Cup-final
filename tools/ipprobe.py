"""Tiny probe: logs caller IP headers, answers the survival contract with no-op actions."""
from fastapi import FastAPI, Request
import uvicorn, json, time
app = FastAPI()
@app.get("/")
def root(): return {"message": "probe"}
@app.post("/predict")
async def predict(req: Request):
    b = await req.json()
    h = {k: v for k, v in req.headers.items() if k.lower() in ("x-forwarded-for", "cf-connecting-ip", "x-real-ip", "user-agent")}
    open("/workspace/logs/ipprobe.jsonl", "a").write(json.dumps({"t": time.time(), "client": req.client.host, "h": h}) + "\n")
    return {"actions": [{"agent_id": a["agent_id"], "move_distance": 0.0, "move_direction": 0.0, "turn_angle": 0.0, "spawn_agent": False} for a in b.get("agent_status", [])]}
uvicorn.run(app, host="0.0.0.0", port=9052, log_level="warning")
