"""Survival agent endpoint (port 9052), upstream contract: POST /predict StepResponse -> {"actions": [...]}, GET / -> ok.

Holds ONE persistent Hivemind across steps; resets it when a new game starts (sim_time back near 0 / time goes
backwards / agent ids restart). Policy file from env POLICY (default survival/policies/v1.py).
Never raises: any failure falls back to zero-move actions for every agent.
Run:  POLICY=survival/policies/v1.py python3 survival/server.py   (or uvicorn survival.server:app --port 9052)
"""
import importlib.util, os, sys, threading, time, traceback
from typing import Any, Dict, List

from fastapi import FastAPI, Request

HERE = os.path.dirname(os.path.abspath(__file__))
POLICY = os.environ.get("POLICY", os.path.join(HERE, "policies", "v1.py"))
if not os.path.isabs(POLICY) and not os.path.exists(POLICY):
    POLICY = os.path.join(os.path.dirname(HERE), POLICY)
PORT = int(os.environ.get("PORT", "9052"))
LOG_EVERY = float(os.environ.get("LOG_EVERY", "100"))  # sim seconds between log lines

app = FastAPI(title="Survival Simulator Agent Endpoint (team Phillips)")
_lock = threading.Lock()
_TYPES = {"tree": "Tree", "fruit": "Fruit", "agent": "Agent", "predator": "Predator", "edge": "Edge"}


def _load(path):
    spec = importlib.util.spec_from_file_location("survival_policy", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


try:
    _mod = _load(POLICY)
    _load_err = None
except Exception as e:  # keep serving zero actions
    _mod, _load_err = None, repr(e)
    traceback.print_exc()

S = {"hm": None, "last_t": -1.0, "ids": set(), "max_id": -1, "games": 0, "steps": 0, "errors": 0,
     "last_err": None, "last_log_t": 0.0, "slowest_ms": 0.0, "no_sim_time": 0, "sum_ms": 0.0, "n_pred_obs": 0, "n_rel_dir": 0}


def _new_hm():
    S["hm"] = _mod.Hivemind() if _mod is not None else None
    S["ids"], S["max_id"], S["last_log_t"] = set(), -1, 0.0
    S["games"] += 1


def _zero(agents):
    out = []
    for a in agents:
        try:
            aid = int(a.get("agent_id"))
        except Exception:
            continue
        out.append({"agent_id": aid, "move_distance": 0.0, "move_direction": 0.0, "turn_angle": 0.0, "spawn_agent": False})
    return out


def _clean(acts, agents):
    """Coerce policy output to valid ActionRequest dicts; agents missing from the output get zero actions."""
    valid = {int(a["agent_id"]) for a in agents if "agent_id" in a}
    out, seen = [], set()
    for x in acts or []:
        try:
            aid = int(x["agent_id"])
            if aid not in valid or aid in seen:
                continue
            d = {"agent_id": aid,
                 "move_distance": float(x.get("move_distance", 0.0)),
                 "move_direction": float(x.get("move_direction", 0.0)),
                 "turn_angle": float(x.get("turn_angle", 0.0)),
                 "spawn_agent": bool(x.get("spawn_agent", False))}
            for k in ("move_distance", "move_direction", "turn_angle"):
                if d[k] != d[k] or d[k] in (float("inf"), float("-inf")):
                    d[k] = 0.0
            out.append(d); seen.add(aid)
        except Exception:
            continue
    out += [z for z in _zero(agents) if z["agent_id"] not in seen]
    return out


def _maybe_reset(t, agents):
    ids = {int(a["agent_id"]) for a in agents if "agent_id" in a}
    # new game: first request ever, time went backwards, or a fresh t=0 after a real game. A repeated t=0
    # (e.g. a payload without sim_time, see README's 422 note) must NOT reset the herd every step.
    reset = S["hm"] is None or t < S["last_t"] - 1e-6 or (t <= 1e-6 and S["last_t"] > 1e-6)
    # ids restart: none of the current ids known, and all of them at/below the previous max id
    if not reset and ids and S["ids"] and not (ids & S["ids"]) and max(ids) <= S["max_id"]:
        reset = True
    if reset:
        _new_hm()
    S["ids"] |= ids
    if ids:
        S["max_id"] = max(S["max_id"], max(ids))
    S["last_t"] = t


@app.get("/")
def index():
    return {"message": "Agent endpoint running!", "status": "ok", "policy": os.path.basename(POLICY), "load_error": _load_err}


@app.get("/stats")
def stats():
    return {k: (v if k not in ("hm", "ids") else None) for k, v in S.items()} | {"policy": POLICY}


@app.post("/predict")
async def predict(request: Request):
    agents: List[Dict[str, Any]] = []
    try:
        body = await request.json()
        agents = [a for a in (body.get("agent_status") or []) if isinstance(a, dict)]
        st = body.get("sim_time")
        if st is None:  # missing sim_time: continue the current game instead of resetting to t=0
            S["no_sim_time"] += 1
            t = S["last_t"] + 0.1 if S["hm"] is not None else 0.0
        else:
            t = float(st)
    except Exception as e:
        S["errors"] += 1; S["last_err"] = repr(e)
        return {"actions": _zero(agents)}
    t0 = time.perf_counter()
    with _lock:
        try:
            _maybe_reset(t, agents)
            S["steps"] += 1
            if S["hm"] is None or not agents:
                return {"actions": _zero(agents)}
            for a in agents:
                obs = [o for o in (a.get("observations") or []) if isinstance(o, dict)]
                for o in obs:  # portal's verify sample uses lowercase types; the simulator uses "Tree"/"Predator"/...
                    t_ = o.get("type")
                    if isinstance(t_, str):
                        o["type"] = _TYPES.get(t_.lower(), t_)
                    if o.get("type") == "Predator":  # key-presence diagnostics (a missing rel_dir costs ~-237/game)
                        S["n_pred_obs"] += 1
                        if "rel_dir" in o:
                            S["n_rel_dir"] += 1
                a["observations"] = obs
            acts = _clean(S["hm"].act(agents), agents)
        except Exception as e:
            S["errors"] += 1; S["last_err"] = repr(e)
            if S["errors"] <= 5:
                traceback.print_exc()
            acts = _zero(agents)
        ms = (time.perf_counter() - t0) * 1000
        S["slowest_ms"] = max(S["slowest_ms"], ms)
        S["sum_ms"] += ms
        if t - S["last_log_t"] >= LOG_EVERY or S["steps"] % 1000 == 0:
            S["last_log_t"] = t
            print(f"[survival] game={S['games']} step={S['steps']} t={t:.1f} n={len(agents)} score={body.get('score')} "
                  f"errs={S['errors']} mean_ms={S['sum_ms'] / max(1, S['steps']):.1f} slowest_ms={S['slowest_ms']:.1f} "
                  f"no_sim_time={S['no_sim_time']} pred_obs={S['n_pred_obs']} rel_dir={S['n_rel_dir']}", flush=True)
    return {"actions": acts}


if __name__ == "__main__":
    import uvicorn
    print(f"[survival] policy={POLICY} load_error={_load_err}", flush=True)
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="warning")
