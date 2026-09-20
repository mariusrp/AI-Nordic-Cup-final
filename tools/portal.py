#!/usr/bin/env python3
"""Portal helper: queue a validation and poll it to completion, or summarise validation history.
The /validate/queue response carries queued_attempt_uuid at the top level (nac.py's poller expects a
'data' wrapper and exits early), so poll here. Final evaluation is deliberately not implemented.

    python3 tools/portal.py validate <survival|drone|medical> <url>   # queue + poll, prints the score
    python3 tools/portal.py history <case> [n]                          # last n validations (url, score)
    python3 tools/portal.py leaderboard <case>                          # our rank + top 10
"""
import json
import os
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path

BASE = "https://cases.nordicaicup.com/api/v1/usecases"
LB = "https://cases.nordicaicup.com/api/v1/leaderboard/validation"
CASES = {"survival": "survival-simulator", "drone": "drone-flyby", "medical": "medical-appointment"}

def api_key():
    """Prefer the operator workspace's environment convention; never write a key."""
    key = os.environ.get("NORDIC_API_KEY", "").strip()
    if not key:
        path = Path.home() / ".secrets" / "nac_api_key"
        if path.is_file():
            key = path.read_text().strip()
    if not key:
        raise SystemExit("Set NORDIC_API_KEY or provision ~/.secrets/nac_api_key outside the repository.")
    return key


def timing(attempt):
    """Portal wall time excludes queue time; score*10 is only a tick estimate."""
    try:
        wall = (datetime.fromisoformat(attempt["finished_at"]) -
                datetime.fromisoformat(attempt["started_at"])).total_seconds()
    except (KeyError, TypeError, ValueError):
        return {}
    return {"wall_seconds": wall}


def req(path, method="GET", body=None, timeout=60):
    parts = path.split("/")
    allowed = (len(parts) == 2 and parts[1] in ("status", "verify")) or (
        len(parts) in (3, 4, 5) and parts[1:3] == ["validate", "queue"]
        and (len(parts) != 5 or parts[4] == "attempt"))
    if not parts or parts[0] not in CASES.values() or not allowed:
        raise ValueError("Only status, verify, and validation routes are supported")
    data = json.dumps(body).encode() if body is not None else None
    r = urllib.request.Request(f"{BASE}/{path}", data=data, method=method,
                               headers={"x-token": api_key(), "Accept": "application/json", "Content-Type": "application/json",
                                        "User-Agent": "curl/8.5.0"})
    try:
        with urllib.request.urlopen(r, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read() or b"null")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()[:1000]


def validate(case, url):
    code, r = req(f"{case}/validate/queue", "POST", {"url": url})
    print(code, r, flush=True)
    uuid = r.get("queued_attempt_uuid") if isinstance(r, dict) else None
    if not uuid:
        sys.exit("not queued")
    t0 = time.time()
    while True:
        time.sleep(15)
        c, q = req(f"{case}/validate/queue/{uuid}")
        st = q.get("status") if isinstance(q, dict) else q
        print(time.strftime("%H:%M:%S"), st, f"{time.time() - t0:.0f}s", flush=True)
        if st == "done":
            c, a = req(f"{case}/validate/queue/{uuid}/attempt")
            print(json.dumps(a, indent=1)[:3000])
            if isinstance(a, dict):
                print(f"SCORE {a.get('score')} errors={len(a.get('errors') or [])} url={url}")
                report = {"case": case, "queue_id": uuid, "attempt": a, **timing(a)}
                if case == CASES["survival"] and a.get("score", 0) > 0 and "wall_seconds" in report:
                    report["approx_ms_per_tick"] = 1000 * report["wall_seconds"] / (a["score"] * 10)
                folder = Path(os.environ.get("NAC_REPORT_DIR", "operator-results"))
                folder.mkdir(parents=True, exist_ok=True)
                # Queue identifiers come from the portal; use only the filename component.
                path = folder / f"{case}-{Path(str(uuid)).name}.json"
                path.write_text(json.dumps(report, indent=2) + "\n")
                print(f"REPORT {path}")
                return report
            return
        if st in ("failed", "error", "cancelled"):
            print(json.dumps(q)[:2000])
            return
        max_wait = float(os.environ.get("NAC_MAX_WAIT_SECONDS", "3600" if case == CASES["survival"] else "1800"))
        if time.time() - t0 > max_wait:
            sys.exit(f"Polling timeout; attempt {uuid} may still be running. Do not queue another attempt.")


def history(case, n=15):
    _, s = req(f"{case}/status")
    for v in s["validations"][:n]:
        t = timing(v)
        suffix = f" wall={t['wall_seconds']:.1f}s" if t else ""
        if case == CASES["survival"] and t and v.get("score", 0) > 0:
            suffix += f" approx_ms/tick={1000*t['wall_seconds']/(v['score']*10):.1f}"
        print(f"{v['submitted_at'][5:16]}Z {v['score']:.4f} err={len(v['errors'])} {v['service_url']}{suffix}")


def leaderboard(case):
    d = json.load(urllib.request.urlopen(LB, timeout=30))
    rows = sorted([(t["scores"][case]["real"], t["team_name"]) for t in d if t["scores"].get(case)], reverse=True)
    for i, (sc, name) in enumerate(rows[:10]):
        print(f"{i + 1:2d}. {sc:.4f} {name}")
    print("Phillips:", [(i + 1, round(sc, 4)) for i, (sc, name) in enumerate(rows) if name == "Phillips"])


if __name__ == "__main__":
    cmd, case = sys.argv[1], CASES[sys.argv[2]]
    if cmd == "validate":
        validate(case, sys.argv[3])
    elif cmd == "history":
        history(case, int(sys.argv[3]) if len(sys.argv) > 3 else 15)
    elif cmd == "leaderboard":
        leaderboard(case)
