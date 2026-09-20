#!/usr/bin/env python3
"""Nordic AI Cup portal client. status / verify / validate ONLY.
The final evaluation endpoint is intentionally NOT implemented: Adrian submits the final himself."""
import json, os, sys, time, urllib.request

BASE = "https://cases.nordicaicup.com/api/v1/usecases"
CASES = {"survival": "survival-simulator", "drone": "drone-flyby", "medical": "medical-appointment"}
KEY = open(os.path.expanduser("~/.secrets/nac_api_key")).read().strip()


def req(path, method="GET", body=None, timeout=60):
    assert "evaluate" not in path, "final evaluation is forbidden from tooling"
    data = json.dumps(body).encode() if body is not None else None
    r = urllib.request.Request(f"{BASE}/{path}", data=data, method=method,
                               headers={"x-token": KEY, "Accept": "application/json", "Content-Type": "application/json", "User-Agent": "curl/8.5.0"})
    try:
        with urllib.request.urlopen(r, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read() or b"null")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()[:2000]


def main():
    cmd, case = sys.argv[1], CASES[sys.argv[2]]
    if cmd == "status":
        print(req(f"{case}/status"))
    elif cmd == "verify":
        print(req(f"{case}/verify", "POST", {"url": sys.argv[3]}, timeout=120))
    elif cmd == "validate":
        if case == "survival-simulator":
            rp = os.environ.get("CONFIRMED", "")
            ok = False
            try:
                rep = json.load(open(rp))
                ok = rep.get("verdict") == "CLEAR WIN" and rep.get("n", 0) >= 128 and time.time() - os.path.getmtime(rp) < 6 * 3600
            except Exception:
                pass
            if not ok and os.environ.get("REVALIDATE_CURRENT") != "1":
                sys.exit("VALIDATION BLOCKED (survival): validate only a clear local win. Run python3 /home/claude/nac/survival/confirm.py "
                         "(paired, >=128 fresh seeds, seeds 2000+) and re-run with CONFIRMED=/home/claude/handoff/confirm/<name>.json "
                         "whose verdict is CLEAR WIN (report < 6 h old). Deploy exactly that policy first.")
        code, r = req(f"{case}/validate/queue", "POST", {"url": sys.argv[3]})
        print(code, r, flush=True)
        uuid = (r.get("data") or {}).get("queued_attempt_uuid") if isinstance(r, dict) else None
        while uuid:
            time.sleep(15)
            c, q = req(f"{case}/validate/queue/{uuid}")
            st = (q.get("data") or {}).get("status") if isinstance(q, dict) else q
            print(time.strftime("%H:%M:%S"), st, flush=True)
            if st == "done":
                print(json.dumps(req(f"{case}/validate/queue/{uuid}/attempt"), indent=1)[:4000])
                break
    else:
        sys.exit("usage: nac.py status|verify|validate <survival|drone|medical> [url]")


if __name__ == "__main__":
    main()
