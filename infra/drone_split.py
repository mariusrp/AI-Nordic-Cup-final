"""Keep validation-flight frames >= CUT unseen: move them (and their metadata lines) to the auditor-only holdout.
Usage: drone_split.py <root_dir_with_sequence_dirs> [--loop]"""
import os, re, sys, json, shutil, time
CUT = 151
root = sys.argv[1]
dest = "/workspace/.holdout/drone_seen_heldout"
os.makedirs(dest, exist_ok=True); os.chmod("/workspace/.holdout", 0o700)
def once():
    moved = 0
    for seq in os.listdir(root):
        d = os.path.join(root, seq)
        if not os.path.isdir(d):
            continue
        for f in os.listdir(d):
            m = re.match(r"(\d{6})_L\d", f)
            if m and int(m.group(1)) >= CUT:
                os.makedirs(os.path.join(dest, seq), exist_ok=True)
                shutil.move(os.path.join(d, f), os.path.join(dest, seq, f)); moved += 1
            elif f.endswith((".jsonl", ".json")) and not f.startswith("."):
                p = os.path.join(d, f); keep, drop = [], []
                for line in open(p):
                    try:
                        (drop if json.loads(line).get("frame", 0) >= CUT else keep).append(line)
                    except Exception:
                        keep.append(line)
                if drop:
                    os.makedirs(os.path.join(dest, seq), exist_ok=True)
                    open(os.path.join(dest, seq, f), "a").writelines(drop)
                    open(p, "w").writelines(keep); moved += len(drop)
    return moved
if "--loop" in sys.argv:
    while True:
        n = once()
        if n: print(time.strftime("%H:%M"), "moved", n, flush=True)
        time.sleep(30)
else:
    print("moved", once())
