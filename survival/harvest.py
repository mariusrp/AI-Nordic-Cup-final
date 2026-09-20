#!/usr/bin/env python3
"""INCREMENTAL survival scorer (companion to the frozen evaluate.py).

Why it exists: evaluate.py uses Pool.map, which returns nothing until every seed is done, so a run that is
killed at 90% yields ZERO rows. On a contended machine (20 Sep: other agents pushed the Mac to load 190 on
12 cores and a game's wall time went from 88 s to ~900 s) two 128-seed runs were stopped at 28% and 33%
complete and produced no data at all.

This runs the SAME game loop (evaluate.one, imported from evaluate.py so the scorer stays frozen and shared)
but with ONE FRESH PROCESS PER SEED (maxtasksperchild=1, imap chunksize=1, seeds in order) and appends each
finished game to a JSONL immediately. Stop it whenever you like and the rows already written are a valid
sample; run several policies side by side and intersect the seeds for a paired comparison at whatever n the
clock allowed. The fresh process per seed also removes the Pool worker-history contamination that LESSONS
blames for part of the 16-seed noise.

Usage:
  UPSTREAM=... SDL_VIDEODRIVER=dummy python3 survival/harvest.py <policy.py> <seed_lo> <seed_hi> <out.jsonl> [procs]
Analyse (paired, restricted to the seeds every file has):
  python3 survival/harvest_stats.py <base.jsonl> <arm.jsonl> ...
"""
import importlib.util
import json
import multiprocessing as mp
import os
import sys

EV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "evaluate.py")
_spec = importlib.util.spec_from_file_location("ev", EV)
ev = importlib.util.module_from_spec(_spec)
sys.modules["ev"] = ev          # spawn workers unpickle ev.one by name
_spec.loader.exec_module(ev)


def main():
    if len(sys.argv) < 5:
        print(__doc__)
        raise SystemExit(2)
    pol, lo, hi, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    procs = int(sys.argv[5]) if len(sys.argv) > 5 else 4
    done = set()
    if os.path.exists(out):                      # resume: never re-run a seed already on disk
        for line in open(out):
            line = line.strip()
            if line:
                try:
                    done.add(json.loads(line)["seed"])
                except Exception:
                    pass
    seeds = [s for s in range(lo, hi) if s not in done]
    args = [(os.path.abspath(pol), s, 3000.0) for s in seeds]
    with mp.get_context("spawn").Pool(procs, maxtasksperchild=1) as pool:
        with open(out, "a", buffering=1) as fh:
            for r in pool.imap(ev.one, args, chunksize=1):
                fh.write(json.dumps(r) + "\n")
    print(f"HARVEST DONE {pol} n={len(seeds)} (skipped {len(done)} already on disk)")


if __name__ == "__main__":
    main()
