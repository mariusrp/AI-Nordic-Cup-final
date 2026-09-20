"""Per-run server-log statistics for frames <= 150 only (drone understanding lane, cycle 5).

Reads drone/server.py per-frame log lines ("fN idxI Lk (cx,cy) A ann -> Lk(cx,cy) | Tms ... dets=D tracks=K out=O ...")
and summarises what the PLATFORM run did in the frames the team may look at: frames answered, latency, level mix,
L2 looks, answered boxes, alive tracks, tracks spawned by frame 150. Lines for frames > 150 are skipped before parsing.
Usage: python srvlog_stats.py name=log [...]
"""
import re, sys, statistics as st
pat = re.compile(r' f(\d+) idx(\d+) L(\d) \((\d+),(\d+)\) (\d+) ann -> L(\d)\((\d+),(\d+)\) \| (\d+)ms.*?dets=(\d+) tracks=(\d+)(?: out=(\d+))?(?: capped=(\d+))?(?: spawned=(\d+))?')
for a in sys.argv[1:]:
    name, p = a.split('=', 1)
    rows = []
    for line in open(p, errors='ignore'):
        m = pat.search(line)
        if not m or int(m.group(1)) > 150:
            continue
        rows.append([int(x) if x is not None else -1 for x in m.groups()])
    if not rows:
        print(name, 'no frame lines'); continue
    fr = [r[0] for r in rows]; ms = sorted(r[9] for r in rows)
    lv = [r[2] for r in rows]
    print(f'{name:14s} frames {len(set(fr)):3d}/150 (max {max(fr)}) ms p50 {st.median(ms):5.0f} p90 {ms[int(.9 * len(ms))]:5.0f} max {ms[-1]:5.0f} '
          f'L0/L1/L2 {lv.count(0):3d}/{lv.count(1):3d}/{lv.count(2):3d} ann/frame {st.mean(r[5] for r in rows):5.1f} '
          f'tracks@150 {rows[-1][11]:3d} spawned@150 {rows[-1][14]:4d} capped {rows[-1][13]}')
