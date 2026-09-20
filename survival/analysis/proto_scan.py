#!/usr/bin/env python3
"""Understanding lane, cycle 3: MEASUREMENT PROTOTYPE of "scan-and-face" on BB2's bb_mpc (pred_mode='fde').
Analysis only: it is not a candidate policy, is never served and must not be used for keep/discard. It exists
to check whether the mechanism ACTS (kill_census.py: first-seen distance, late-kill share) and what it costs.

It changes ONE thing in bb_mpc: when the agent is on BB2's no-danger path (the returned turn is the forage turn,
i.e. no predicted chaser is being faced), the turn becomes SCAN_W * scan_dir rad/step:
  SCAN_MODE=idle : only when the forage move is <= 3 px (waiting at a tree / deferred / wandering in place)
  SCAN_MODE=all  : always on the no-danger path (also while walking; fruit beyond hearing may drop out of view)
Everything else (odometry, predator tracking, FDE response, spawning, claiming) is bb_mpc unchanged; the modified
turn is what bb_mpc's own odometry integrates, so its resting-predator detector stays exact.
bb_mpc.py is loaded from $BB_MPC or ../policies/bb_mpc.py (it lives on branch survival-bigbet-2).
"""
import importlib.util as _u, os as _o

_path = _o.environ.get("BB_MPC") or _o.path.join(_o.path.dirname(_o.path.dirname(_o.path.abspath(__file__))),
                                                  "policies", "bb_mpc.py")
_s = _u.spec_from_file_location("bb_mpc_scan", _path)
_m = _u.module_from_spec(_s); _s.loader.exec_module(_m)
W = float(_o.environ.get("SCAN_W", "0.9"))
MODE = _o.environ.get("SCAN_MODE", "all")
_Base = _m.make(pred_mode="fde")


class Hivemind(_Base):
    def _forage(self, *a, **k):
        r = super()._forage(*a, **k)
        self._ft, self._fmd = r[2], r[0]
        return r

    def _act_one(self, a, m, *rest):
        self._ft = None
        act, spawned = super()._act_one(a, m, *rest)
        if self._ft is not None and act["turn_angle"] == float(self._ft):
            if MODE == "all" or act["move_distance"] <= 3.0:
                act["turn_angle"] = W * m["scan_dir"]
                self.n_scan = getattr(self, "n_scan", 0) + 1
        self.n_act = getattr(self, "n_act", 0) + 1
        return act, spawned
