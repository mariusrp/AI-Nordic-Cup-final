#!/usr/bin/env python3
"""Understanding lane, cycle 3: MEASUREMENT PROTOTYPE (analysis only, never a candidate) for "break contact with
sleeping predators". It is bb_mpc (pred_mode='fde') with ONE parameter changed: sleep_margin (default 110 px), the
distance inside which an agent that sees a resting predator walks away from it instead of foraging. Question: does
the share of kills made within 6 s of the killer waking (prechase.py 'woke') fall, and what does it cost in starvation?
Env: SLEEP_MARGIN (px). bb_mpc.py is loaded from $BB_MPC or ../policies/bb_mpc.py.
"""
import importlib.util as _u, os as _o

_path = _o.environ.get("BB_MPC") or _o.path.join(_o.path.dirname(_o.path.dirname(_o.path.abspath(__file__))),
                                                  "policies", "bb_mpc.py")
_s = _u.spec_from_file_location("bb_mpc_sleep", _path)
_m = _u.module_from_spec(_s); _s.loader.exec_module(_m)
Hivemind = _m.make(pred_mode="fde", sleep_margin=float(_o.environ.get("SLEEP_MARGIN", "250")))
