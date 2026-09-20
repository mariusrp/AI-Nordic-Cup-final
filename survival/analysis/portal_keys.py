#!/usr/bin/env python3
"""Understanding lane, cycle 4: MEASUREMENT WRAPPER (analysis only, never a candidate).

Question: what does bb_mpc lose if the portal's observations lack the keys it reads with silent defaults?
bb_mpc uses o.get("rel_dir", 0.0) for predator headings (rest detection, 'does it perceive me') and
sibling poses, and it builds its sibling list from Agent observations that carry "id" (claiming, crowding,
'is a sibling a closer target'). This wrapper is bb_mpc with those keys REMOVED from every observation
before act(), to price the risk. Env STRIP_KEYS: comma list (default "id,rel_dir").
bb_mpc.py is loaded from $BB_MPC or ../policies/bb_mpc.py.
Run with the frozen scorer: STRIP_KEYS=id,rel_dir python3 survival/evaluate.py survival/analysis/portal_keys.py --seeds ...
"""
import importlib.util as _u, os as _o

_path = _o.environ.get("BB_MPC") or _o.path.join(_o.path.dirname(_o.path.dirname(_o.path.abspath(__file__))),
                                                  "policies", "bb_mpc.py")
_s = _u.spec_from_file_location("bb_mpc_keys", _path)
_m = _u.module_from_spec(_s); _s.loader.exec_module(_m)
STRIP = {k for k in _o.environ.get("STRIP_KEYS", "id,rel_dir").split(",") if k}


class Hivemind(_m.Hivemind):
    def act(self, agents, n_total=None):
        for a in agents:
            a["observations"] = [{k: v for k, v in o.items() if k not in STRIP} for o in a.get("observations", [])]
        return super().act(agents)
