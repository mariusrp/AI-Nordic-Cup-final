"""ratchet4 (n3): ratchet3 + a LEAN late herd. Once the herd is fast (median speed >= fast_speed) the herd cap drops
to max_herd_late (default 4) so a predator-proof herd stops eating the shrinking fruit supply faster than it can
reproduce; small-herd breeding protection stays at small_herd_n=late_small_herd (default 3) in the ratchet phase.
Params via R2_PARAMS as before (max_herd_late, late_small_herd, plus ratchet3/ratchet2 params)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ratchet3  # noqa: E402

R4 = dict(max_herd_late=4, late_small_herd=3)


class Hivemind(ratchet3.Hivemind):
    def __init__(self, params=None):
        q = dict(R4)
        if params:
            q.update(params)
        super().__init__(q)

    def act(self, agents, n_total=None):
        out = super().act(agents, n_total)
        if self._phase == 1 and self.p["small_herd_n"] != self.p["late_small_herd"]:
            self.p["small_herd_n"] = self.p["late_small_herd"]
        return out


def make(**kw):
    class H(Hivemind):
        def __init__(self, params=None):
            q = dict(kw)
            if params:
                q.update(params)
            super().__init__(q)
    return H
