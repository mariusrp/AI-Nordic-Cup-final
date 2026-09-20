"""ratchet3 (n3): ratchet2 with PHASED breeding bars.

train32 of ratchet2 was bimodal: +300..+730 when a fast lineage appears early, -400..-670 when the changed
early-game bars (spawn bar 180, no old-spare births, small_herd 2) collapse the herd before speed matters
(seed 1014 wipes out at 480 s). So: run bb_juke's proven breeding (bars 260/140, small_herd 4, old-spare
180) while the herd's top speed is below `mutant_speed`; once a faster mutant exists switch to the ratchet
bars (elite-only, bar `elite_bar`, no old-spare, small_herd 2). Optionally raise the herd cap early
(`early_herd` until t < `early_t`) for more mutation draws while food is plentiful.
Params via R2_PARAMS as in ratchet2 (mutant_speed, elite_bar, early_herd, early_t, plus ratchet2's).
"""
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bb_juke  # noqa: E402
import ratchet2  # noqa: E402

R3 = dict(mutant_speed=11.5, elite_bar=180.0, early_herd=12, early_t=600.0,
          # start from bb_juke's bars; act() switches them per step
          spawn_energy=260.0, spawn_energy_old=140.0, small_herd_n=4, old_spare_energy=180.0)


class Hivemind(ratchet2.Hivemind):
    def __init__(self, params=None):
        q = dict(R3)
        if params:
            q.update(params)
        super().__init__(q)
        self._phase = 0

    def act(self, agents, n_total=None):
        p = self.p
        if agents:
            top = max(a["speed"] for a in agents)
            t = self.step * 0.1
            if top >= p["mutant_speed"] and self._phase == 0:
                self._phase = 1
                p["spawn_energy"] = p["elite_bar"]
                p["small_herd_n"] = 2
                p["old_spare_energy"] = 1e9
            # early herd cap (more births = more mutation draws) while food is plentiful and nobody is fast yet
            if self._phase == 0:
                self._max_herd0 = p["early_herd"] if t < p["early_t"] else bb_juke.DEFAULT["max_herd"]
                p["max_herd"] = self._max_herd0
            else:
                self._max_herd0 = bb_juke.DEFAULT["max_herd"]
        return super().act(agents, n_total)


def make(**kw):
    class H(Hivemind):
        def __init__(self, params=None):
            q = dict(kw)
            if params:
                q.update(params)
            super().__init__(q)
    return H
