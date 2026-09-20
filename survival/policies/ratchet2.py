"""ratchet2 (n3 session, Sat 19 Sep): speed-breeding graft on bb_juke.

Simulator facts (upstream environment.py:501-518, predator.py): walking costs 0.05/px whatever the speed trait,
a predator sprints at 15, and the sprint lock-out only caps a move at agent.speed, so an agent with
speed >= 15.1 can never be caught while moving away. Traits mutate on spawn with p=0.1 each by U(0.5,1.5),
capped at speed 20 / sprint 40 / max_energy 1000 / hearing 100 / vision 400 / cone pi/2, no lower bound,
so unselected lineages drift down (geometric mean 0.956 per mutation).

Graft (everything else is bb_juke):
  * breeding eligibility (the 'may' median gate in _act_one) becomes a ratchet: only agents whose fitness is
    within `elite_margin` of the herd maximum, or in the top `elite_k`, may spawn (small herds excepted as in
    bb_juke). fitness = speed (dominant) + small hearing/vision terms; degraded lineages (speed < min_speed,
    max_energy <= min_max_e) are never bred from.
  * foraging moves are capped at `walk_cap` px/step (default 10 = the base speed) so a fast lineage does not
    pay 1.0/step wandering instead of 0.5; flee/escape moves keep the real trait speed.
Params (make(**kw) or Hivemind(params)): elite_margin, elite_k, min_speed, min_max_e, walk_cap, w_hear, w_vis.
"""
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bb_juke  # noqa: E402

R2 = dict(elite_margin=0.5, elite_k=3, min_speed=8.0, min_max_e=150.0, walk_cap=10.0, w_hear=0.02, w_vis=0.01, w_maxe=1.0,
          max_e_lo=0.0, max_e_hi=1e9,
          # close bb_juke's breeding leaks: no 'old spare' births, no free-for-all below small_herd_n, cheaper elite births
          old_spare_energy=1e9, small_herd_n=2, spawn_energy=180.0, spawn_energy_old=140.0,
          # late-game herd cap: once the herd is fast (median speed >= fast_speed) or t >= herd_late_t, cap the herd
          # at max_herd_late (score = time, and a predator-proof herd only needs enough agents to keep breeding)
          herd_late_t=1e9, max_herd_late=12, fast_speed=15.1)
if os.environ.get("R2_PARAMS"):  # quick arms: R2_PARAMS='{"walk_cap": 1e9}'
    import json
    R2.update(json.loads(os.environ["R2_PARAMS"]))


def r2_fitness(a, p):
    """Speed-first fitness. Returns -inf for lineages that must not breed."""
    sp = a["speed"]
    me = a["max_energy"]
    if sp < p["min_speed"] or me <= p["min_max_e"]:
        return float("-inf")
    f = sp + p["w_hear"] * a["hearing_radius"] + p["w_vis"] * a["vision_range"] / 4.0 + p["w_maxe"] * me / 500.0
    if not (p["max_e_lo"] <= me <= p["max_e_hi"]):
        f -= 2.0
    return f


class Hivemind(bb_juke.Hivemind):
    def __init__(self, params=None):
        q = dict(R2)
        if params:
            q.update(params)
        super().__init__(q)
        self._max_herd0 = self.p["max_herd"]

    def act(self, agents, n_total=None):
        p = self.p
        n = len(agents)
        self.step += 1
        fits = {a["agent_id"]: r2_fitness(a, p) for a in agents}
        finite = [v for v in fits.values() if v != float("-inf")]
        if finite:
            top = max(finite)
            kth = sorted(finite, reverse=True)[min(p["elite_k"], len(finite)) - 1]
            med = min(top - p["elite_margin"], kth)  # eligible: fit >= med
        else:
            med = 0.0
        stats = {a["agent_id"]: (a["energy"], a["max_energy"], a["age"]) for a in agents}
        if p["herd_late_t"] < 1e8 or p["max_herd_late"] < self._max_herd0:
            med_speed = statistics.median(a["speed"] for a in agents) if agents else 0.0
            late = self.step * 0.1 >= p["herd_late_t"] or med_speed >= p["fast_speed"]
            p["max_herd"] = self._max_herd0 if not late else min(self._max_herd0, p["max_herd_late"])
        alive = set()
        cur_by = {}
        fde = p["pred_mode"] == "fde"
        for a in agents:
            aid = a["agent_id"]
            alive.add(aid)
            m = self.mem.setdefault(aid, {"scan_dir": 1 if aid % 2 else -1, "wander": 0, "X": 0.0, "Y": 0.0, "H": 0.0,
                                          "tracks": []})
            cur_by[aid] = self._track(a, m) if fde else []
        self._last_cur = cur_by
        actions = []
        spawned = 0
        for a in agents:
            aid = a["agent_id"]
            m = self.mem[aid]
            act, spawned = self._act_one(a, m, n, spawned, fits, med, stats, cur_by, fde)
            if fde:
                self._odom(a, m, act["move_distance"], act["move_direction"], act["turn_angle"])
            actions.append(act)
        for k in list(self.mem):
            if k not in alive:
                del self.mem[k]
        self.diag["agent_steps"] += n
        return actions

    def _forage(self, a, m, fruits, trees, sibs, edges, stats):
        cap = self.p["walk_cap"]
        if a["speed"] > cap:
            a = dict(a)
            a["speed"] = cap
        return super()._forage(a, m, fruits, trees, sibs, edges, stats)


def make(**kw):
    class H(Hivemind):
        def __init__(self, params=None):
            q = dict(kw)
            if params:
                q.update(params)
            super().__init__(q)
    return H
