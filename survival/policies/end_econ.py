"""end_econ = lin_map (lineage6 + shared dead-reckoned world map) + ENDGAME ECONOMY.

Target: the MEAN and the RELIABILITY of a game (the final evaluation averages three games), not a lucky draw.
The failure it attacks: games end at 1000-1300 s with 3-5 agents standing at a tree that has just died, with
10-20 trees still alive and 30-45 uneaten fruit a median 320-350 px away.

Simulator facts this is built on (all read off the upstream code, not guessed):
  * a fruit spawns with energy 20 and grows 2 energy/s to 60, then keeps 60 until it ROTS at age 50 s
    (env: fruit.grow(2*dt), removed at age > 100 with age advancing 2/s). Eating a fruit the moment it appears
    collects 20; waiting until it is ripe collects 60. Standing still costs 0.1/step = 1 energy/s and is paid
    either way, so waiting next to a fruit you can already reach is a straight 3x on the harvest.
  * fruit spawns 20-60 px from the trunk of a tree with radius 20, so a camper that stands ON its tree keeps most
    of its fruit inside the hearing disc (everything within hearing_radius is perceived every step, cone or not).
  * a tree bears fruit from age 20 at 0.1 fruit/s (forest/grassland) and dies with per-step hazard
    ((age-50)/50)^2, i.e. almost surely between age 55 and 65. A tree in VIEW is by definition alive, so leaving
    a live tree is nearly always wrong; a tree that vanishes from view is the signal to move.

Mechanisms (all switchable, END_PARAMS='{"end_on":0}' = plain lin_map for the orchestrator's fallback):
  M1 RIPENESS HARVEST (ripe_on): the map records for every fruit entry whether it was seen BORN (created while a
     motionless agent had the spot inside its hearing disc), which makes its age - and therefore its energy and
     its rot deadline - exact. A fruit whose estimated energy is still below ripe_e is left alone as long as the
     agent can afford to wait for it (energy - 1.2 * seconds_to_ripe > ripe_floor); the agent stands and scans
     instead of walking over to collect 20 energy.
  M2 SITE CIRCUIT (circ_on): a camper whose tree has died has NO tree in view; instead of standing there it walks
     to the best remembered site in the shared map - a remembered fruit cluster valued at its energy ON ARRIVAL,
     or a remembered tree young enough to still be alive - one agent per site (tclaims). Crowded rich agents
     split off to a free remembered tree (disp_on) so two agents never share one tree's 0.1 fruit/s.
  M3 SEARCH LEGS (srch_on): with no tree in view and nothing worth walking to in the map, standing still is a
     certain death (income 0), so the agent commits to a straight leg at full walking speed, facing where it
     walks, turning when it runs into an edge. Only agents that can still afford it (srch_min_e) search.
  M4 CAMP ON THE TRUNK (camp_close): tree_stay 35 -> camp_d, which shortens every walk to a fruit and puts more
     fruit inside the hearing disc, which is what makes M1's born/age estimate exact.

MEASURED (seeds 9000-9015, frozen scorer, paired against a lin_map run of the same command, mean 1306.3):
  M1+M4 alone   1368.1  diff +61.8 se 86.8, 8/16 wins, median +194, games >= 1500 8/16 vs 4/16, but p25 -163
  M2+M3 alone   1266.0  diff -40.3 se 75.1, starvation +31% (406 vs 310) -> the walking half is the cost
  all four      1225.7  diff -80.5 se 56.5, 5/16 wins
So M2/M3/disp ship OFF (circ_on/srch_on/disp_on default 0): map-driven relocation is the fourth failure of that
idea in this lane and should be treated as closed. The defaults below are the M1+M4 configuration.
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lin_map  # noqa: E402
from bb_juke import wrap  # noqa: E402

ENDP = dict(
    end_on=1,
    # ---- M1 ripeness harvest
    ripe_on=1, ripe_e=56.0, ripe_floor=70.0, ripe_hard=60.0, defer_d=95.0, rot_marg=8.0, ripe_rate=1.2,
    # ---- M2 site circuit
    # circ/srch/disp measured NEGATIVE (see the header note) and are OFF by default; 1 re-enables them
    circ_on=0, circ_t=400.0, circ_max_d=620.0, circ_min_d=70.0, circ_arr=26.0, circ_gain=20.0, circ_tau=8.0,
    circ_min_e=28.0, circ_cool=10.0, circ_tmo=2.2, tree_life=45.0, tree_fresh=70.0, tree_val=150.0,
    clu_d=55.0, see_d=45.0,
    disp_on=0, disp_e=180.0, disp_n=2, disp_r=85.0,
    # ---- M3 search legs
    srch_on=0, srch_min_e=45.0, srch_leg=90, srch_cap=16.0, srch_t=300.0, srch_edge=70.0,
    # ---- M4 camp on the trunk
    camp_close=1, camp_d=14.0,
    # fruit rots at 50 s, not 80 (lin_map default) - a stale entry is a wasted trip
    fr_ttl=48.0,
)
if os.environ.get("END_PARAMS"):
    import json
    ENDP.update(json.loads(os.environ["END_PARAMS"]))

ROT = 50.0  # a fruit is removed at age 50 s (age advances 2/s, removed above 100)


class Hivemind(lin_map.Hivemind):
    def __init__(self, params=None):
        q = dict(ENDP)
        if params:
            q.update(params)
        super().__init__(q)
        if self.p["end_on"] and self.p["camp_close"]:
            self.p["tree_stay"] = self.p["camp_d"]
        self.tclaims = {}
        self.diag.update({"defer_steps": 0, "ripe_take": 0, "raw_take": 0, "born_ents": 0,
                          "circ_go": 0, "circ_arr": 0, "circ_abort": 0, "circ_steps": 0, "circ_px": 0.0,
                          "circ_tree": 0, "circ_fruit": 0, "disp_go": 0, "srch_steps": 0, "notree": 0})
        self._end_diag = bool(os.environ.get("END_DIAG"))

    # ------------------------------------------------------------------ map: fruit identity + birth time
    def _map_update(self, agents):
        moved = {}
        for a in agents:
            m = self.mem.get(a["agent_id"])
            pr = m.get("pred") if m else None
            moved[a["agent_id"]] = pr[2] if pr else 0.0
        n0 = self.next_ent
        super()._map_update(agents)
        if not (self.p["end_on"] and self.p["ripe_on"]):
            return
        t = (self.step + 1) * 0.1
        for a in agents:
            aid = a["agent_id"]
            m = self.mem.get(aid)
            if m is None:
                continue
            F, X, Y, H = m["F"], m["X"], m["Y"], m["H"]
            static = moved.get(aid, 99.0) < 1.0
            dturn = abs(wrap(H - m["Hp"])) if "Hp" in m else 9.0
            m["Hp"] = H
            hear = a["hearing_radius"]
            half = a["vision_angle"] / 2.0
            vr = a["vision_range"]
            fmap = {}
            for o in a["observations"]:
                if o["type"] != "Fruit":
                    continue
                ang = H + o["angle"]
                gx = X + o["distance"] * math.cos(ang)
                gy = Y + o["distance"] * math.sin(ang)
                best, bd = None, 36.0
                for i in self._near(F, gx, gy, 8.0):
                    e = self.ents[i]
                    if e["k"] != "F":
                        continue
                    d2 = (e["x"] - gx) ** 2 + (e["y"] - gy) ** 2
                    if d2 <= bd:
                        best, bd = i, d2
                if best is None:
                    continue
                fmap[id(o)] = best
                e = self.ents[best]
                # the spot was certainly watched last step too -> the entry appearing now means the fruit SPAWNED
                watched = o["distance"] < hear - 4.0 or (
                    o["distance"] < vr - 15.0 and abs(o["angle"]) + dturn + 0.15 < half)
                if best >= n0 and static and watched:
                    if not e.get("born"):
                        e["born"] = True
                        e["tb"] = t
                        self.diag["born_ents"] += 1
                elif "born" not in e:
                    e["born"] = False
            m["fmap"] = fmap
        # drop claims of agents that died
        if self.tclaims:
            live = {a["agent_id"] for a in agents}
            for i, aid in list(self.tclaims.items()):
                if aid not in live or i not in self.ents:
                    self.tclaims.pop(i, None)

    # ------------------------------------------------------------------ M1: which visible fruit to take now
    def _split_fruits(self, a, m, fruits):
        p = self.p
        if not fruits or not p["ripe_on"]:
            return fruits, 0
        t = self.step * 0.1
        energy = a["energy"]
        fmap = m.get("fmap") or {}
        take, defer = [], 0
        for o in fruits:
            i = fmap.get(id(o))
            e = self.ents.get(i) if i is not None else None
            if e is None or not e.get("born") or o["distance"] > p["defer_d"]:
                take.append(o); continue
            age = t - e.get("tb", e["t0"])
            est = min(60.0, 20.0 + 2.0 * age)
            if est >= p["ripe_e"] or age > ROT - p["rot_marg"]:
                take.append(o); continue
            wait = (p["ripe_e"] - est) * 0.5
            if energy < p["ripe_hard"] or energy - p["ripe_rate"] * wait < p["ripe_floor"]:
                take.append(o); continue
            defer += 1
        return take, defer

    # ------------------------------------------------------------------ M2: remembered sites
    def _drop_site(self, aid, m, why):
        r = m.pop("troute", None)
        if r is not None:
            if self.tclaims.get(r["i"]) == aid:
                self.tclaims.pop(r["i"], None)
            self.diag["circ_" + why] = self.diag.get("circ_" + why, 0) + 1

    def _pick_site(self, a, m, t):
        p = self.p
        aid = a["agent_id"]
        F, X, Y = m["F"], m["X"], m["Y"]
        spd = max(1.0, min(a["speed"], p["srch_cap"]))
        best, bsc = None, 0.0
        for i in self._near(F, X, Y, p["circ_max_d"]):
            e = self.ents.get(i)
            if e is None:
                continue
            own = self.tclaims.get(i)
            if own is not None and own != aid:
                continue
            dx, dy = e["x"] - X, e["y"] - Y
            D = math.hypot(dx, dy)
            if D > p["circ_max_d"] or D < p["circ_min_d"]:
                continue
            travel = D / spd * 0.1                      # seconds of walking
            cost = 0.05 * D + 0.1 * travel * 10.0       # move charge + base drain
            if e["k"] == "F":
                age = t - e.get("tb", e["t0"])
                if age + travel > ROT - 2.0:
                    continue
                val = 0.0
                for j in self._near(F, e["x"], e["y"], p["clu_d"]):
                    ej = self.ents.get(j)
                    if ej is None or ej["k"] != "F":
                        continue
                    if (ej["x"] - e["x"]) ** 2 + (ej["y"] - e["y"]) ** 2 > p["clu_d"] ** 2:
                        continue
                    aj = t - ej.get("tb", ej["t0"])
                    if aj + travel > ROT - 2.0:
                        continue
                    val += min(60.0, 20.0 + 2.0 * (aj + travel)) * max(0.15, 1.0 - (t - ej["t1"]) / 30.0)
                kind = "fruit"
            else:
                age = t - e["t0"]
                if age > p["tree_life"] or t - e["t1"] > p["tree_fresh"]:
                    continue
                val = p["tree_val"] * max(0.15, 1.0 - age / p["tree_life"])
                kind = "tree"
            gain = val - cost
            sc = gain / (travel + p["circ_tau"])
            if gain >= p["circ_gain"] and sc > bsc:
                best, bsc = (i, e, D, kind), sc
        if best is None:
            return None
        i, e, D, kind = best
        self.tclaims[i] = aid
        self.diag["circ_" + kind] += 1
        return {"i": i, "x": e["x"], "y": e["y"], "t0": t,
                "T": p["circ_tmo"] * D / max(1.0, min(a["speed"], p["srch_cap"])) * 0.1 + 6.0}

    def _walk_to(self, a, m, dx, dy, edges):
        p = self.p
        D = math.hypot(dx, dy)
        rel = wrap(math.atan2(dy, dx) - m["H"])
        md = min(min(a["speed"], p["srch_cap"]), D)
        turn = max(-p["route_turn"], min(p["route_turn"], rel))
        mdir = self._avoid_edges(rel, edges)
        self.diag["circ_steps"] += 1
        self.diag["circ_px"] += md
        m["wander"] = 0
        m["srch"] = 0
        m["t"] = m.get("t", 0) + 1
        return md, mdir, turn

    # ------------------------------------------------------------------ M3: search leg
    def _search(self, a, m, edges):
        p = self.p
        aid = a["agent_id"]
        psi = m.get("srch_psi")
        if psi is None or m.get("srch", 0) <= 0:
            base = psi if psi is not None else m["H"]
            k = (self.step + aid * 7) % 5
            sgn = 1 if (aid + m.get("legs", 0)) % 2 else -1
            psi = wrap(base + (1.7 + 0.22 * k) * sgn)
            m["legs"] = m.get("legs", 0) + 1
            m["srch"] = int(p["srch_leg"])
        rel = wrap(psi - m["H"])
        if self._edge_room(rel, edges) < p["srch_edge"]:
            psi = wrap(psi + 1.6)
            rel = wrap(psi - m["H"])
            m["srch"] = int(p["srch_leg"])
        m["srch_psi"] = psi
        m["srch"] = m.get("srch", 0) - 1
        mdir = self._avoid_edges(rel, edges)
        turn = max(-0.4, min(0.4, rel))
        self.diag["srch_steps"] += 1
        m["wander"] = 0
        m["t"] = m.get("t", 0) + 1
        return min(a["speed"], p["srch_cap"]), mdir, turn

    # ------------------------------------------------------------------ forager
    def _forage(self, a, m, fruits, trees, sibs, edges, stats):
        p = self.p
        if not p["end_on"]:
            return super()._forage(a, m, fruits, trees, sibs, edges, stats)
        aid = a["agent_id"]
        t = self.step * 0.1
        take, defer = self._split_fruits(a, m, fruits)
        # 1. every visible fruit is still growing and affordable to wait for: hold the site
        if defer and not take:
            self._drop_site(aid, m, "food")
            self.diag["defer_steps"] += 1
            m["wander"] = 0
            m["srch"] = 0
            m["t"] = m.get("t", 0) + 1
            return 0.0, 0.0, p["scan_rate"] * m["scan_dir"]
        if take:
            self.diag["ripe_take"] += 1
            self._drop_site(aid, m, "food")
            m["srch"] = 0
            return super()._forage(a, m, take, trees, sibs, edges, stats)
        # 2. no fruit worth taking in view
        if m.get("elder") or not p["circ_on"] or t < p["circ_t"]:
            return super()._forage(a, m, take, trees, sibs, edges, stats)
        near_tree = any(o["distance"] < p["tree_stay"] + p["see_d"] for o in trees)
        r = m.get("troute")
        if r is not None:
            dx, dy = r["x"] - m["X"], r["y"] - m["Y"]
            D = math.hypot(dx, dy)
            if D < p["circ_arr"] or (near_tree and D < p["circ_min_d"]):
                self._drop_site(aid, m, "arr")
                return super()._forage(a, m, take, trees, sibs, edges, stats)
            if t - r["t0"] > r["T"] or a["energy"] < p["circ_min_e"]:
                self._drop_site(aid, m, "abort")
                m["circ_cool"] = t + p["circ_cool"]
            else:
                return self._walk_to(a, m, dx, dy, edges)
        # 3. decide whether to leave for a remembered site
        want = False
        if not near_tree:
            self.diag["notree"] += 1
            want = True
        elif p["disp_on"] and a["energy"] > p["disp_e"] and \
                sum(1 for s in sibs if s["distance"] < p["disp_r"]) >= p["disp_n"]:
            want = True
            self.diag["disp_go"] += 1
        if want and t >= m.get("circ_cool", 0.0):
            r = self._pick_site(a, m, t)
            if r is not None:
                m["troute"] = r
                self.diag["circ_go"] += 1
                self._release(aid, m)   # give up lin_map's own fruit route
                return self._walk_to(a, m, r["x"] - m["X"], r["y"] - m["Y"], edges)
            if p["srch_on"] and not near_tree and t >= p["srch_t"] and a["energy"] > p["srch_min_e"]:
                self._release(aid, m)
                return self._search(a, m, edges)
        return super()._forage(a, m, take, trees, sibs, edges, stats)

    def act(self, agents, n_total=None):
        out = super().act(agents, n_total)
        if self._end_diag and self.step % 2000 == 0:
            print("ENDDIAG step=%d n=%d " % (self.step, len(agents))
                  + " ".join("%s=%.0f" % (k, v) for k, v in sorted(self.diag.items())),
                  file=sys.stderr, flush=True)
        return out


def make(**kw):
    class H(Hivemind):
        def __init__(self, params=None):
            q = dict(kw)
            if params:
                q.update(params)
            super().__init__(q)
    return H
