"""Trait trajectory of a policy: prints herd size, median/max speed, hearing, max_energy every 100 s.
Usage: UPSTREAM=... python survival/analysis_traits.py policies/ratchet2.py SEED [MAX_T]"""
import importlib.util, os, statistics, sys
sys.path.insert(0, os.environ.get("UPSTREAM", "/home/claude/Nordic-AI-Cup-2026") + "/survival-simulator")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
import io, contextlib
from src.core import SimulationCore
from src.utils.DTOs import ActionRequest
from src.elements.environment import Environment
deaths = {"predator": 0, "starve": 0, "old": 0}
_orig = Environment.kill_agent
def _kill(self, agent):
    if agent in self.agents:
        deaths["predator" if agent.energy > 0 else ("old" if agent.age > agent.max_age else "starve")] += 1
    _orig(self, agent)
Environment.kill_agent = _kill
spec = importlib.util.spec_from_file_location("pol", sys.argv[1]); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
seed = int(sys.argv[2]); max_t = float(sys.argv[3]) if len(sys.argv) > 3 else 3000
with contextlib.redirect_stdout(io.StringIO()):
    sim = SimulationCore(seed=seed)
hm = mod.Hivemind(); actions = []; nxt = 0
while True:
    with contextlib.redirect_stdout(io.StringIO()):
        state = sim.step(actions)
    obs = [o for o in state["observations"] if o is not None]
    acts = hm.act(obs); actions = [(a["agent_id"], ActionRequest(**a)) for a in acts]
    if sim.env.time >= nxt:
        ag = sim.env.agents
        if ag:
            sp = [a.speed for a in ag]; he = [a.hearing_radius for a in ag]; me = [a.max_energy for a in ag]; vi = [a.vision_radius for a in ag]
            print(f"t={sim.env.time:6.0f} n={len(ag):2d} score={state['score']:7.1f} preds={len(sim.env.predators):2d} "
                  f"speed med/max {statistics.median(sp):5.1f}/{max(sp):5.1f} hear {statistics.median(he):5.1f}/{max(he):5.1f} "
                  f"maxE {statistics.median(me):5.0f}/{min(me):4.0f}-{max(me):4.0f} vis {statistics.median(vi):4.0f} deaths p/s/o {deaths['predator']}/{deaths['starve']}/{deaths['old']}", flush=True)
        nxt += 100
    if state["num_agents"] == 0 or sim.env.time > max_t:
        print(f"END t={sim.env.time:.0f} score={state['score']:.1f}"); break
