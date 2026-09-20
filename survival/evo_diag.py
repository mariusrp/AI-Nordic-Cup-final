"""One game with rule-act counters and a per-100 s trace (n, births, deaths by cause in the window, eats, mean eaten energy,
trees, predators). Usage: UPSTREAM=<dir holding survival-simulator> python survival/evo_diag.py policies/evo_x.py SEED [MAX_T]"""
import importlib.util, io, contextlib, os, sys
sys.path.insert(0, os.environ.get("UPSTREAM", "/home/claude/Nordic-AI-Cup-2026") + "/survival-simulator")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
from src.core import SimulationCore
from src.utils.DTOs import ActionRequest
from src.elements.environment import Environment
deaths = {"predator": 0, "starve": 0, "old": 0}
births = [0]; eats = [0, 0.0]
_kill = Environment.kill_agent
def kill(self, agent):
    if agent in self.agents:
        deaths["predator" if agent.energy > 0 else ("old" if agent.age > agent.max_age else "starve")] += 1
    _kill(self, agent)
Environment.kill_agent = kill
_spawn = Environment.spawn_agent
def spawn(self, *a, **k):
    if k.get("parent") is not None or (len(a) > 2 and a[2] is not None):
        births[0] += 1
    return _spawn(self, *a, **k)
Environment.spawn_agent = spawn
_rm = Environment.remove_fruit
def rm(self, fruit):
    # remove_fruit is called for eaten fruit and rotten fruit; eaten ones are those with an agent touching
    for ag in self.agents:
        if ((ag.x - fruit.x) ** 2 + (ag.y - fruit.y) ** 2) ** 0.5 < ag.size + fruit.radius:
            eats[0] += 1; eats[1] += fruit.energy; break
    _rm(self, fruit)
Environment.remove_fruit = rm
spec = importlib.util.spec_from_file_location("pol", sys.argv[1]); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
seed = int(sys.argv[2]); max_t = float(sys.argv[3]) if len(sys.argv) > 3 else 3000
with contextlib.redirect_stdout(io.StringIO()):
    sim = SimulationCore(seed=seed)
hm = mod.Hivemind(); actions = []; nxt = 100
prev = dict(deaths); pb = 0; pe = [0, 0.0]
print("   t   n births  d_pred d_starve d_old  eats meanE trees preds score")
while True:
    with contextlib.redirect_stdout(io.StringIO()):
        state = sim.step(actions)
    obs = [o for o in state["observations"] if o is not None]
    acts = hm.act(obs); actions = [(a["agent_id"], ActionRequest(**a)) for a in acts]
    if sim.env.time >= nxt or state["num_agents"] == 0:
        ne = eats[0] - pe[0]; me = (eats[1] - pe[1]) / max(1, ne)
        print(f"{sim.env.time:5.0f} {len(sim.env.agents):3d} {births[0]-pb:6d}  {deaths['predator']-prev['predator']:6d} {deaths['starve']-prev['starve']:8d} {deaths['old']-prev['old']:5d} {ne:5d} {me:5.1f} {len(sim.env.trees):5d} {len(sim.env.predators):5d} {state['score']:7.1f}")
        prev = dict(deaths); pb = births[0]; pe = list(eats); nxt += 100
    if state["num_agents"] == 0 or sim.env.time > max_t:
        break
print(f"END t={sim.env.time:.0f} score={state['score']:.1f} deaths={deaths} births={births[0]} eats={eats[0]} meanE={eats[1]/max(1,eats[0]):.1f}")
d = getattr(hm, "diag", {})
print("DIAG " + " ".join(f"{k}={v:.0f}" if isinstance(v, float) else f"{k}={v}" for k, v in d.items()))
