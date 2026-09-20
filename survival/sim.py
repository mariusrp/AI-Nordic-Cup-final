import sys, os, time, random, math
sys.path.insert(0, os.environ.get("UPSTREAM", "/home/claude/Nordic-AI-Cup-2026") + "/survival-simulator")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
from src.core import SimulationCore
from src.utils.DTOs import ActionRequest
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from policy import Hivemind

def run(seed, params=None, max_time=3000, verbose=False):
    sim = SimulationCore(seed=seed)
    hm = Hivemind(params)
    actions = []
    stats = {"eaten": 0, "max_agents": 0}
    while True:
        prev = len(sim.env.agents)
        state = sim.step(actions)
        obs = [o for o in state["observations"] if o is not None]
        stats["max_agents"] = max(stats["max_agents"], len(obs))
        acts = hm.act(obs)
        actions = [(x["agent_id"], ActionRequest(**x)) for x in acts]
        if verbose and int(sim.env.time * 10) % 1000 == 0:
            print(f"t={sim.env.time:.0f} n={state['num_agents']} score={state['score']:.1f} preds={len(sim.env.predators)} trees={len(sim.env.trees)} fruits={len(sim.env.fruits)}", flush=True)
        if state["num_agents"] == 0 or sim.env.time > max_time:
            break
    return {"score": state["score"], "time": sim.env.time, **stats}

if __name__ == "__main__":
    seeds = [int(s) for s in sys.argv[1:]] or [1]
    for s in seeds:
        t = time.time()
        r = run(s, verbose=True)
        print(s, r, f"{time.time()-t:.0f}s wall", flush=True)
