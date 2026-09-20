import sys, time, collections
import sim
from src.elements.environment import Environment
deaths = []
_orig = Environment.kill_agent
def kill(self, agent):
    cause = "predator" if agent.energy > 0 else ("old" if agent.age > agent.max_age else "starve")
    deaths.append((round(self.time), cause, round(agent.age), round(agent.max_age), round(agent.energy)))
    _orig(self, agent)
Environment.kill_agent = kill
seed = int(sys.argv[1]); mt = float(sys.argv[2]) if len(sys.argv) > 2 else 3000
r = sim.run(seed, max_time=mt)
print(r)
c = collections.Counter(d[1] for d in deaths); print(c)
for bucket in range(0, int(r["time"]) + 300, 300):
    cc = collections.Counter(d[1] for d in deaths if bucket <= d[0] < bucket + 300)
    print(bucket, dict(cc))
print("old ages at death:", sorted(d[2] for d in deaths if d[1]=="old")[:5], "...")
print("starve ages:", sorted(d[2] for d in deaths if d[1]=="starve")[:30])
