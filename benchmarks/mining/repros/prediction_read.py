"""Compare predictive reads with the miner's near-deterministic learned truth.

The data contain only Y(t) = F0(t-1); the adapter supplies no hypothesis.
Run using the shared Python, capped and the environment in docs/mining_bench.md.
"""
import json
from miningbench.world import Config, World
from miningbench.petta import Petta

w = World(Config(samples=64, kinds=('single',)))
a = Petta(w.features, w.domains, budget=40)
for s in w.stream():
    a.predict(s['history'])
    a.learn(s['history'], s['y'])
print(json.dumps({'verdicts': a.raw}))
for truth, period in ((1, 100), (0, 101)):
    row = {(f'T0_single_F{i}', ()): truth if i == 0 else 0 for i in range(4)}
    a.add_inputs(row, period)
    for budget in (40, 400, 2000):
        rows = a.h.query_many_refs([f'(: $p (T0_single p{period}) $tv)'], steps=budget)
        print(json.dumps({'input':truth,'budget':budget,'answers':[
            {'proof':r.proof,'expanded_proof':a.h.expand_proof(r.proof),'statement':r.statement,'tv':r.tv} for r in rows[0]]}))
