"""Measure exact belief cost omitted by the shared regret-scoring pass.

Myopic/greedy use the same belief constructor but need no two-period
expectimax. Omit that scoring pass, verify every action against the full
run, and add belief construction time to the original decision timings.
"""
import json
import sys
from pathlib import Path
import time
from unittest.mock import patch
from supplynet.game import GameConfig
from supplynet.lookahead import Oracle,run
from supplynet.world import Action

root=Path(sys.argv[1]) if len(sys.argv)>1 else Path('/nexus/Dev/OpenCog/bench/results/supplynet-stage5/review-fixes')
belief=Oracle.belief
samples=[]
def timed(model,*args):
    start=time.perf_counter()
    value=belief(model,*args)
    samples.append(time.perf_counter()-start)
    return value
def omit_scoring(model,*args,**kwargs):
    return {(Action(),Action()):0.0},{a:0.0 for a in model.first},{a:0 for a in model.first}

for backend in ('myopic','greedy'):
    for seed in range(1,5):
        samples.clear()
        with patch.object(Oracle,'belief',timed),patch.object(Oracle,'evaluate',omit_scoring):
            result=run(GameConfig(seed=seed,rounds=30,stage=5,plants=1),backend)
        original=json.loads((root/f'{backend}-s{seed}.json').read_text())
        assert result['actions']==original['actions']
        posterior=sum(samples)/len(samples)
        print(json.dumps({'type':'baseline-cost-profile','backend':backend,'seed':seed,'rounds':30,
                          'posterior_seconds':posterior,'scoring_seconds':original['decision_seconds'],
                          'amortized_seconds':original['amortized_seconds']+posterior,'actions_match':True}),flush=True)
