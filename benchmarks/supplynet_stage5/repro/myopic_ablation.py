"""Strict immediate-reward ablation: remove the legacy stage 4 proxies.
The headline myopic comparison retains those proxies; this checks that its
loss is not only a proxy-calibration effect. No production code is changed.
"""
import json
from unittest.mock import patch
from supplynet.game import GameConfig
from supplynet.lookahead import Oracle,run


def immediate(model,weights,decision):
    return {a:float(weights@model.reward(model.current,a,decision.phase,decision.problem.inputs,decision.bank)) for a in model.first}


with patch.object(Oracle,'myopic',immediate):
    for seed in range(1,5):
        r=run(GameConfig(seed=seed,rounds=30,stage=5,plants=1),'myopic')
        r['ablation']='strict immediate reward, without repair or fuel terminal proxies'
        print(json.dumps(r),flush=True)
