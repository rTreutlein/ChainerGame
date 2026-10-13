"""Count unique reached decision beliefs as well as rounds.
Keys include network, rounded posterior, current public state and future
probe itinerary. This supplements uniform structural-state enumeration.
"""
import hashlib
import json
from dataclasses import asdict
from unittest.mock import patch
from supplynet.game import GameConfig
from supplynet.lookahead import Oracle,run

original=Oracle.evaluate
keys=[]
def counted(model,weights,decision,*args,**kwargs):
    public={'inputs':decision.problem.inputs,'bank':decision.bank,'phase':decision.phase,'next_site':decision.next_site,'next_route':decision.next_route,'inspected':decision.inspected}
    rows=weights[None,:] if weights.ndim==1 else weights
    key=json.dumps({'network':asdict(model.net),'public':public},sort_keys=True).encode()+b''.join(sorted(row.round(12).tobytes() for row in rows))
    keys.append(hashlib.sha256(key).hexdigest())
    return original(model,weights,decision,*args,**kwargs)

for backend in ('reference','myopic'):
    unique={}
    rows=[]
    for seed in range(1,5):
        def record(row):
            key=keys[-1]
            loss=row['myopic_suboptimal']
            if key in unique:assert unique[key]==loss
            unique[key]=loss
            rows.append({'seed':seed,'round':row['round'],'belief':key,'myopic_suboptimal':loss,'myopic_regret':row['myopic_regret']})
        with patch.object(Oracle,'evaluate',counted):
            run(GameConfig(seed=seed,rounds=30,stage=5,plants=1),backend,on_round=record)
    print(json.dumps({'type':'visited-belief-probe','policy':backend,'unique_states':len(unique),'suboptimal_states':sum(unique.values()),'state_fraction':sum(unique.values())/len(unique),'rounds':len(rows),'suboptimal_rounds':sum(r['myopic_suboptimal'] for r in rows),'round_fraction':sum(r['myopic_suboptimal'] for r in rows)/len(rows),'myopic_regret':sum(r['myopic_regret'] for r in rows)/len(rows),'rows':rows}),flush=True)
