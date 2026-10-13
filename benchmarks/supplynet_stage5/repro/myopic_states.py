"""Uniform structural coverage of control states, distinct from rollouts.
Current core labels are point beliefs; uninspected shipping blocks remain
conditional on weather. Vary phase, observed input and warehouse receipt.
The next itinerary is held at mine/rf. No visitation distribution is implied.
"""
import json
from supplynet.decisions import Problem, Values, best
from supplynet.lookahead import Decision, Oracle, network
from supplynet.world import Rates

model = Oracle(network(),Rates(persist=.7),Values())
count = losses = 0
witnesses = []
for phase in (0,1):
 for arrived in (False,True):
  for bank in (False,True):
   for state in range(model.current_size):
    weights = model.np.zeros(model.current_size)
    weights[state] = 1
    d = Decision(Problem('now','old',{'plant-a':arrived},True,'rf',model.first,model.values),bank,phase,'mine','rf',{})
    _,q,_ = model.evaluate(weights,d,pairs=False)
    reference,myopic = best(q),best(model.myopic(weights,d))
    regret = q[reference]-q[myopic]
    count += 1
    losses += regret>1e-8
    if regret>1e-8:
     witnesses.append({'state':state,'phase':phase,'input':arrived,'bank':bank,'reference':repr(reference),'myopic':repr(myopic),'regret':regret})
print(json.dumps({'type':'structural-state-probe','states':count,'myopic_suboptimal':losses,'fraction':losses/count,'max_myopic_regret':max(x['regret'] for x in witnesses),'witnesses':witnesses}),flush=True)
