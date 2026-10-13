"""Known-current-state control: compare returned pairs with exact sums.
This is a diagnostic fixture, not the partially observed rollout. It also
reports cold versus proposal-context answers for the largest mismatch.
"""
import json
import sys
from dataclasses import asdict
from supplynet.lookahead import Decision,Oracle,Parameters,network
from supplynet.lookahead_worker import decide,statements,pair_query,outcomes
from supplynet.decisions import Problem,Values
from supplynet.world import Action,Rates
from pettachainer import PeTTaChainer

net=network();rates=Rates(persist=.7);values=Values();m=Oracle(net,rates,values)
d=Decision(Problem('now','old',{'plant-a':True},True,'rf',m.first,values),False,0,'mine','rf',{r.name:False for r in net.routes})
marg=[['Storm','north',0],*(['Blocked',r.name,0] for r in net.routes),*(['Degraded',s,0] for s in net.cell.sites)]
uncertain = '--uncertain-mine' in sys.argv
if uncertain:
    marg = [[kind,site,.5 if kind=='Degraded' and site=='mine' else value] for kind,site,value in marg]
payload={'network':asdict(net),'rates':asdict(rates),'values':asdict(values),'parameters':asdict(Parameters()),'decision':asdict(d),'marginals':marg,'fuel':1,'backend':'pettachainer','budget':400,'pair_cap':64,'beam':2,'proposer':'extend'}
result=decide(payload)
w=m.np.zeros(128);w[64]=1
if uncertain:
    w[64]=.5;w[72]=.5
pairs,q,_=m.evaluate(w,d)
errors=[(abs(v-pairs[m.first[a],m.second[b]]),a,b,v,pairs[m.first[a],m.second[b]]) for a,b,v in result['pair_scores'] if v is not None]
worst=max(errors)
print(json.dumps({'uncertain_mine':uncertain,'options':len(m.first),'pairs':len(errors),'mean_error':sum(x[0] for x in errors)/len(errors),'max_error':worst[0],'worst':worst,'chosen':result['chosen']}),flush=True)
a,b=m.first[worst[1]],m.second[worst[2]]
c=PeTTaChainer()
for h in ('Producing','Fuelled','FuelArrived','Repairing','InService','Buffer','Operational'):c.set_complete_predicate(h)
for h in ('InputArrived','FuelVia','StockedFuel','Unstocked','Purchased','NoPurchase','PlanRepair','NoRepair','Guard'):c.set_positive_only_predicate(h)
c.add_atoms_no_check(statements(net,rates,marg,1,values,Parameters(),d))
for budget in (400,2000):
 rows=c.query_many_refs([pair_query(net.cell,a,b,d)],steps=budget)[0]
 print('cold/requery budget',budget,'value',outcomes(rows)-Parameters().cost(values,a,0)-values.cost['reroute']*b.reroute,'exact',worst[4],flush=True)
 for row in rows:print(row.statement[2],row.tv,flush=True)
