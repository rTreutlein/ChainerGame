"""Bounded pair queries in a fresh planning KB; no oracle probabilities.

The snapshot contains the backend's own marginals and the public causal
rules. Process exit bounds hypothetical contexts independently of history.
"""
import importlib
import json
import sys
import time

from . import metta
from .decisions import Problem, Values, options
from .lookahead import Decision, Parameters, propose_pairs
from .world import Action, Cell, Network, Rates, Route


def deserialize(payload):
    d = payload['network']
    routes = tuple(Route(**{**r,'shipments':tuple(r['shipments'])}) for r in d['routes'])
    by_name = {r.name:r for r in routes}
    cell = Cell(by_name[d['cell']['fuel_route']['name']],d['cell']['inputs'],by_name[d['cell']['alternate']['name']])
    net = Network(d['regions'],routes,cell)
    values = Values(**payload['values'])
    p = payload['decision']['problem']
    problem = Problem(p['period'],p['previous'],p['inputs'],p['mine_before'],p['fuel_before'],options(cell,values.capacity,inspect=True),values)
    decision = Decision(problem,**{k:v for k,v in payload['decision'].items() if k!='problem'})
    return net,Rates(**payload['rates']),values,Parameters(**payload['parameters']),decision


def action_facts(cell, action, time, decision):
    if time=='now':
        return [f'(FuelVia {action.via(cell)} now)',f'({"StockedFuel" if action.stock or decision.bank else "Unstocked"} now)',
                f'({"Purchased" if action.stock else "NoPurchase"} now)',
                *(f'({"Repairing" if s in action.repairs else "InService"} {s} now)' for s in cell.sites),
                *(f'(Not (InService {s} now))' for s in action.repairs),
                *(f'(Guard {s} {"free" if s==decision.next_site else "paid" if s in action.inspections else "none"})' for s in cell.sites)]
    return [f'(FuelVia {action.via(cell)} later)',f'({"StockedFuel" if action.stock else "Unstocked"} later)',
            *(f'({"PlanRepair" if s in action.repairs else "NoRepair"} {s} later)' for s in cell.sites)]


def statements(net,rates,marginals,fuel,values,params,decision):
    cell = net.cell
    lines = []
    def fact(name,goal,p): lines.append(f'(: {name} {goal} (STV {p} 1))')
    def rule(name,body,head,p=1,q=0,positive=False):
        tv = f'(STV {p} 1)' if positive else f'(CTV (STV {p} 1) (STV {q} 1))'
        lines.append(f'(: (no_inverse {name}) (Implication {body} {head}) {tv})')
    for kind,s,p in marginals: fact(f'belief-{kind}-{s}',f'({kind} {s} now)',p)
    fact('incoming','(FuelArrived now)',fuel)
    # Positive sufficient causes avoid contradictory negative branches. Their
    # heads are complete, so a cause whose condition fails derives nothing.
    rule('fuel-stock','(StockedFuel $t)',f'(Fuelled power-plant $t)',positive=True)
    rule('fuel-mine','(And (Unstocked $t) (FuelArrived $t))','(Fuelled power-plant $t)',positive=True)
    rule('purchased-fuel','(And (Purchased now) (FuelVia $r now) (Not (Blocked $r now)))','(FuelArrived later)',positive=True)
    rule('natural-fuel','(And (NoPurchase now) (FuelVia $r now) (Producing mine now) (Not (Blocked $r now)))','(FuelArrived later)',positive=True)
    region = cell.fuel_route.region
    rule('future-storm',f'(Storm {region} now)',f'(Storm {region} later)',rates.persist,rates.storm['coastal'])
    for route in net.routes:
        rule(f'future-block-{route.name}',f'(Storm {region} later)',f'(Blocked {route.name} later)',rates.block_given_storm[route.kind],rates.block_without_storm)
        for shipment in route.shipments:
            rule(f'late-{shipment}',f'(Blocked {route.name} now)',f'(Late {shipment} now)',rates.late_given_blocked,rates.late_given_open)
    for site,kind in cell.sites.items():
        rule(f'degrade-{site}',f'(And (Degraded {site} now) (InService {site} now))',f'(Degraded {site} later)',rates.degraded_persist,rates.degrade[kind])
        rule(f'unplanned-{site}',f'(NoRepair {site} later)',f'(InService {site} later)',positive=True)
        for guard,t in [('free','later'),('paid','later')]:
            rule(f'repair-{guard}-{site}',f'(And (PlanRepair {site} later) (Guard {site} {guard}) (Degraded {site} {t}))',f'(Repairing {site} later)',positive=True)
            rule(f'service-{guard}-{site}',f'(And (PlanRepair {site} later) (Guard {site} {guard}) (Not (Degraded {site} {t})))',f'(InService {site} later)',positive=True)
        rule(f'repair-none-{site}',f'(And (PlanRepair {site} later) (Guard {site} none))',f'(Repairing {site} later)',positive=True)
        if site in cell.inputs:
            fact(f'input-now-{site}',f'(InputArrived {site} now)',int(decision.problem.inputs[site]))
            rule(f'input-later-{site}',f'(Not (Late {cell.inputs[site]} now))',f'(InputArrived {site} later)')
        rule(f'operational-unplanned-{site}',f'(And (NoRepair {site} later) (Not (Degraded {site} later)))',f'(Operational {site} later)',positive=True)
        for guard in ('free','paid'):
            rule(f'operational-{guard}-{site}',f'(And (PlanRepair {site} later) (Guard {site} {guard}) (Not (Degraded {site} later)))',f'(Operational {site} later)',positive=True)
        for t,phase in [('now',decision.phase),('later',1-decision.phase)]:
            healthy = f'(Not (Degraded {site} now)) (InService {site} now)' if t=='now' else f'(Operational {site} later)'
            source = f'(Fuelled power-plant {t})' if site=='power-plant' else f'(Producing power-plant {t})'
            body = f'(And {source} {healthy}{" (InputArrived "+site+" "+t+")" if site in cell.inputs else ""})'
            rule(f'production-{site}-{t}',body,f'(Producing {site} {t})')
            amount = values.output[kind]*(params.rush_output if phase else 1)
            rule(f'output-{site}-{t}',f'(Producing {site} {t})',f'(Outcome {t} (Output {site}) {amount})')
            if phase:
                branch = f'(InService {site} now)' if t=='now' else f'(NoRepair {site} later)'
                rule(f'fine-{site}-{t}',f'(And (Degraded {site} {t}) {branch})',f'(Outcome {t} (Fine {site}) {-params.certification})')
        rule(f'repair-cost-{site}',f'(Repairing {site} later)',f'(Outcome later (RepairCost {site}) {-params.rush_repair if not decision.phase else -values.cost["repair"]})')
    stock_cost = -params.rush_stock if not decision.phase else -values.cost['stock']
    rule('stock-cost-no-buffer','(And (StockedFuel later) (NoPurchase now))',f'(Outcome later StockCost {stock_cost})',positive=True)
    rule('stock-cost-failed-buffer','(And (StockedFuel later) (Purchased now) (FuelVia $r now) (Blocked $r now))',f'(Outcome later StockCost {stock_cost})',positive=True)
    rule('inventory','(And (FuelVia $r later) (Producing mine later) (Not (Blocked $r later)))',f'(Outcome later Inventory {values.fuel})')
    for t in ('now','later'): fact(f'empty-{t}',f'(Outcome {t} Empty 0)',1)
    return lines


def pair_query(cell,first,second,decision):
    facts = action_facts(cell,first,'now',decision)+action_facts(cell,second,'later',decision)
    return f'(: $p (Assuming (And {" ".join(facts)}) (Outcome $t $o $v)) $tv)'


def outcomes(answers):
    result = {}
    for answer in answers:
        goal = answer.statement[2]
        if answer.tv[0]=='STV':
            key = repr(goal[1:-1])
            old = result.get(key)
            if old is None or answer.tv[2]>old[1]: result[key] = (float(goal[-1])*answer.tv[1],answer.tv[2])
    return sum(v for v,c in result.values()) if result else None


def decide(payload):
    started = time.perf_counter()
    if payload.get('python_path'): sys.path.insert(0,payload['python_path'])
    net,rates,values,params,decision = deserialize(payload)
    cell = net.cell
    unsupported = []
    if payload['backend']=='mm2':
        probe = importlib.import_module('mm2_chainer.petta').PeTTaChainer()
        missing = [setting for setting in ('set_complete_predicate','set_positive_only_predicate') if not hasattr(probe,setting)]
        if missing:
            probe.add_atoms_no_check(['(: empty (Outcome now Empty 0) (STV 1 1))'])
            rows = probe.query_many_refs(['(: $p (Assuming (Decision now) (Outcome now Empty 0)) $tv)'],steps=payload['budget'])
            return {'chosen':0,'candidate_pairs':0,'context_bound':1,'decision_queries':1,'decision_answered':0,
                    'open_answered':0,'worker_setup_seconds':time.perf_counter()-started,
                    'unsupported':missing+(['Assuming returned no answers'] if not rows[0] else []),
                    'backend_unavailable':True,'pair_scores':[[0,0,None]]}
    module = 'mm2_chainer.petta' if payload['backend']=='mm2' else 'pettachainer'
    c = importlib.import_module(module).PeTTaChainer()
    clock = c.timing() if hasattr(c,'timing') else None
    c.set_evidence_confidence_k(5)
    # Each head a positive rule concludes is complete: its rules are every way it
    # holds. An open head would read a failed cause as unknown, (STV 0.5 0),
    # and count half a stock cost (PeTTaChainer #126).
    for head in ('Producing','Fuelled','FuelArrived','Repairing','InService','Operational','Outcome'):
        if hasattr(c,'set_complete_predicate'): c.set_complete_predicate(head)
        else: unsupported.append('set_complete_predicate '+head)
    for head in ('InputArrived','FuelVia','StockedFuel','Unstocked','Purchased','NoPurchase','PlanRepair','NoRepair','Guard'):
        if hasattr(c,'set_positive_only_predicate'): c.set_positive_only_predicate(head)
        else: unsupported.append('set_positive_only_predicate '+head)
    c.add_atoms_no_check(statements(net,rates,payload['marginals'],payload['fuel'],values,params,decision))
    setup = time.perf_counter()-started
    firsts,seconds = options(cell,values.capacity,inspect=True),options(cell,values.capacity)
    queries = 0
    def evaluate(pairs):
        nonlocal queries
        queries += len(pairs)
        goals = [pair_query(cell,a,b,decision) for a,b in pairs]
        rows = c.query_many_refs(goals,steps=payload['budget']*len(goals))
        return {pair:None if (v:=outcomes(row)) is None else v-params.cost(values,pair[0],decision.phase)-values.cost['reroute']*pair[1].reroute for pair,row in zip(pairs,rows)}
    cap,beam = payload['pair_cap'],payload['beam']
    if payload['proposer']=='open':
        evaluated = evaluate([(a,b) for a in firsts if a.units<=1 for b in (Action(),Action(stock=True))][:cap])
        available = {p:v for p,v in evaluated.items() if v is not None}
        scores = {a:max(v for (x,b),v in available.items() if x==a) for a,b in available}
        top = sorted(scores,key=scores.get,reverse=True)[:beam]
        count = min(len(seconds),max(0,(cap-len(evaluated))//max(1,len(top))))
        rules = []
        for i,b in enumerate(seconds[:count]):
            facts = ' '.join(action_facts(cell,b,'later',decision))
            rules.append(f'(: (no_inverse choice-{i}) (Implication (Assuming (And {facts}) (Outcome $t $o $v)) (SecondOutcome b{i} $t $o $v)) (CTV (STV 1 1) (STV 0 1)))')
        if rules: c.add_atoms_no_check(rules)
        goals = [f'(: $p (Assuming (And {" ".join(action_facts(cell,a,"now",decision))}) (SecondOutcome $b $t $o $v)) $tv)' for a in top]
        rows = c.query_many_refs(goals,steps=payload['budget']*len(top)*count) if count else []
        queries += len(rows)
        open_values = {}
        for a,row in zip(top,rows):
            for b in range(count):
                selected = [answer._replace(statement=[*answer.statement[:2],['Outcome',*answer.statement[2][2:]]]) for answer in row if answer.statement[2][1]==f'b{b}']
                v = outcomes(selected)
                if v is not None: open_values[a,seconds[b]] = v-params.cost(values,a,decision.phase)-values.cost['reroute']*seconds[b].reroute
        ranked = open_values or available
        candidates = len(evaluated)+len(top)*count
        context_bound = candidates+len(top)
        open_answered = len(open_values)
    else:
        evaluated = propose_pairs(firsts,seconds,values,evaluate,cap,beam,payload['proposer']=='exhaustive')
        ranked = {p:v for p,v in evaluated.items() if v is not None}
        candidates = context_bound = len(evaluated)
        open_answered = 0
    chosen = max(ranked,key=ranked.get)[0] if ranked else Action()
    result = {'chosen':firsts.index(chosen),'candidate_pairs':candidates,'context_bound':context_bound,'decision_queries':queries,'decision_answered':len(ranked),'open_answered':open_answered,'worker_setup_seconds':setup,'unsupported':unsupported,'pair_scores':[[firsts.index(a),seconds.index(b),v] for (a,b),v in evaluated.items()]}
    if clock is not None:
        after = c.timing()
        result.update({f'chainer_{part}_seconds':after[part]-clock[part] for part in ('wall','inference','overhead')})
    return result


if __name__=='__main__':
    print(json.dumps(decide(json.load(sys.stdin))),flush=True)
