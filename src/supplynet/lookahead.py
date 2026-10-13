"""Stage 5 planning on stage 4's partially observed, delayed network.

The oracle conditions on intervening observations, never on the hidden next
state. Planned maintenance precedes a rush shift, with higher emergency
prices and contractual equipment-certification penalties. Stock deliveries
leave a one-period buffer; commissioned diagnostics report next-shift condition before the next decision.
"""
from __future__ import annotations

import itertools
import json
import random
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from .decisions import Problem, Values, best, combine, greedy, options, singles
from .game import describe
from .world import MINE, POWER, Action, Knowledge, Observation, Rates, enact, generate_network, observe, production, sample_period


@dataclass(frozen=True)
class Parameters:
    rush_output: float = 4.0
    rush_stock: float = 8.0
    rush_repair: float = 6.0
    certification: float = 16.0

    def cost(self, values, action, phase):
        return (values.cost_of(action) + phase * ((self.rush_stock-values.cost['stock'])*action.stock
                + (self.rush_repair-values.cost['repair'])*len(action.repairs)))


def network(seed=1, plants=1):
    """A size reduction of the actual stage 4 generator: one coastal region,
    two input/weather routes with one shipment each, and rf/ra. No contracts
    or hidden labels replace arrivals. Plants use distinct real shipments."""
    net = generate_network(random.Random(seed), 1, 1, 'timed', plants, True)
    chosen = list(dict.fromkeys(next(r for r in net.routes if s in r.shipments) for s in net.cell.inputs.values()))
    chosen += [r for r in net.routes if r not in chosen and r not in net.cell.fuel_routes][:max(0, 2-len(chosen))]
    routes = []
    inputs = {}
    for route in chosen:
        required = [s for s in net.cell.inputs.values() if s in route.shipments]
        kept = tuple(required or route.shipments[:1])
        routes.append(replace(route, shipments=kept))
        inputs.update({site: s for site, s in net.cell.inputs.items() if s in kept})
    return replace(net, regions={next(iter(net.regions)): 'coastal'}, routes=tuple(routes)+net.cell.fuel_routes,
                   cell=replace(net.cell, inputs=inputs))


@dataclass(frozen=True)
class Decision:
    problem: Problem
    bank: bool
    phase: int
    next_site: str
    next_route: str
    inspected: dict[str, bool]


class Oracle:
    """Exact finite-observation expectimax, with a reusable transition model.

    Current indices: storm, rf, ra, site degradations, usable arriving fuel.
    Next indices additionally include each plant's observed input arrival.
    The joint observation contains current production, purchased diagnostics,
    buffer receipt, delayed shipment lateness, and the scheduled next probes.
    """
    def __init__(self, net, rates, values, parameters=Parameters()):
        import numpy as np
        self.np, self.net, self.cell, self.rates, self.values, self.parameters = np, net, net.cell, rates, values, parameters
        self.sites = tuple(self.cell.sites)
        self.core_bits = len(self.sites)+3
        self.core_size = 1 << self.core_bits
        self.current_size = self.core_size*2
        self.next_size = self.current_size*(1 << len(self.cell.inputs))
        self.first = options(self.cell, values.capacity, inspect=True)
        self.second = options(self.cell, values.capacity)
        self.second_index = {a:i for i,a in enumerate(self.second)}
        self.core = np.arange(self.core_size)
        self.current = np.arange(self.current_size)
        self.next = np.arange(self.next_size)
        self.shipping = tuple(r for r in net.routes if r not in self.cell.fuel_routes)
        self.shipments = tuple(s for r in self.shipping for s in r.shipments)
        self.input_positions = tuple(self.shipments.index(s) for s in self.cell.inputs.values())
        self.transitions = {}
        for repaired in dict.fromkeys(a.repairs for a in self.first):
            matrix = np.ones((self.core_size, self.core_size))
            p = np.where(self.core & 1, rates.persist, rates.storm['coastal'])
            matrix *= np.where((self.core & 1)[None,:], p[:,None], 1-p[:,None])
            for bit, route in enumerate(self.cell.fuel_routes, 1):
                p = np.where(self.core & 1, rates.block_given_storm[route.kind], rates.block_without_storm)
                matrix *= np.where(((self.core >> bit)&1)[None,:], p[None,:], 1-p[None,:])
            for i,(site,kind) in enumerate(self.cell.sites.items(),3):
                p = np.where(((self.core >> i)&1) & (site not in repaired), rates.degraded_persist, rates.degrade[kind])
                matrix *= np.where(((self.core >> i)&1)[None,:], p[:,None], 1-p[:,None])
            self.transitions[repaired] = matrix
        self.next_rewards = {phase: np.column_stack([self.reward(self.next,a,phase,terminal=True) for a in self.second]) for phase in (0,1)}

    def produced(self, states, action, inputs=None, bank=False):
        np = self.np
        fuel = ((states >> self.core_bits)&1).astype(bool) | action.stock | bank
        power = fuel & ~((states >> (3+self.sites.index(POWER)))&1).astype(bool) & (POWER not in action.repairs)
        outputs = []
        for i,site in enumerate(self.sites):
            on = power & ~((states >> (3+i))&1).astype(bool) & (site not in action.repairs)
            if site in self.cell.inputs:
                pos = tuple(self.cell.inputs).index(site)
                on &= inputs[site] if inputs is not None else ((states >> (self.core_bits+1+pos))&1).astype(bool)
            outputs.append(on)
        return np.column_stack(outputs)

    def reward(self, states, action, phase, inputs=None, bank=False, terminal=False):
        np = self.np
        outputs = self.produced(states,action,inputs,bank)
        amounts = np.array([self.values.output[k] for k in self.cell.sites.values()])*(self.parameters.rush_output if phase else 1)
        result = outputs@amounts-self.parameters.cost(self.values,action,phase)
        if phase:
            result -= self.parameters.certification*sum(((states >> (3+i))&1)*(site not in action.repairs) for i,site in enumerate(self.sites))
        if terminal:
            result += self.values.fuel*outputs[:,self.sites.index(MINE)]*(1-((states >> (1+int(action.reroute)))&1))
        return result

    def labelled_components(self, previous, observation, problem, repaired):
        """Joint weights of released previous labels and current state.
        Labels with identical conditional current distributions are merged:
        unused previous blocks, repaired ancestors and ancestors of observed
        current sites cannot change a future belief once other labels are known.
        """
        np = self.np
        local = np.ones(self.core_size)
        for site,value in observation.degraded.items():
            local *= ((self.core >> (3+self.sites.index(site)))&1)==value
        for route in self.net.routes:
            if route.name not in observation.inspected: continue
            value = observation.inspected[route.name]
            if route in self.cell.fuel_routes:
                bit = 1+self.cell.fuel_routes.index(route)
                local *= ((self.core >> bit)&1)==value
            else:
                p = np.where(self.core&1,self.rates.block_given_storm[route.kind],self.rates.block_without_storm)
                local *= p if value else 1-p
        bit = 1+tuple(r.name for r in self.cell.fuel_routes).index(problem.fuel_before)
        relevant = 1 | ((1 << bit) if problem.mine_before else 0)
        relevant |= sum(1 << (3+i) for i,site in enumerate(self.sites) if site not in repaired and site not in observation.degraded)
        groups = {}
        for state,p in enumerate(previous):
            if p>1e-14:
                key = state & relevant
                groups[key] = groups.get(key,0)+float(p)
        components = []
        for state,p in groups.items():
            row = self.transitions[repaired][state]*local
            norm = float(row.sum())
            if norm==0: raise ValueError('impossible released-label posterior')
            fuel = problem.mine_before and not bool((state>>bit)&1)
            weights = np.zeros(self.current_size)
            weights[self.core+int(fuel)*self.core_size] = p*row/norm
            components.append(weights)
        return np.array(components)

    def belief(self, knowledge, observation, problem):
        """An exact observation mixture for labels released after this act.
        Stage 4's oldest unresolved period resolves between the two actions;
        its labels are not known to the first action but inform the second.
        """
        region = self.cell.fuel_route.region
        marginals,_,last = knowledge._filter(region)
        if len(knowledge.window)>1:
            previous = marginals[-2]
            repaired = knowledge.repairing.get(problem.previous,())
        else:
            previous = self.np.zeros(self.core_size)
            state = sum(int(value) << i for i,value in enumerate(knowledge.before[region]))
            previous[state] = 1
            repaired = knowledge.repairs_before
        components = self.labelled_components(previous,observation,problem,repaired)
        # Compare with the filter's independent carried-fuel marginal.
        carried = self.np.zeros(self.current_size)
        for (block,state),p in last.items():
            carried[state+int(problem.mine_before and not block)*self.core_size] += p
        if not self.np.allclose(components.sum(axis=0),carried,rtol=1e-10,atol=1e-12):
            raise ValueError('released-label mixture disagrees with current filter')
        return components

    def factored(self, marginals, mine_before=True):
        np = self.np
        weights = np.ones(self.current_size)
        probs = [marginals['Storm',self.cell.fuel_route.region], *(marginals['Blocked',r.name] for r in self.cell.fuel_routes),
                 *(marginals['Degraded',s] for s in self.sites), marginals['fuel'] if mine_before else 0]
        for i,p in enumerate(probs): weights *= np.where((self.current >> i)&1,p,1-p)
        return weights

    def myopic(self, weights, decision):
        """Stage 4's one-period joint policy, including its repair/fuel proxies,
        evaluated on current observed inputs and current stage 5 prices/fines."""
        np = self.np
        result = {}
        for a in self.first:
            r = self.reward(self.current,a,decision.phase,decision.problem.inputs,decision.bank)
            prod = self.produced(self.current,a,decision.problem.inputs,decision.bank)
            r += self.values.fuel*prod[:,self.sites.index(MINE)]*(1-((self.current >> (1+int(a.reroute)))&1))
            r += sum(self.values.repair[self.cell.sites[s]]*((self.current >> (3+self.sites.index(s)))&1) for s in a.repairs)
            result[a] = float(weights@r)
        return result

    def forecast(self, weights, decision, first):
        """P(observation, next hidden state | belief, first action).
        Observation weights are kept unnormalized for max_b sum_s w*r."""
        np, S = self.np, self.core_size
        prod = self.produced(self.current,first,decision.problem.inputs,decision.bank)
        reports = prod@np.array([1<<i for i in range(len(self.sites))])
        opened = 1-((self.current >> (1+int(first.reroute)))&1)
        bank = first.stock*opened
        fuel = opened*(first.stock | prod[:,self.sites.index(MINE)])
        diag = sum((((self.core >> (3+self.sites.index(s)))&1) << i for i,s in enumerate(first.inspections)), np.zeros(S,dtype=int))
        groups = (reports << (1+len(first.inspections))) | bank
        observations = (1 << (len(self.sites)+1+len(first.inspections)))*4*(1 << len(self.shipments))
        joint = np.zeros((observations,self.next_size))
        transition = self.transitions[first.repairs][self.current % S,:]*weights[:,None]
        sensor_site = (self.core >> (3+self.sites.index(decision.next_site)))&1
        probe = next(r for r in self.net.routes if r.name==decision.next_route)
        route_bit = 1 if probe==self.cell.fuel_route else 2 if probe==self.cell.alternate else None
        for late_bits in range(1 << len(self.shipments)):
            # A retained route may have several plant-input shipments. Sum
            # its block once, preserving their shared cause and correlations.
            late_weight = np.ones(self.current_size)
            offset = 0
            for route in self.shipping:
                pblock = np.where(self.current & 1,self.rates.block_given_storm[route.kind],self.rates.block_without_storm)
                if route.name in decision.inspected: pblock = np.full(self.current_size,float(decision.inspected[route.name]))
                like = []
                for blocked in (False,True):
                    p = self.rates.late_given_blocked if blocked else self.rates.late_given_open
                    likelihood = 1.0
                    for j in range(len(route.shipments)):
                        likelihood *= p if (late_bits >> (offset+j))&1 else 1-p
                    like.append(likelihood)
                late_weight *= pblock*like[1]+(1-pblock)*like[0]
                offset += len(route.shipments)
            inputs = sum((1-((late_bits >> bit)&1)) << i for i,bit in enumerate(self.input_positions))
            next_index = self.core[None,:]+fuel[:,None]*S+inputs*S*2
            if route_bit is not None:
                cases = [((self.core >> route_bit)&1, np.ones(S))]
            else:
                p = np.where(self.core & 1,self.rates.block_given_storm[probe.kind],self.rates.block_without_storm)
                cases = [(np.zeros(S,dtype=int),1-p),(np.ones(S,dtype=int),p)]
            for sensor_route,likelihood in cases:
                obs = (groups[:,None]+(diag[None,:] << 1))*(4*(1 << len(self.shipments)))+late_bits*4+sensor_site[None,:]*2+sensor_route[None,:]
                w = transition*late_weight[:,None]*likelihood[None,:]
                np.add.at(joint,(obs.ravel(),np.broadcast_to(next_index,w.shape).ravel()),w.ravel())
        return joint

    def execute(self, first, second, observation, decision):
        """A legal pair's second prescription: use the observed buffer and
        repair only positive diagnostics where a diagnostic is available.
        All executed actions are subsets, so capacity cannot be exceeded."""
        group = observation // (4*(1 << len(self.shipments)))
        bank = bool(group & 1)
        repaired = []
        for s in second.repairs:
            if s == decision.next_site:
                positive = bool((observation >> 1)&1)
            elif s in first.inspections:
                positive = bool((group >> (1+first.inspections.index(s)))&1)
            else:
                positive = True
            if positive: repaired.append(s)
        return Action(second.stock and not bank,second.reroute,tuple(repaired))

    def evaluate(self, weights, decision, pairs=True, firsts=None):
        np = self.np
        components = weights if weights.ndim==2 else weights[None,:]
        marginal = components.sum(axis=0)
        feedback, fixed, counts = {}, {}, {}
        future = self.next_rewards[1-decision.phase]
        for a in self.first if firsts is None else firsts:
            immediate = float(marginal@self.reward(self.current,a,decision.phase,decision.problem.inputs,decision.bank))
            total = 0.0
            branches = 0
            for component in components:
                joint = self.forecast(component,decision,a)
                useful = np.flatnonzero(joint.sum(axis=1)>1e-14)
                conditioned = joint[useful]@future
                total += float(conditioned.max(axis=1).sum())
                branches += len(useful)
            feedback[a] = immediate+total
            counts[a] = branches
            if pairs:
                # Pair prescriptions do not read the retrospective labels;
                # linearity permits their values from the marginal once.
                joint = self.forecast(marginal,decision,a) if len(components)>1 else joint
                useful = np.flatnonzero(joint.sum(axis=1)>1e-14)
                conditioned = joint[useful]@future
                for b in self.second:
                    indices = np.array([self.second_index[self.execute(a,b,int(o),decision)] for o in useful])
                    fixed[a,b] = immediate+float(conditioned[np.arange(len(useful)),indices].sum())
        return fixed, feedback, counts


def propose_pairs(firsts, seconds, values, evaluate, cap, beam=2, exhaustive=False):
    evaluated = {}
    def ask(pairs):
        fresh = list(dict.fromkeys(p for p in pairs if p not in evaluated))[:max(0,cap-len(evaluated))]
        if fresh: evaluated.update(evaluate(fresh))
    if exhaustive:
        ask(itertools.product(firsts,seconds))
        return evaluated
    seed = [a for a in firsts if a.units<=1]
    ask((a,b) for a in seed for b in (Action(),Action(stock=True)))
    # Information has value only with a responsive continuation. Include
    # each inspection's repair-on-positive prescription as a seed.
    ask((a,Action(repairs=a.inspections)) for a in seed if a.inspections)
    layer = [a for a in firsts if a.units==1]
    feasible = set(firsts)
    for _ in range(values.capacity-1):
        scores = {a:max(v for (x,b),v in evaluated.items() if x==a and v is not None)
                  for a in layer if any(x==a and v is not None for (x,b),v in evaluated.items())}
        top = sorted(scores,key=scores.get,reverse=True)[:beam]
        layer = list(dict.fromkeys(combine([a,b]) for a in top for b in seed if b.units==1 and b not in singles(a)))
        layer = [a for a in layer if a in feasible]
        ask((a,b) for a in layer for b in (Action(),Action(stock=True)))
    scores = {a:max(v for (x,b),v in evaluated.items() if x==a and v is not None)
              for a in dict.fromkeys(a for a,b in evaluated) if any(x==a and v is not None for (x,b),v in evaluated.items())}
    top = sorted(scores,key=scores.get,reverse=True)[:beam]
    ask((a,b) for b in seconds for a in top)
    return evaluated


class ProbLogPlanning:
    """ProbLog independently infers joint resolving-label beliefs and conditional
    state-transition tables; the common numeric planner conditions/maximizes those
    probabilities. It receives observations, rates and labelled history, not
    the reference's posterior or transition weights."""
    def __init__(self, net, rates, values, history, parameters, engine, timeout):
        import tempfile
        from .problog_backend import ProblogBackend, solve, summary
        scratch = Path(__file__).resolve().parents[2]/'benchmarks/supplynet_stage5/scratch'
        scratch.mkdir(exist_ok=True)
        tempfile.tempdir = str(scratch)
        self.backend = ProblogBackend(engine=engine,timeout=timeout)
        self.backend.begin(net,rates,history)
        self.model = Oracle(net,rates,values,parameters)
        self.records = []
        sites = self.model.sites
        tables = []
        self.available = True
        # Compile each conditional table separately: one shared six-bit
        # future, rather than sixteen independent copies in one circuit.
        for parent in range(1 << (len(sites)+1)):
            ps = rates.storm_rate('coastal',parent&1)
            lines = [f'{ps}::ns.']
            for i,r in enumerate(net.cell.fuel_routes):
                lines += [f'{rates.block_given_storm[r.kind]}::nb_{i} :- ns.',
                          f'{rates.block_without_storm}::nb_{i} :- \\+ns.']
            for i,(site,kind) in enumerate(net.cell.sites.items()):
                lines.append(f'{rates.degraded_rate(kind,(parent>>(i+1))&1)}::nd_{i}.')
            atoms = ['ns','nb_0','nb_1',*[f'nd_{i}' for i in range(len(sites))]]
            queries = []
            for state in range(self.model.core_size):
                name = f'tr_{state}'
                body = ', '.join(atom if (state>>i)&1 else '\\+'+atom for i,atom in enumerate(atoms))
                lines += [f'{name} :- {body}.',f'query({name}).']
                queries.append(name)
            probabilities,record = solve('\n'.join(lines),queries,engine,timeout)
            self.records.append(record)
            if probabilities is None:
                self.available = False
                break
            tables.append(probabilities)
        if self.available:
            tables = self.model.np.array(tables)
            for repaired in self.model.transitions:
                parents = []
                for c in range(self.model.core_size):
                    p = c&1
                    for i,s in enumerate(sites): p |= int(bool((c>>(i+3))&1) and s not in repaired) << (i+1)
                    parents.append(p)
                self.model.transitions[repaired] = tables[parents]

    def weights(self, observation, decision, extra):
        from .problog_backend import solve, term
        backend,model = self.backend,self.model
        backend._add(extra)
        backend._observe(observation)
        atoms = [('Storm',model.cell.fuel_route.region,decision.problem.previous),
                 *(('Blocked',r.name,decision.problem.previous) for r in model.cell.fuel_routes),
                 *(('Degraded',site,decision.problem.previous) for site in model.sites)]
        lines,queries = [],[]
        for state in range(model.core_size):
            literals = [term(a) if (state>>i)&1 else '\\+'+term(a) for i,a in enumerate(atoms)]
            name = f'previous_{state}'
            lines += [f"{name} :- {', '.join(literals)}.",f'query({name}).']
            queries.append(name)
        probabilities,record = solve(backend.program([])+'\n'+'\n'.join(lines),queries,backend.engine,backend.timeout)
        self.records.append(record)
        if probabilities is None: return None
        repaired = tuple(sorted(expr[1] for expr,value in backend.facts.get(decision.problem.previous,()) if expr[0]=='Repairing' and value))
        return model.labelled_components(probabilities,observation,decision.problem,repaired)

    def summary(self):
        from .problog_backend import summary
        return summary(self.records)


def run(config, backend, proposer='extend', pair_cap=64, beam=2, python_path=None,
        problog_engine='ddnnf', problog_timeout=15, on_round=None, parameters=Parameters()):
    setup_start = time.perf_counter()
    net, rates = network(config.seed,config.plants), Rates(persist=0.7)
    cell,values = net.cell,config.values
    model = Oracle(net,rates,values,parameters)
    knowledge = Knowledge(net,rates.nodes(net))
    rng = random.Random(config.seed)
    probe_rng = random.Random(10000+config.seed)
    timeline,history = [],[]
    for i in range(config.history):
        p = sample_period(net,rates,rng,f'h{i+1}',timeline)
        timeline.append(p)
        obs = observe(net,p,rng,1.0,{(s,p.name):v for s,v in p.late.items()},timeline[-2].producing if len(timeline)>1 else {})
        knowledge.observe(obs)
        knowledge.act(p.name,p.action(cell))
        knowledge.resolve(p)
        history.append((p,obs))
    belief_backend = None
    problog = None
    if backend in ('pettachainer','mm2'):
        # MM2's compiled ABI differs. Its worker independently builds the
        # observed-history belief KB in its matching interpreter each round.
        if backend=='pettachainer':
            from .backends import PeTTaChainerBackend
            belief_backend = PeTTaChainerBackend(python_path)
            belief_backend.begin(net,rates,history)
    if backend=='prior':
        from .backends import PriorBackend
        belief_backend = PriorBackend()
        belief_backend.begin(net,rates,history)
    if backend=='problog':
        problog = ProbLogPlanning(net,rates,values,history,parameters,problog_engine,problog_timeout)
    # Probe itineraries are known before acting; probe values are not. This
    # preserves partial inspection while keeping the observation tree small.
    itinerary = [(probe_rng.choice(tuple(cell.sites)),probe_rng.choice(net.routes).name) for _ in range(config.rounds+1)]
    bank = False
    paid = ()
    receipt = None
    rounds = []
    observations = []
    resolved = []
    actions = []
    setup_seconds = time.perf_counter()-setup_start
    for index in range(config.rounds):
        previous = timeline[-1]
        p = sample_period(net,rates,rng,f't{index+1}',timeline)
        timeline.append(p)
        probe_site,probe_route = itinerary[index]
        obs = Observation(p.name,previous.name,{(s,previous.name):previous.late[s] for s in model.shipments},
                          {probe_route:p.blocked[probe_route]}, {s:p.degraded[s] for s in (probe_site,*paid)},None,p.inputs,previous.producing)
        extra = []
        if receipt is not None:
            r,blocked = receipt
            extra.append(f'(: buffer-receipt-{previous.name} (Blocked {r} {previous.name}) (STV {int(blocked)} 1))')
            if previous.name in knowledge.window: knowledge.inspected[r,previous.name] = blocked
        knowledge.observe(obs)
        problem = Problem(p.name,previous.name,p.inputs,previous.producing[MINE],previous.via,model.first,values)
        decision = Decision(problem,bank,index%2,*itinerary[index+1],obs.inspected)
        start = time.perf_counter()
        components = model.belief(knowledge,obs,problem)
        weights = components.sum(axis=0)
        pairs,qs,counts = model.evaluate(components,decision)
        reference_seconds = time.perf_counter()-start
        reference = best(qs)
        start = time.perf_counter()
        myopic_values = model.myopic(weights,decision)
        myopic = best(myopic_values)
        myopic_seconds = time.perf_counter()-start
        start = time.perf_counter()
        details = {}
        if backend=='reference':
            action = reference
        elif backend=='myopic':
            action = myopic
        elif backend=='greedy':
            action = greedy(myopic_values,values.capacity)
        elif backend in ('openloop','beam'):
            evaluated = pairs if backend=='openloop' else propose_pairs(model.first,model.second,values,lambda ps:{x:pairs[x] for x in ps},pair_cap,beam)
            action = max(evaluated,key=evaluated.get)[0]
            details['candidate_pairs'] = len(evaluated)
        elif backend=='problog':
            wp = problog.weights(obs,decision,extra)
            if wp is None or not problog.available:
                action = Action()
            else:
                _,qp,_ = problog.model.evaluate(wp,decision,pairs=False)
                action = best(qp)
                details['problog_reference_max_error'] = max(abs(qp[a]-qs[a]) for a in qs)
        else:
            keys = [('Storm',cell.fuel_route.region,p.name), *(('Degraded',s,p.name) for s in cell.sites),
                    *(('Blocked',r.name,p.name) for r in net.routes),('Blocked',previous.via,previous.name)]
            if belief_backend is not None:
                if backend=='pettachainer' and extra: belief_backend._handler.add_atoms_no_check(extra)
                # Always ask uninspected variables; directly observed values
                # are supplied as observations, never as hidden-state labels.
                unknown = keys if backend=='prior' else [k for k in keys if not (k[2]==p.name and k[1] in (obs.inspected|obs.degraded))]
                beliefs = belief_backend.beliefs(obs,unknown,config.budget)
                if backend!='prior':
                    beliefs.update({('Blocked',s,p.name):float(v) for s,v in obs.inspected.items()})
                    beliefs.update({('Degraded',s,p.name):float(v) for s,v in obs.degraded.items()})
                if receipt is not None and backend!='prior': beliefs['Blocked',receipt[0],previous.name] = float(receipt[1])
                if previous.name not in knowledge.window and backend!='prior':
                    beliefs['Blocked',previous.via,previous.name] = float(previous.blocked[previous.via])
                marginal = {(kind,s):beliefs.get((kind,s,p.name),0.5) for kind,s,t in keys if t==p.name}
                marginal['fuel'] = float(problem.mine_before)*(1-beliefs.get(('Blocked',previous.via,previous.name),0.5))
            else:
                marginal = {}
            if backend=='prior':
                wp = model.factored(marginal,problem.mine_before)
                pp,_,_ = model.evaluate(wp,replace(decision,inspected={}))
                action = max(pp,key=pp.get)[0]
            else:
                observed_keys = [('Storm',cell.fuel_route.region),*(('Blocked',r.name) for r in cell.fuel_routes),*(('Degraded',s) for s in model.sites)]
                differences = [abs(marginal[k]-float(weights@((model.current>>i)&1))) for i,k in enumerate(observed_keys)] if marginal else []
                if differences: details['belief_mean_error'] = sum(differences)/len(differences)
                # Only noisy observations, labelled past history and the
                # chainer's own marginals cross the worker boundary.
                payload = {'network':asdict(net),'rates':asdict(rates),'values':asdict(values),
                           'parameters':asdict(parameters),'decision':asdict(decision),'marginals':[[k[0],k[1],v] for k,v in marginal.items() if k!='fuel'],
                           'fuel':marginal.get('fuel',0.5),'backend':backend,'python_path':python_path,
                           'budget':config.budget,'pair_cap':pair_cap,'beam':beam,'proposer':proposer,
                           'history':[[asdict(a),as_observation(b)] for a,b in history] if backend=='mm2' else [],
                           'observations':observations+[as_observation(obs)] if backend=='mm2' else [],
                           'extra_facts':extra, 'actions':actions, 'resolved':resolved}
                executable = '/nexus/Dev/OpenCog/NL2PLN_Project/mm2-chainer/.venv/bin/python' if backend=='mm2' else sys.executable
                child = subprocess.run([executable,'-m','supplynet.lookahead_worker'],input=json.dumps(payload),text=True,capture_output=True,timeout=180)
                if child.returncode: raise RuntimeError(child.stderr[-4000:])
                details.update(json.loads(child.stdout.splitlines()[-1]))
                action = model.first[details.pop('chosen')]
                errors = [abs(v-pairs[model.first[a],model.second[b]]) for a,b,v in details['pair_scores'] if v is not None]
                if errors:
                    details['pair_mean_error'] = sum(errors)/len(errors)
                    details['pair_max_error'] = max(errors)
                candidates = {model.first[a] for a,b,v in details['pair_scores']}
                details['proposal_regret'] = qs[reference]-max(qs[a] for a in candidates)
        seconds = time.perf_counter()-start
        if backend in ('reference','openloop','beam'): seconds += reference_seconds
        if backend in ('myopic','greedy'): seconds += myopic_seconds
        # Physical enactment is stage 4's, using the observed stored buffer.
        effective = replace(action,stock=action.stock or bank)
        p = timeline[-1] = enact(net,p,previous,effective)
        natural = previous.producing[MINE] and not previous.blocked[previous.via]
        actual_mask = int(p.storms[cell.fuel_route.region]) | (int(p.blocked[cell.fuel_route.name])<<1) | (int(p.blocked[cell.alternate.name])<<2)
        for i,s in enumerate(model.sites,3): actual_mask |= int(p.degraded[s]) << i
        actual_mask |= int(natural) << model.core_bits
        realized = float(model.reward(model.np.array([actual_mask]),action,decision.phase,p.inputs,bank)[0])
        record = {'type':'round','round':index+1,'phase':decision.phase,'action':describe(action),'reference_action':describe(reference),
                  'myopic_action':describe(myopic),'regret':max(0,qs[reference]-qs[action]),
                  'myopic_regret':qs[reference]-qs[myopic],'myopic_suboptimal':qs[reference]-qs[myopic]>1e-8,
                  'lookahead_gain':qs[action]-qs[myopic],'utility':realized,'decision_seconds':seconds,'reference_seconds':reference_seconds,
                  'observation_branches':max(counts.values()),'retrospective_branches':len(components),'posterior_support':int((weights>1e-12).sum()),
                  'feedback_value_gap':qs[reference]-max(pairs.values()),
                  'paid_inspections':len(action.inspections),'bank':bank,'inputs':p.inputs,**details}
        rounds.append(record)
        if on_round: on_round(record)
        knowledge.act(p.name,effective)
        if belief_backend is not None: belief_backend.act(p.name,effective)
        if problog is not None: problog.backend.act(p.name,effective)
        observations.append(as_observation(obs))
        actions.append([p.name,asdict(effective)])
        # The previous period resolves only AFTER the current decision.
        if previous.name in knowledge.window:
            knowledge.resolve(previous)
            if belief_backend is not None: belief_backend.resolve(previous,prev_observation)
            if problog is not None: problog.backend.resolve(previous,prev_observation)
            resolved.append([asdict(previous),as_observation(prev_observation)])
        paid = action.inspections
        receipt = (action.via(cell),p.blocked[action.via(cell)]) if action.stock else None
        bank = bool(action.stock and not p.blocked[action.via(cell)])
        prev_observation = obs
    result = {'type':'run-summary','stage':5,'backend':backend,'seed':config.seed,'rounds':config.rounds,'history':config.history,
              'budget':config.budget,'plants':config.plants,'capacity':values.capacity,'options':len(model.first),'second_options':len(model.second),
              'all_pairs':len(model.first)*len(model.second),'current_states':model.current_size,'next_states':model.next_size,
              'information':'stage 4 partial observations; exact observation-conditioned continuation',
              'horizon':2,'proposer':proposer,'pair_cap':pair_cap,'actions':[r['action'] for r in rounds],
              'myopic_suboptimal_fraction':sum(r['myopic_suboptimal'] for r in rounds)/len(rounds),
              'unsupported':sorted({s for r in rounds for s in r.get('unsupported',[])}),
              'backend_unavailable':any(r.get('backend_unavailable',False) for r in rounds)}
    fields = {'retrospective_branches','belief_mean_error','feedback_value_gap','regret','lookahead_gain','myopic_regret','utility','decision_seconds','reference_seconds','observation_branches','posterior_support',
              'paid_inspections','candidate_pairs','decision_queries','decision_answered','open_answered','context_bound','worker_setup_seconds',
              'chainer_wall_seconds','chainer_inference_seconds','chainer_overhead_seconds','pair_mean_error','pair_max_error','proposal_regret',
              'problog_reference_max_error'}
    for f in fields:
        if all(r.get(f) is not None for r in rounds): result[f] = sum(r[f] for r in rounds)/len(rounds)
    result['setup_seconds'] = setup_seconds
    result['amortized_seconds'] = result['decision_seconds']+setup_seconds/config.rounds
    result['early_decision_seconds'] = sum(r['decision_seconds'] for r in rounds[:10])/len(rounds[:10])
    result['late_decision_seconds'] = sum(r['decision_seconds'] for r in rounds[-10:])/len(rounds[-10:])
    if problog is not None: result.update(problog.summary())
    return result


def as_observation(obs):
    # JSON cannot encode tuple keys for delayed shipments.
    d = asdict(obs)
    d['late'] = [[s,t,v] for (s,t),v in obs.late.items()]
    return d
