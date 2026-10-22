import itertools
import os
import random
import unittest
from unittest.mock import Mock, call, patch
from dataclasses import replace
from types import SimpleNamespace
import json

from supplynet import metta
from supplynet import lookahead
from supplynet.backends import GreedyBackend, LearnedReferenceBackend, PeTTaChainerBackend, PriorBackend, ReferenceBackend
from supplynet.decisions import Problem, Values, best, expected_utilities, factored_utilities, greedy, options, utility
from supplynet.game import GameConfig, run_game, stage_rates
from supplynet.world import (
    MINE,
    POWER,
    Action,
    Cell,
    Knowledge,
    Network,
    Observation,
    Period,
    Rates,
    Route,
    counts,
    enact,
    generate_network,
    learned_rates,
    observe,
    production,
    sample_period,
)


class LookaheadTests(unittest.TestCase):
    def setUp(self):
        self.net = lookahead.network()
        self.values, self.rates = Values(), Rates(persist=.7)
        self.model = lookahead.Oracle(self.net,self.rates,self.values)
        self.cell = self.net.cell
        self.decision = lookahead.Decision(Problem('now','before',{'plant-a':True},True,'rf',self.model.first,self.values),False,0,'mine','rf',{})

    def test_real_inputs_partial_weather_and_legal_inspections(self):
        self.assertEqual(len(self.net.routes),4)
        self.assertTrue(all(r.transit==1 for r in self.net.routes))
        self.assertTrue(all(any(s in r.shipments for r in self.net.routes) for s in self.cell.inputs.values()))
        self.assertIn(Action(repairs=(POWER,),inspections=(POWER,)),self.model.first)
        self.assertEqual((len(self.model.first),len(self.model.second)),(37,16))
        self.assertTrue(all(a.units<=2 for a in self.model.first))

    def test_forecast_normalizes_and_repairs_affect_hidden_next_damage(self):
        weights = self.model.np.ones(128)/128
        for a in self.model.first:
            joint = self.model.forecast(weights,self.decision,a)
            self.assertAlmostEqual(float(joint.sum()),1)
            states = joint.sum(axis=0)
            for i,(site,kind) in enumerate(self.cell.sites.items(),3):
                p = self.rates.degrade[kind] if site in a.repairs else (self.rates.degrade[kind]+self.rates.degraded_persist)/2
                self.assertAlmostEqual(float(states@((self.model.next>>i)&1)),p)
            for o in self.model.np.flatnonzero(joint.sum(axis=1)):
                for b in self.model.second:
                    self.assertLessEqual(self.model.execute(a,b,int(o),self.decision).units,2)

    def test_expectimax_conditions_on_observations_not_hidden_next_labels(self):
        # Independently calculate all next-world rewards with the physical
        # production function, then compare max after conditioning vs before.
        weights = self.model.np.zeros(128)
        weights[64] = 1  # healthy, open routes, fuel now, unknown future
        first = Action()
        joint = self.model.forecast(weights,self.decision,first)
        future = []
        for state in range(self.model.next_size):
            d = {site:bool((state>>(3+i))&1) for i,site in enumerate(self.cell.sites)}
            amounts = []
            for a in self.model.second:
                prod = production(self.cell,d,a.stock,{'plant-a':bool((state>>7)&1)},not bool((state>>6)&1),True,a.repairs)
                value = 4*sum(self.values.output[self.cell.sites[s]] for s,on in prod.items() if on)
                value -= lookahead.Parameters().cost(self.values,a,1)
                value -= 16*sum(on and site not in a.repairs for site,on in d.items())
                value += self.values.fuel*prod[MINE]*(not bool((state>>(1+int(a.reroute)))&1))
                amounts.append(value)
            future.append(amounts)
        future = self.model.np.array(future)
        self.model.np.testing.assert_allclose(future,self.model.next_rewards[1])
        immediate = sum(self.values.output.values())
        paired,q,_ = self.model.evaluate(weights,self.decision,firsts=[first])
        expectimax = immediate+sum(max(row) for row in joint@future)
        clairvoyant = immediate+float(joint.sum(axis=0)@future.max(axis=1))
        self.assertAlmostEqual(q[first],expectimax)
        self.assertLess(q[first],clairvoyant-.1)
        self.assertTrue(all(v<=q[first]+1e-8 for v in paired.values()))

    def test_delayed_noisy_shipments_update_next_weather_posterior(self):
        weights = self.model.np.zeros(128)
        weights[64] = weights[65] = .5
        joint = self.model.forecast(weights,self.decision,Action())
        late_pattern = (1 << len(self.model.shipments))-1
        rows = [i for i in range(len(joint)) if (i//4)%(1 << len(self.model.shipments))==late_pattern]
        conditioned = joint[rows].sum(axis=0)
        actual = float(conditioned@(self.model.next&1))/float(conditioned.sum())
        likelihood = []
        for storm in (False,True):
            value = 1
            for route in self.model.shipping:
                pblock = self.rates.block_rate(route.kind,storm)
                value *= pblock*self.rates.late_given_blocked**len(route.shipments)+(1-pblock)*self.rates.late_given_open**len(route.shipments)
            likelihood.append(value)
        expected = (likelihood[0]*.3+likelihood[1]*.7)/sum(likelihood)
        self.assertAlmostEqual(actual,expected)
        self.assertGreater(actual,.65)

    def test_retrospective_labels_condition_the_second_decision(self):
        # No fuel now: the production report cannot identify power damage.
        # A released past label distinguishes two current damage priors.
        components = self.model.np.zeros((2,128))
        components[0,0],components[0,16] = .5*.95,.5*.05
        components[1,0],components[1,16] = .5*.25,.5*.75
        first = Action()
        pairs,without,_ = self.model.evaluate(components.sum(axis=0),self.decision,firsts=[first])
        labelled,with_labels,_ = self.model.evaluate(components,self.decision,firsts=[first])
        self.assertGreater(with_labels[first]-without[first],.01)
        for b in self.model.second:
            self.assertAlmostEqual(pairs[first,b],labelled[first,b])

    def test_stock_and_paid_next_shift_information_have_real_value(self):
        m = {('Storm','north'):.5,('Blocked','rf'):.415,('Blocked','ra'):.215,('Degraded',MINE):.2,('Degraded',POWER):.1,('Degraded','plant-a'):.2,'fuel':1}
        w = self.model.factored(m)
        _,q,_ = self.model.evaluate(w,self.decision,pairs=False)
        chosen = best(q)
        self.assertEqual(chosen,Action(stock=True,inspections=('plant-a',)))
        self.assertGreater(q[chosen]-max(v for a,v in q.items() if not a.inspections),.7)
        joint = self.model.forecast(w,self.decision,chosen)
        # Delivered diagnostics label the NEXT site's condition, including
        # new damage; they do not reveal it to the first decision.
        for o in self.model.np.flatnonzero(joint.sum(axis=1)):
            group = int(o)//16
            diagnosed = bool((group>>1)&1)
            support = self.model.np.flatnonzero(joint[o])
            self.assertTrue(all(bool((int(s)>>5)&1)==diagnosed for s in support))

    def test_no_terminal_repair_proxy_and_whole_period_downtime(self):
        a = Action(repairs=(MINE,))
        states = self.model.np.array([64])
        healthy = self.model.reward(states,Action(),0,{'plant-a':True})[0]
        repairing = self.model.reward(states,a,0,{'plant-a':True})[0]
        self.assertLess(repairing,healthy)
        # Repair uncertain power now sacrifices 5.2 expected credits of
        # current production/cost, but avoids next-shift emergency losses.
        weights = self.model.np.zeros(128)
        weights[64],weights[80] = .6,.4
        repair = Action(repairs=(POWER,))
        _,q,_ = self.model.evaluate(weights,self.decision,pairs=False)
        self.assertGreater(q[repair]-q[Action()],.7)
        self.assertLess(float(weights@self.model.reward(self.model.current,repair,0,self.decision.problem.inputs)),
                        float(weights@self.model.reward(self.model.current,Action(),0,self.decision.problem.inputs)))
        altered = lookahead.Oracle(self.net,self.rates,replace(self.values,repair={'mine':9999,'power':9999,'plant':9999}))
        self.model.np.testing.assert_allclose(self.model.next_rewards[1],altered.next_rewards[1])

    def test_proposals_respect_cap_and_include_diagnostic_continuations(self):
        for cap in (1,16,32,64,256):
            calls = []
            def evaluate(ps):
                calls.extend(ps)
                return {p:p[0].units+p[1].units for p in ps}
            proposed = lookahead.propose_pairs(self.model.first,self.model.second,self.values,evaluate,cap)
            self.assertLessEqual(len(calls),cap)
            self.assertEqual(len(calls),len(set(calls)))
            self.assertEqual(set(calls),set(proposed))
            self.assertTrue(all(a in self.model.first and b in self.model.second for a,b in calls))
            if cap>=32:
                self.assertIn((Action(inspections=(MINE,)),Action(repairs=(MINE,))),calls)

    def test_myopic_is_suboptimal_in_substantial_partially_observed_rounds(self):
        fractions = []
        for seed in range(1,5):
            rows = []
            result = lookahead.run(GameConfig(seed=seed,rounds=30,stage=5,plants=1),'myopic',on_round=rows.append)
            fractions.append(result['myopic_suboptimal_fraction'])
            self.assertGreater(result['regret'],.5)
            self.assertEqual(result['lookahead_gain'],0)
            self.assertTrue(any(r['posterior_support']>1 for r in rows))
            self.assertTrue(any(not r['inputs']['plant-a'] for r in rows))
        self.assertGreaterEqual(sum(fractions)/4,.2)

    def test_prior_uses_history_frequencies_even_for_inspected_variables(self):
        prior = PriorBackend()
        captured = []
        original = lookahead.Oracle.factored
        def record(model,marginals,*args,**kwargs):
            captured.append(marginals.copy())
            return original(model,marginals,*args,**kwargs)
        with patch('supplynet.backends.PriorBackend',return_value=prior),patch.object(lookahead.Oracle,'factored',record):
            lookahead.run(GameConfig(seed=1,rounds=1,stage=5,plants=1),'prior')
        for key,value in captured[0].items():
            if key!='fuel':
                yes,total = prior.counts[key]
                self.assertEqual(value,(yes+1)/(total+2))

    def test_unavailable_pair_values_are_reported_without_summary_crash(self):
        output = {'chosen':0,'candidate_pairs':0,'context_bound':1,'decision_queries':1,'decision_answered':0,'pair_scores':[[0,0,None]],'unsupported':['Assuming']}
        child = SimpleNamespace(returncode=0,stderr='',stdout=json.dumps(output))
        with patch('supplynet.lookahead.subprocess.run',return_value=child):
            result = lookahead.run(GameConfig(seed=1,rounds=1,stage=5,plants=1),'mm2')
        self.assertEqual(result['decision_answered'],0)
        self.assertNotIn('pair_mean_error',result)

    @unittest.skipUnless(os.environ.get('SUPPLYNET_LIVE_PETTACHAINER')=='1','set SUPPLYNET_LIVE_PETTACHAINER=1')
    def test_live_lookahead_choice_and_bounded_contexts(self):
        rows = []
        result = lookahead.run(GameConfig(seed=1,rounds=2,stage=5,plants=1,budget=400),'pettachainer',python_path=os.environ.get('PETTACHAINER_PYTHONPATH'),on_round=rows.append)
        self.assertLessEqual(result['context_bound'],64)
        self.assertGreater(result['decision_answered'],0)
        self.assertEqual(len(rows[0]['pair_scores']),rows[0]['candidate_pairs'])
        self.assertTrue(all(r['posterior_support']>1 for r in rows))
        for row in rows:
            self.assertEqual(len(row['belief_errors']),8)
            self.assertEqual(row['belief_max_error'],max(e['error'] for e in row['belief_errors']))
            self.assertAlmostEqual(row['own_myopic_true_gain'],row['own_myopic_regret']-row['regret'])
            self.assertGreater(row['control_seconds'],0)
            self.assertIn('own_belief_lookahead_gain',row)

    @unittest.skipUnless(os.environ.get('SUPPLYNET_LIVE_PETTACHAINER')=='1','set SUPPLYNET_LIVE_PETTACHAINER=1')
    def test_live_known_snapshot_pairs_match_exact_sums(self):
        from dataclasses import asdict
        from supplynet.lookahead_worker import decide
        d = replace(self.decision,inspected={r.name:False for r in self.net.routes})
        for state in (64,72,80,96):  # healthy, degraded mine/power/plant
            with self.subTest(state=state):
                marginal = [['Storm','north',0],*(['Blocked',r.name,0] for r in self.net.routes),*(['Degraded',site,int(bool((state>>(3+i))&1))] for i,site in enumerate(self.cell.sites))]
                payload = {'network':asdict(self.net),'rates':asdict(self.rates),'values':asdict(self.values),'parameters':asdict(lookahead.Parameters()),'decision':asdict(d),'marginals':marginal,'fuel':1,'backend':'pettachainer','budget':400,'pair_cap':64,'beam':2,'proposer':'extend'}
                result = decide(payload)
                weights = self.model.np.zeros(128)
                weights[state] = 1
                exact,_,_ = self.model.evaluate(weights,d)
                self.assertEqual(result['decision_answered'],64)
                for a,b,value in result['pair_scores']:
                    self.assertAlmostEqual(value,exact[self.model.first[a],self.model.second[b]],places=6)

    @unittest.skipUnless(os.environ.get('SUPPLYNET_LIVE_PETTACHAINER')=='1','set SUPPLYNET_LIVE_PETTACHAINER=1')
    def test_live_shared_mine_carry_positive_not_alias(self):
        from pettachainer import PeTTaChainer
        c = PeTTaChainer()
        for head in ('Fuel','Operational'):
            c.set_complete_predicate(head)
        c.add_atoms_no_check([
            '(: d (D now) (STV 0.5 1))',
            '(: (no_inverse carry) (Implication (D now) (D later)) (CTV (STV 0.75 1) (STV 0.08 1)))',
            '(: (no_inverse fuel) (Implication (Not (D now)) (Fuel later)) (STV 1 1))',
            '(: (no_inverse operational) (Implication (Not (D later)) (Operational mine later)) (STV 1 1))',
        ])
        rows = c.query_many_refs([
            '(: $p (And (Fuel later) (Operational mine later)) $tv)',
        ],steps=400)
        self.assertTrue(rows[0])  # P(Not D now, Not D later) = 0.5 * (1 - 0.08).
        self.assertAlmostEqual(float(rows[0][0].tv[1]),.46,places=6)

    @unittest.skipUnless(os.environ.get('SUPPLYNET_LIVE_PROBLOG')=='1','set SUPPLYNET_LIVE_PROBLOG=1')
    def test_live_problog_two_stage_matches_reference(self):
        result = lookahead.run(GameConfig(seed=1,rounds=3,stage=5,plants=1),'problog')
        self.assertEqual(result['problog_answered'],19)  # 16 conditional tables + 3 posteriors
        self.assertLess(result['problog_reference_max_error'],1e-8)
        self.assertEqual(result['problog_timeouts'],0)
        self.assertLess(result['regret'],1e-8)


def brute_force_posterior(network, rates, before, periods, observations, actions=None, utility=None):
    """Enumerates every storm, block and degradation of every region, route
    and site over ``periods``, with production computed from them period by
    period. ``before`` holds the period preceding them: P(on) of each storm
    ("Storm", region) and degradation ("Degraded", site), and, with a cell, the
    fuel route's block ("Blocked", route) and the mine's production
    ("Producing", mine). With decisions, ``actions`` holds each period's
    action (the stock, the fuel's route, the repairs; ``before`` then also
    holds ("Action", None), the preceding period's), and ``utility``, when
    given, is a function of the last period's degradations, the block the
    arriving fuel met and the fuel routes' blocks whose expectation is
    returned with the posterior."""
    cell = network.cell
    sites = cell.sites if cell else {}
    late = {key: value for o in observations for key, value in o.late.items()}
    inspected = {(subject, o.period): value for o in observations for subject, value in (o.inspected | o.degraded).items()}
    reported = {(site, o.previous): value for o in observations for site, value in o.reported.items()}
    by_period = {o.period: o for o in observations}
    chained = [("Storm", region) for region in network.regions] + [("Degraded", site) for site in sites]
    variables = [(var, period) for period in periods for var in chained + [("Blocked", route.name) for route in network.routes]]
    routes = {route.name: route for route in network.routes}
    totals = {}
    norm = expected = 0.0
    for values in itertools.product((True, False), repeat=len(variables)):
        state = dict(zip(variables, values))
        weight = 1.0
        previous = dict(before)
        for period in periods:
            action, before_action = (actions[period], previous[("Action", None)]) if actions else (None, None)
            for predicate, subject in chained:
                kind = network.regions[subject] if predicate == "Storm" else sites[subject]
                rate = rates.storm_rate if predicate == "Storm" else rates.degraded_rate
                repaired = before_action is not None and subject in before_action.repairs
                p = rate(kind, 0.0 if repaired else float(previous[(predicate, subject)]))
                weight *= p if state[((predicate, subject), period)] else 1 - p
            for route in network.routes:
                blocked = state[(("Blocked", route.name), period)]
                if inspected.get((route.name, period), blocked) != blocked:
                    weight = 0.0
                p = rates.block_rate(route.kind, state[(("Storm", route.region), period)])
                weight *= p if blocked else 1 - p
                p_late = rates.late_given_blocked if blocked else rates.late_given_open
                for shipment in route.shipments:
                    if (shipment, period) in late:
                        weight *= p_late if late[(shipment, period)] else 1 - p_late
            if cell:
                degraded = {site: state[(("Degraded", site), period)] for site in sites}
                if any(inspected.get((site, period), value) != value for site, value in degraded.items()):
                    weight = 0.0
                fuel = before_action.via(cell) if before_action else cell.fuel_route.name
                fuel_blocked = previous[("Blocked", fuel)] if cell.timed else state[(("Blocked", fuel), period)]
                observation = by_period[period]
                stocked, repairs = (action.stock, action.repairs) if action else (observation.stocked, ())
                produced = production(cell, degraded, stocked, observation.inputs, fuel_blocked, previous[("Producing", MINE)], repairs)
                if (MINE, period) in reported and any(reported[(site, period)] != on for site, on in produced.items()):
                    weight = 0.0
                for site, on in produced.items():
                    state[(("Producing", site), period)] = on
            if weight == 0.0:
                break
            last = (degraded, fuel_blocked, {route.name: state[(("Blocked", route.name), period)] for route in cell.fuel_routes}) if cell else None
            previous = {var: state[(var, period)] for var in chained} | (
                {**{("Blocked", route.name): state[(("Blocked", route.name), period)] for route in cell.fuel_routes},
                 ("Producing", MINE): produced[MINE], ("Action", None): action}
                if cell
                else {}
            )
        if weight == 0.0:
            continue
        norm += weight
        if utility:
            expected += weight * utility(*last)
        for ((predicate, subject), period), value in state.items():
            if (subject, period) in inspected or (predicate == "Producing" and period != periods[-1]):
                continue
            totals[(predicate, subject, period)] = totals.get((predicate, subject, period), 0.0) + weight * value
    posterior = {key: total / norm for key, total in totals.items()}
    return (posterior, expected / norm) if utility else posterior


def linked_run(network, rates, rng, count, inspect_rate):
    """``count`` linked periods, each observed with the shipments arriving in
    it (departures from the first period on) and the previous period's
    production."""
    periods, observations = [], []
    for index in range(count):
        period = sample_period(network, rates, rng, f"t{index}", periods)
        periods.append(period)
        arrivals = {
            (shipment, periods[index - route.transit].name): periods[index - route.transit].late[shipment]
            for route in network.routes
            if index - route.transit >= 0
            for shipment in route.shipments
        }
        observations.append(observe(network, period, rng, inspect_rate, arrivals, periods[-2].producing if index else {}))
    return periods, observations


def cell_network(cycle):
    """One region, one route carrying a plant's input, and the fuel route."""
    fuel = Route("rf", "north", "exposed", (), 1 if cycle == "timed" else 0)
    return Network({"north": "coastal"}, (Route("r1", "north", "sheltered", ("s1-1",), 1), fuel), Cell(fuel, {"plant-a": "s1-1"}))


def decision_network():
    """Stage 4's cell in one region: a plant's input route, the exposed fuel
    route and its sheltered alternate."""
    fuel, alternate = Route("rf", "north", "exposed", (), 1), Route("ra", "north", "sheltered", (), 1)
    return Network({"north": "coastal"}, (Route("r1", "north", "sheltered", ("s1-1",), 1), fuel, alternate), Cell(fuel, {"plant-a": "s1-1"}, alternate))


class WorldTests(unittest.TestCase):
    def assert_matches_brute_force(self, knowledge, brute):
        exact = knowledge.posterior()
        self.assertTrue(exact)
        for key, value in exact.items():
            self.assertAlmostEqual(value, brute[key], places=9, msg=key)

    def test_stage1_exact_posterior_matches_brute_force(self):
        rates = Rates()
        rng = random.Random(3)
        network = generate_network(rng, 2)
        for index in range(20):
            period = sample_period(network, rates, rng, f"t{index}", [])
            observation = observe(network, period, rng, 0.3, {(s, period.name): v for s, v in period.late.items()})
            knowledge = Knowledge(network, rates.nodes(network))
            knowledge.observe(observation)
            before = {("Storm", region): rates.stationary(kind) for region, kind in network.regions.items()}
            self.assert_matches_brute_force(knowledge, brute_force_posterior(network, rates, before, [period.name], [observation]))

    def test_stage2_exact_posterior_matches_brute_force(self):
        rates = stage_rates(2)
        network = Network(
            {"north": "coastal"},
            (Route("r1", "north", "exposed", ("s1-1", "s1-2"), 1), Route("r2", "north", "sheltered", ("s2-1",), 3)),
        )
        for seed in range(3):
            periods, observations = linked_run(network, rates, random.Random(seed), 6, 0.3)
            names = [period.name for period in periods]
            knowledge = Knowledge(network, rates.nodes(network))
            for observation in observations:
                knowledge.observe(observation)
            before = {("Storm", "north"): rates.stationary("coastal")}
            self.assert_matches_brute_force(knowledge, brute_force_posterior(network, rates, before, names, observations))
            # Resolving the first period conditions the rest on its storm.
            knowledge.resolve(periods[0])
            before = {("Storm", "north"): periods[0].storms["north"]}
            self.assert_matches_brute_force(knowledge, brute_force_posterior(network, rates, before, names[1:], observations))

    def test_stage3_exact_posterior_matches_brute_force(self):
        rates = stage_rates(3)
        for cycle in ("timed", "untimed"):
            network = cell_network(cycle)
            for seed in range(2):
                periods, observations = linked_run(network, rates, random.Random(seed), 3, 0.3)
                names = [period.name for period in periods]
                knowledge = Knowledge(network, rates.nodes(network))
                for observation in observations:
                    knowledge.observe(observation)
                before = {("Storm", "north"): rates.stationary("coastal"), ("Blocked", "rf"): False, ("Producing", MINE): False}
                before |= {("Degraded", site): rates.degraded_stationary(kind) for site, kind in network.cell.sites.items()}
                self.assert_matches_brute_force(knowledge, brute_force_posterior(network, rates, before, names, observations))
                # Resolving the first period carries its labels, the fuel
                # route's block and the mine's production included.
                knowledge.resolve(periods[0])
                first = periods[0]
                before = {("Storm", "north"): first.storms["north"], ("Blocked", "rf"): first.blocked["rf"], ("Producing", MINE): first.producing[MINE]}
                before |= {("Degraded", site): value for site, value in first.degraded.items()}
                self.assert_matches_brute_force(knowledge, brute_force_posterior(network, rates, before, names[1:], observations))

    def test_stage4_exact_posterior_and_utilities_match_brute_force(self):
        """After an action (a repair restarts a site healthy, a reroute sends
        the fuel over the alternate route, the stock is decided), the next
        period's posterior and every option's expected utility match
        enumeration over every storm, block and degradation."""
        rates, network = stage_rates(4), decision_network()
        cell = network.cell
        choices = options(cell, 2)
        for seed, first in ((0, Action(stock=True, repairs=(POWER,))), (1, Action(reroute=True, repairs=(MINE,))), (2, Action())):
            rng = random.Random(seed)
            periods, observations = [], []
            for index in range(2):
                period = sample_period(network, rates, rng, f"t{index}", periods)
                if index == 0:
                    period = enact(network, period, None, first)
                periods.append(period)
                arrivals = {("s1-1", "t0"): periods[0].late["s1-1"]} if index else {}
                observations.append(observe(network, period, rng, 0.5, arrivals, periods[0].producing if index else {}))
            knowledge = Knowledge(network, rates.nodes(network))
            knowledge.observe(observations[0])
            knowledge.act("t0", first)
            knowledge.observe(observations[1])
            before = {("Storm", "north"): rates.stationary("coastal"), ("Blocked", "rf"): False, ("Blocked", "ra"): False,
                      ("Producing", MINE): False, ("Action", None): None}
            before |= {("Degraded", site): rates.degraded_stationary(kind) for site, kind in cell.sites.items()}
            problem = Problem("t1", "t0", periods[1].inputs, periods[0].producing[MINE], periods[0].via, choices, Values())
            exact = expected_utilities(knowledge, problem)
            for action in choices[:: 3]:
                brute, expected = brute_force_posterior(network, rates, before, ["t0", "t1"], observations, {"t0": first, "t1": None},
                                                        lambda d, f, b: utility(cell, problem, action, d, f, b))
                self.assertAlmostEqual(exact[action], expected, places=9, msg=action)
            self.assert_matches_brute_force(knowledge, brute)

    def test_production_is_the_least_fixed_point(self):
        cell = cell_network("untimed").cell
        healthy = dict.fromkeys(cell.sites, False)
        supplied = {"plant-a": True}
        # Without stock, everything running is a fixed point of the untimed
        # rules too; the least one has nothing running.
        self.assertEqual(production(cell, healthy, False, supplied, False, True), dict.fromkeys(cell.sites, False))
        self.assertEqual(production(cell, healthy, True, supplied, False, False), dict.fromkeys(cell.sites, True))
        self.assertEqual(production(cell, healthy | {POWER: True}, True, supplied, False, False), dict.fromkeys(cell.sites, False))
        self.assertEqual(
            production(cell, healthy | {MINE: True}, True, {"plant-a": False}, False, False),
            {MINE: False, POWER: True, "plant-a": False},
        )
        # Timed, last period's fuel keeps the loop running over an open route.
        timed = cell_network("timed").cell
        self.assertEqual(production(timed, healthy, False, supplied, False, True), dict.fromkeys(cell.sites, True))
        self.assertEqual(production(timed, healthy, False, supplied, True, True), dict.fromkeys(cell.sites, False))
        self.assertEqual(production(timed, healthy, False, supplied, False, False), dict.fromkeys(cell.sites, False))

    def test_without_stock_the_untimed_loop_does_not_run(self):
        rates = stage_rates(3)
        network = cell_network("untimed")
        periods, observations = linked_run(network, rates, random.Random(5), 30, 0.3)
        knowledge = Knowledge(network, rates.nodes(network))
        for period, observation in zip(periods, observations):
            knowledge.observe(observation)
            posterior = knowledge.posterior()
            if not period.stocked:
                self.assertFalse(any(period.producing.values()))
                self.assertTrue(all(posterior[("Producing", site, period.name)] == 0.0 for site in network.cell.sites))
            knowledge.resolve(period)
        self.assertTrue(any(period.producing[MINE] for period in periods))

    def test_a_late_shipment_raises_its_region_and_an_inspected_clear_route_lowers_it(self):
        rates = Rates()
        network = generate_network(random.Random(1), 1)
        region = next(iter(network.regions))
        quiet = {(s, "t"): False for route in network.routes for s in route.shipments}
        first = network.routes[0]
        late = quiet | {(first.shipments[0], "t"): True}

        def storm(late, inspected):
            knowledge = Knowledge(network, rates.nodes(network))
            knowledge.observe(Observation("t", None, late, inspected))
            return knowledge.posterior()[("Storm", region, "t")]

        base, raised, cleared = storm(quiet, {}), storm(late, {}), storm(late, {first.name: False})
        self.assertGreater(raised, base)
        self.assertLess(cleared, raised)

    def test_a_late_arrival_raises_its_departure_storm_and_the_storm_after_it(self):
        rates = stage_rates(2)
        network = Network({"north": "coastal"}, (Route("r1", "north", "exposed", ("s1-1",), 1),))

        def posterior(arrivals):
            knowledge = Knowledge(network, rates.nodes(network))
            knowledge.observe(Observation("t1", None, {}, {}))
            knowledge.observe(Observation("t2", "t1", arrivals, {}))
            return knowledge.posterior()

        quiet, late = posterior({("s1-1", "t1"): False}), posterior({("s1-1", "t1"): True})
        self.assertGreater(late[("Storm", "north", "t1")], quiet[("Storm", "north", "t1")])
        self.assertGreater(late[("Storm", "north", "t2")], quiet[("Storm", "north", "t2")])


class DecisionTests(unittest.TestCase):
    def test_options_are_the_joint_actions_within_capacity(self):
        cell = generate_network(random.Random(1), 3, 3, "timed", 2, decisions=True).cell
        self.assertEqual([len(options(cell, capacity)) for capacity in (1, 2, 3)], [7, 22, 42])
        self.assertTrue(all(action.units <= 2 for action in options(cell, 2)))
        # Stage 4's network is stage 3's with the alternate route, and the fuel route exposed.
        stage3 = generate_network(random.Random(1), 3, 3, "timed")
        self.assertEqual(cell.inputs, stage3.cell.inputs)
        self.assertEqual((cell.fuel_route.kind, cell.alternate.kind, cell.alternate.region), ("exposed", "sheltered", cell.fuel_route.region))

    def test_the_best_joint_action_is_not_the_best_singles(self):
        """No fuel arrives (the mine stopped), so only a stockpile runs the
        cell; a storm makes the exposed fuel route likely blocked. Rerouting
        alone gains nothing, since the mine does not run; with the stockpile
        it saves the fuel sent on."""
        cell = decision_network().cell
        problem = Problem("t1", "t0", {"plant-a": True}, False, "rf", options(cell, 2), Values())
        marginals = {("Degraded", site, "t1"): 0.0 for site in cell.sites} | {("Blocked", "rf", "t1"): 0.8, ("Blocked", "ra", "t1"): 0.4,
                                                                               ("Blocked", "rf", "t0"): 0.0}
        utilities = factored_utilities(cell, problem, marginals)
        self.assertEqual(greedy(utilities, 2), Action(stock=True))
        self.assertEqual(best(utilities), Action(stock=True, reroute=True))
        self.assertAlmostEqual(utilities[Action(stock=True, reroute=True)] - utilities[Action(stock=True)], 8 * 0.4 - 1)


class LearnedRatesTests(unittest.TestCase):
    def test_counts_and_laplace(self):
        """Persistence samples are consecutive labelled periods; a block's
        parent is its region's storm and a lateness's its route's block."""
        route = Route("r1", "north", "exposed", ("s1-1",))
        network = Network({"north": "coastal"}, (route,))
        periods = [
            Period(name, previous, {"north": storm}, {"r1": blocked}, {"s1-1": late})
            for name, previous, storm, blocked, late in (
                ("h1", None, True, True, True),
                ("h2", "h1", True, False, False),
                ("h3", "h2", False, False, True),
                ("h4", "h3", False, False, False),
            )
        ]
        tally = counts(network, periods, linked=True)
        self.assertEqual(tally[("Storm", "north")], ((1, 2), (0, 1)))
        self.assertEqual(tally[("Blocked", "r1")], ((1, 2), (0, 2)))
        self.assertEqual(tally[("Late", "s1-1")], ((1, 1), (1, 3)))
        independent = counts(network, [replace(period, previous=None) for period in periods], linked=False)
        self.assertEqual(independent[("Storm", "north")], ((2, 4), (2, 4)))
        rates = learned_rates(network, periods, linked=True)
        self.assertEqual(rates.table[("Storm", "north")], (2 / 4, 1 / 3))
        self.assertEqual(rates.table[("Late", "s1-1")], (2 / 3, 2 / 5))

    def test_the_learned_reference_converges_to_the_exact_posterior(self):
        """With a long labelled history the counted rates approach the true
        ones, and so does its posterior."""
        short, long = (
            run_game(GameConfig(seed=5, rounds=5, stage=2, history=history), LearnedReferenceBackend())["posterior_error"]
            for history in (30, 3000)
        )
        self.assertLess(long, short)
        self.assertLess(long, 0.01)


class GameTests(unittest.TestCase):
    def test_reference_matches_the_posterior_and_beats_the_prior(self):
        kinds = {"storm", "blocked"}
        for stage, cycle, extra in (
            (1, "timed", set()),
            (2, "timed", {"previous_storm"}),
            (3, "timed", {"previous_storm", "degraded", "producing", "producing_unstocked"}),
            (3, "untimed", {"previous_storm", "degraded", "producing", "producing_unstocked"}),
        ):
            config = GameConfig(seed=7, rounds=10, stage=stage, cycle=cycle)
            reference = run_game(config, ReferenceBackend())
            prior = run_game(config, PriorBackend())
            self.assertEqual(reference["posterior_error"], 0.0)
            self.assertLess(reference["brier"], prior["brier"])
            self.assertEqual(reference["coverage"], 1.0)
            self.assertEqual(set(reference["kinds"]), kinds | extra)
            for kind in extra & {"degraded", "producing"}:
                self.assertLess(reference["kinds"][kind]["brier"], prior["kinds"][kind]["brier"])

    def test_stage4_reference_has_no_regret_and_the_interaction_matters(self):
        config = GameConfig(seed=7, rounds=15, stage=4)
        reference, greedy_run, prior = (run_game(config, backend) for backend in (ReferenceBackend(), GreedyBackend(), PriorBackend()))
        self.assertEqual((reference["regret"], reference["agreement"]), (0.0, 1.0))
        self.assertEqual(greedy_run["posterior_error"], 0.0)  # exact beliefs; its actions, and so its world, differ
        self.assertGreater(greedy_run["regret"], 0.0)
        self.assertGreater(prior["regret"], greedy_run["regret"])
        self.assertEqual(set(reference["kinds"]), {"storm", "previous_storm", "blocked", "degraded"})

    def test_runs_are_deterministic(self):
        for stage, cycle in ((1, "timed"), (2, "timed"), (3, "timed"), (3, "untimed"), (4, "timed")):
            config = GameConfig(seed=11, rounds=5, stage=stage, cycle=cycle)
            first, second = (
                {field: value for field, value in run_game(config, ReferenceBackend()).items() if "seconds" not in field}
                for _ in range(2)
            )
            self.assertEqual(first, second)
        # Stage 3 adds the cell after stage 2's network, which it keeps.
        stage2, stage3 = generate_network(random.Random(4), 3, 3), generate_network(random.Random(4), 3, 3, "timed")
        self.assertEqual(stage3.routes[:-1], stage2.routes)

    def test_statements(self):
        rates = stage_rates(2)
        network = generate_network(random.Random(2), 1, 3)
        route = network.routes[0]
        region, kind = next(iter(network.regions.items()))
        lines = metta.rules(network, rates)
        self.assertIn(
            f"(: block-{route.name} (Implication (Storm {route.region} $p) (Blocked {route.name} $p)) "
            f"(CTV (STV {rates.block_given_storm[route.kind]:.6g} 1) (STV 0.03 1)))",
            lines,
        )
        self.assertIn(
            f"(: persist-{region} (Implication (And (NextPeriod $p $t) (Storm {region} $p)) (Storm {region} $t)) "
            f"(CTV (STV 0.7 1) (STV {rates.storm[kind]:.6g} 1)))",
            lines,
        )
        self.assertFalse(any("persist" in line for line in metta.rules(network, Rates())))
        observation = Observation("t3", "t2", {("s1-1", "t1"): True}, {})
        self.assertEqual(
            metta.observation_facts(observation),
            ["(: next-t3 (NextPeriod t2 t3) (STV 1 1))", "(: late-s1-1-t1 (Late s1-1 t1) (STV 1 1))"],
        )
        self.assertEqual(metta.query(("Storm", "north", "t3")), "(: $prf (Storm north t3) $tv)")

    def test_learned_statements_leave_every_rate_but_the_certain_rules_to_the_data(self):
        network, rates = cell_network("timed"), stage_rates(3)
        given, learned = metta.rules(network, rates), metta.rules(network, rates, learned=True)
        self.assertEqual(len(given), len(learned))
        for stated, open_ in zip(given, learned, strict=True):
            if "(CTV (STV 1 1) (STV 0 1))" in stated:  # the production rules
                self.assertEqual(open_, stated)
                self.assertIsNone(metta.review(open_))
            else:
                self.assertTrue(open_.endswith(f" {metta.PRIOR})"))
                self.assertEqual(open_.split(" (CTV")[0], stated.split(" (CTV")[0])
        self.assertIn(
            "(: $prf (RuleTruth (Implication (And (NextPeriod $p $t) (Degraded mine $p)) (Degraded mine $t))) $tv)",
            [metta.review(line) for line in learned],
        )

    def test_cycle_statements(self):
        rates = stage_rates(3)
        timed, untimed = (metta.rules(cell_network(cycle), rates) for cycle in ("timed", "untimed"))
        self.assertEqual([line for line in timed if line not in untimed], [
            "(: fuel-arrives (Implication (And (NextPeriod $p $t) (Producing mine $p) (Not (Blocked rf $p))) (FuelArrived $t)) "
            "(CTV (STV 1 1) (STV 0 1)))"
        ])
        self.assertIn(
            "(: fuel-arrives (Implication (And (Producing mine $t) (Not (Blocked rf $t))) (FuelArrived $t)) (CTV (STV 1 1) (STV 0 1)))",
            untimed,
        )
        self.assertIn(
            "(: persist-degraded-mine (Implication (And (NextPeriod $p $t) (Degraded mine $p)) (Degraded mine $t)) "
            "(CTV (STV 0.75 1) (STV 0.08 1)))",
            timed,
        )
        self.assertIn(
            "(: runs-plant-a (Implication (And (Producing power-plant $t) (InputArrived plant-a $t) (Not (Degraded plant-a $t))) "
            "(Producing plant-a $t)) (CTV (STV 1 1) (STV 0 1)))",
            timed,
        )
        self.assertEqual(metta.complete_predicates(cell_network("untimed")), ["Producing", "Fuelled", "FuelArrived"])
        self.assertEqual(metta.complete_predicates(cell_network("timed")), ["Producing", "Fuelled", "FuelArrived"])
        self.assertEqual(metta.positive_only_predicates(cell_network("timed")), [])
        observation = Observation("t3", "t2", {}, {}, {"mine": True}, False, {"plant-a": True}, {"mine": False})
        self.assertEqual(
            metta.observation_facts(observation),
            [
                "(: next-t3 (NextPeriod t2 t3) (STV 1 1))",
                "(: degraded-mine-t3 (Degraded mine t3) (STV 1 1))",
                "(: stocked-t3 (StockedFuel t3) (STV 0 1))",
                "(: input-plant-a-t3 (InputArrived plant-a t3) (STV 1 1))",
                "(: producing-mine-t2 (Producing mine t2) (STV 0 1))",
            ],
        )
        soft = Observation("t3", "t2", {}, {}, {"mine": True}, False, {}, {"mine": False}, report_confidence=0.9)
        self.assertEqual(metta.observation_facts(soft)[1:], ["(: degraded-mine-t3 (Degraded mine t3) (STV 1 1))", "(: stocked-t3 (StockedFuel t3) (STV 0 1))",
                                                            "(: producing-mine-t2 (Producing mine t2) (STV 0 0.9))"])

    def test_decision_statements(self):
        network = decision_network()
        rates = stage_rates(4)
        semantic, chainer = metta.rules(network, rates), metta.rules(network, rates, positive_actions=True)
        self.assertIn("(: fuelled (Implication (Or (StockedFuel $t) (FuelArrived $t)) (Fuelled power-plant $t)) (CTV (STV 1 1) (STV 0 1)))", semantic)
        self.assertIn(
            "(: runs-mine (Implication (And (Producing power-plant $t) (Not (Degraded mine $t)) (InService mine $t)) "
            "(Producing mine $t)) (CTV (STV 1 1) (STV 0 1)))",
            chainer,
        )
        self.assertIn(
            "(: persist-degraded-mine (Implication (And (NextPeriod $p $t) (Degraded mine $p) (Not (Repairing mine $p))) (Degraded mine $t)) "
            "(CTV (STV 0.75 1) (STV 0.08 1)))",
            chainer,
        )
        self.assertEqual(metta.positive_only_predicates(network), ["FuelVia", "StockedFuel", "Unstocked"])
        action = Action(reroute=True, repairs=(MINE,))
        self.assertEqual(
            metta.action_facts(network, "t3", action),
            [
                "(: via-t3 (FuelVia ra t3) (STV 1 1))",
                "(: unstocked-t3 (Unstocked t3) (STV 1 1))",
                "(: repairing-mine-t3 (Repairing mine t3) (STV 1 1))",
                "(: repairing-power-plant-t3 (Repairing power-plant t3) (STV 0 1))",
                "(: repairing-plant-a-t3 (Repairing plant-a t3) (STV 0 1))",
                "(: in-service-power-plant-t3 (InService power-plant t3) (STV 1 1))",
                "(: in-service-plant-a-t3 (InService plant-a t3) (STV 1 1))",
            ],
        )
        self.assertEqual(
            metta.decision_query(network, "t3", action),
            "(: $prf (Assuming (And (FuelVia ra t3) (Unstocked t3) (Repairing mine t3) (InService power-plant t3) (InService plant-a t3)) "
            "(Outcome t3 $o $v)) $tv)",
        )

    @unittest.skipUnless(os.environ.get("SUPPLYNET_LIVE_PETTACHAINER") == "1", "set SUPPLYNET_LIVE_PETTACHAINER=1")
    def test_pettachainer_decides_with_assuming_queries(self):
        """Round 1 of seed 1: every joint action asked is answered, and the
        choice is the reference's (its values match the exact ones there)."""
        rounds = []
        summary = run_game(GameConfig(seed=1, rounds=2, budget=200, stage=4), PeTTaChainerBackend(os.environ.get("PETTACHAINER_PYTHONPATH")),
                           on_round=rounds.append)
        self.assertEqual(summary["coverage"], 1.0)
        self.assertEqual([r["decision_answered"] for r in rounds], [r["decision_queries"] for r in rounds])
        self.assertEqual(rounds[0]["regret"], 0.0)

    @unittest.skipUnless(os.environ.get("SUPPLYNET_LIVE_PETTACHAINER") == "1", "set SUPPLYNET_LIVE_PETTACHAINER=1")
    def test_pettachainer_answers_every_belief(self):
        for stage in (1, 2, 3):
            backend = PeTTaChainerBackend(os.environ.get("PETTACHAINER_PYTHONPATH"))
            summary = run_game(GameConfig(seed=7, rounds=3, budget=50, stage=stage), backend)
            self.assertEqual(summary["coverage"], 1.0)

    def test_pettachainer_records_review_stvs_and_refreshes_after_resolution(self):
        handler = SimpleNamespace(
            set_evidence_confidence_k=Mock(), set_rule_refinement=Mock(),
            add_atoms_no_check=Mock(), query_many_refs=Mock(),
        )
        module = SimpleNamespace(PeTTaChainer=lambda: handler)
        with patch("supplynet.backends.importlib.import_module", return_value=module):
            backend = PeTTaChainerBackend(learned=True, review_steps=123)
        network, rates = generate_network(random.Random(2), 1, 1), stage_rates(1)
        backend.begin(network, rates, [])
        self.assertTrue(backend.reviews)
        observation = Observation("t1", None, {}, {})
        handler.query_many_refs.side_effect = lambda goals, steps: [
            [SimpleNamespace(tv=("STV", 0.123456, 0.234567))] for _ in goals
        ]
        backend.beliefs(observation, [], budget=50)
        self.assertEqual(handler.query_many_refs.call_args_list,
                         [call([goal], steps=123) for goal in backend.reviews])
        self.assertEqual(backend.summary()["rule_truths"],
                         {goal: [0.1235, 0.2346] for goal in backend.reviews})

        backend.resolve(sample_period(network, rates, random.Random(3), "t1", []), observation)
        handler.query_many_refs.reset_mock()
        handler.query_many_refs.side_effect = lambda goals, steps: [
            [SimpleNamespace(tv=("STV", 0.345678, 0.456789))] for _ in goals
        ]
        backend.beliefs(Observation("t2", "t1", {}, {}), [], budget=50)
        handler.query_many_refs.assert_called_once_with(backend.reviews, steps=0)
        self.assertEqual(backend.summary()["rule_truths"],
                         {goal: [0.3457, 0.4568] for goal in backend.reviews})

    @unittest.skipUnless(os.environ.get("SUPPLYNET_LIVE_PETTACHAINER") == "1", "set SUPPLYNET_LIVE_PETTACHAINER=1")
    def test_pettachainer_learns_every_rule(self):
        """With learned rules every rule's review folds its instances: no
        rule keeps its prior (in this game every rule's antecedent holds in
        some labelled period; one that never holds would keep it)."""
        backend = PeTTaChainerBackend(os.environ.get("PETTACHAINER_PYTHONPATH"), learned=True)
        summary = run_game(GameConfig(seed=1, rounds=2, budget=50, stage=3), backend)
        self.assertEqual(summary["coverage"], 1.0)
        self.assertEqual(len(summary["rule_truths"]), len(backend.reviews))
        self.assertEqual(set(summary["rule_truths"]), set(backend.reviews))
        self.assertTrue(all(len(truth) == 2 for truth in summary["rule_truths"].values()))
        self.assertFalse([goal for goal, truth in summary["rule_truths"].items() if truth[1] <= 0.02])

    @unittest.skipUnless(os.environ.get("SUPPLYNET_LIVE_PETTACHAINER") == "1", "set SUPPLYNET_LIVE_PETTACHAINER=1")
    def test_pettachainer_does_not_run_the_untimed_loop_without_stock(self):
        # Seed 1's first rounds have no fuel stock, so the exact production is
        # 0; with the cycle's predicates complete the chainer answers it.
        backend = PeTTaChainerBackend(os.environ.get("PETTACHAINER_PYTHONPATH"))
        summary = run_game(GameConfig(seed=1, rounds=3, budget=50, stage=3, cycle="untimed"), backend)
        unstocked = summary["kinds"]["producing_unstocked"]
        self.assertEqual(unstocked["posterior_brier"], 0.0)
        self.assertEqual(unstocked["coverage"], 1.0)
        self.assertLess(unstocked["posterior_error"], 0.01)


if __name__ == "__main__":
    unittest.main()
