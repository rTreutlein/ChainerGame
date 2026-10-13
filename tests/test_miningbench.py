import json
import os
from dataclasses import replace
from itertools import product
from pathlib import Path
import tempfile
import unittest

from miningbench.world import Config, World, Literal, Rule, serial_sample
from miningbench.scoring import losses, normal_form, recovery
from miningbench.systems import Marginal
from miningbench.runner import run
from miningbench.trace import audit
from miningbench.petta import parse, decode


class MiningBenchTest(unittest.TestCase):
    def test_seed_prefix_and_distractor_independence(self):
        a = list(World(Config(samples=32)).stream())
        b = list(World(Config(samples=64, distractors=100, targets=2)).stream())
        self.assertEqual([s['q'] for s in a], [{t:q for t,q in s['q'].items() if t.startswith('T0_')} for s in b[:32]])
        self.assertEqual([s['y'] for s in a], [{t:q for t,q in s['y'].items() if t.startswith('T0_')} for s in b[:32]])
        self.assertEqual(serial_sample(a[0]), serial_sample(next(World(Config(samples=32)).stream())))

    def test_exact_noisy_or_overlap_and_observation_channel(self):
        w = World(Config(kinds=('or',), strength=.7, observation_noise=.2))
        row = {(f'T0_or_F{i}', ()): 1 for i in range(4)}
        q = w.probabilities([row]*3, 0)['T0_or'][1]
        self.assertAlmostEqual(q, .2+.6*(1-.3**2))
        self.assertAlmostEqual(sum(w.probabilities([row]*3, 0)['T0_or']), 1)

    def test_xor_has_no_marginal_information(self):
        w = World(Config(kinds=('xor',)))
        cells = []
        for a,b in product((0,1), repeat=2):
            row = {('T0_xor_F0', ()): a, ('T0_xor_F1', ()): b}
            p = w.probabilities([row]*3,0)['T0_xor'][1]
            self.assertEqual(p, a ^ b)
            cells.append((a,b,p))
        for i in (0,1):
            for side in (0,1):
                self.assertEqual(sum(c[2] for c in cells if c[i] == side)/2, .5)

    def test_binding_requires_shared_entity(self):
        w = World(Config(kinds=('binding',)))
        row = {('T0_binding_At', ('e0',)): 1, ('T0_binding_Did', ('e1',)): 1}
        self.assertEqual(w.probabilities([row]*3,0)['T0_binding'][1],0)
        row[('T0_binding_Did', ('e0',))] = 1
        self.assertEqual(w.probabilities([row]*3,0)['T0_binding'][1],1)

    def test_numeric_boundary_exception(self):
        w = World(Config(kinds=('numeric',)))
        samples = list(w.stream())
        for s in samples:
            x1,x2 = [next(int(a[0]) for (h,a),v in s['history'][-lag].items() if v) for lag in (1,2)]
            expected = x2 if x1 in (0,4) else 2*x1-x2
            self.assertEqual(s['q']['T0_numeric'][expected], 1)
        self.assertTrue(any(s['q']['T0_numeric'][1] for s in samples))

    def test_equivalence_absorption_consensus_alpha_and_contradiction(self):
        a,b = Literal('A'),Literal('B')
        pos,neg = Rule('Y',(a,b)),Rule('Y',(a,replace(b,negative=True)))
        self.assertEqual(normal_form([pos,neg]),normal_form([Rule('Y',(a,))]))
        self.assertEqual(normal_form([Rule('Y',(a,)),pos]),normal_form([Rule('Y',(a,))]))
        self.assertEqual(normal_form([Rule('Y',(a,replace(a,negative=True)))]),set())
        r = Rule('Y',(Literal('A',('$x',)),Literal('B',('$x',))))
        s = Rule('Y',(Literal('B',('$z',)),Literal('A',('$z',))))
        self.assertEqual(normal_form([r]), normal_form([s]))
        different = Rule('Y',(Literal('A',('$x',)),Literal('B',('$y',))))
        self.assertNotEqual(normal_form([r]),normal_form([different]))

    def test_numeric_formula_algebra_and_all_boolean_ground_truth(self):
        a = Rule('X',(Literal('wall'),), '*', '2*x1-x2')
        b = replace(a,formula='x1+x1-x2')
        self.assertEqual(normal_form([a]),normal_form([b]))
        for k in ('single','and2','and3','and4','negation','or'):
            w = World(Config(kinds=(k,)))
            for bits in product((0,1),repeat=4):
                row = {(f'T0_{k}_F{i}',()):v for i,v in enumerate(bits)}
                expected = {'single':bits[0], 'and2':all(bits[:2]), 'and3':all(bits[:3]),
                            'and4':all(bits), 'negation':bits[0] and not bits[1],
                            'or':all(bits[:2]) or all(bits[2:])}[k]
                self.assertEqual(w.probabilities([row]*3,0)[f'T0_{k}'][1],expected)

    def test_oracle_scores_distribution_not_sample(self):
        scores = losses([.3,.7],[.3,.7])
        self.assertEqual(scores['brier'],0)
        self.assertAlmostEqual(scores['excess_log_loss'],0)
        self.assertGreater(scores['log_loss'],0)
        self.assertGreater(scores['expected_brier'],0)
        with self.assertRaises(ValueError):
            losses([.7,.7],[.5,.5])

    def test_drift_and_oracle_recovery(self):
        w = World(Config(samples=32,drift=True))
        self.assertNotEqual(normal_form(w.rules[0]),normal_form(w.rules[1]))
        self.assertEqual([s['phase'] for s in w.stream()], [0]*16+[1]*16)
        self.assertEqual(recovery(w.rules[1],w.rules[1],w.types)['all']['recall'],1)

    def test_baseline_cannot_recover_interactions_and_work_is_bounded(self):
        w = World(Config(samples=512,kinds=('xor','and3')))
        m = Marginal(w.features,w.domains)
        work = []
        for s in w.stream():
            m.predict(s['history'])
            m.learn(s['history'],s['y'])
            work.append(m.work)
        self.assertEqual(len(set(work)),1)
        self.assertEqual(recovery(m.rules(),w.rules[0],w.types)['all']['recall'],0)

    def test_trace_never_treats_absent_telemetry_as_pass(self):
        self.assertEqual(audit([])['decisions']['queue']['status'],'not_reported')
        e = {'event':'budget','active_targets':1,'beam':4,'candidate_updates':5,'pair_updates':0,'new_candidates':0}
        self.assertEqual(len(audit([e])['violations']),1)

    def test_petta_parser_lag_and_equality(self):
        r = decode(parse('(Implication (And (NextPeriod $u $t) (At $x $u) (Did $x $u)) (Y $t))'),{'Y':('0','1')})
        self.assertEqual({l.lag for l in r.body},{2})
        self.assertEqual({l.args for l in r.body},{('$x',)})

    def test_prequential_reproducibility_and_raw_artifacts(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as d:
            a = run(Config(samples=16,kinds=('xor',)), 'marginal', Path(d)/'a')
            b = run(Config(samples=16,kinds=('xor',)), 'marginal', Path(d)/'b')
            self.assertEqual(a['data_sha256'],b['data_sha256'])
            self.assertEqual(a['scores'],b['scores'])
            rows = [json.loads(line) for line in (Path(d)/'a'/'steps.jsonl').read_text().splitlines()]
            self.assertEqual(rows[0]['predicted']['T0_xor'],[.5,.5])
            self.assertEqual(len(rows),16)


@unittest.skipUnless(os.environ.get('MININGBENCH_LIVE_PETTACHAINER') == '1',
                     'set MININGBENCH_LIVE_PETTACHAINER=1 inside capped')
class LiveMiningTest(unittest.TestCase):
    def test_live_miner_learns_without_planted_rules(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as d:
            result = run(Config(samples=16,kinds=('single',)), 'petta', d, budget=40)
            self.assertEqual(result['metadata']['petta_commit'][:8], '4e6b059b')
            self.assertTrue(result['petta']['last_verdicts'])
            self.assertFalse(result['petta']['unparsed'])
            self.assertGreater(result['petta']['anticipated_negative_count'],0)


class DirectedContractTest(unittest.TestCase):
    def test_requests_have_all_signals_and_decision_irrelevance(self):
        from miningbench.trace import target_requests
        domains = {t:('0','1') for t in ('A','B','C')}
        requests = target_requests({'A':[.99,.01]}, {'A':1}, domains, {'A'})
        self.assertEqual({r['signal'] for r in requests},{'gap','low_confidence','surprise'})
        self.assertTrue(all(r['priority'] == 0 for r in requests if r['target'] == 'C'))
        self.assertGreater(next(r['weight'] for r in requests if r['target']=='A' and r['signal']=='surprise'),4)

    def test_trace_numeric_pair_promotion_and_generalization_checks(self):
        events = [
            {'event':'template','candidate':'c','template':'linear','role':'formula','bound_variables':['x'],
             'used_variables':['x','y'],'learned_constant':2,'constant_from_data':False},
            {'event':'pair','candidate':'c','cells':[[8,0],[0,8],[0,8],[8,0]],
             'singles_stalled':True,'interaction_score':10},
            {'event':'soft_sample','candidate':'c','strength':.49,'positive_mass':0,'negative_mass':1},
            {'event':'promote','candidate':'c','support':20,'parent_log_loss':.2,'child_log_loss':.3,
             'complexity_cost':.01,'confident':True},
            {'event':'generalize','parents':['a','b'],'candidate':'c','skeletons':['AB','AC'],
             'variables':['x','y'],'variable_cap':1,'pooled_cases':[1,1]},
        ]
        report = audit(events)
        self.assertEqual(report['violation_count'],6)
        self.assertEqual(report['means']['pair_interaction_contrast'],2)


class EvidenceIntegrityTest(unittest.TestCase):
    def test_run_refuses_to_overwrite_evidence(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as d:
            run(Config(samples=8,kinds=('single',)), 'oracle', d)
            original = (Path(d)/'result.json').read_bytes()
            with self.assertRaises(FileExistsError):
                run(Config(samples=8,kinds=('xor',)), 'oracle', d)
            self.assertEqual((Path(d)/'result.json').read_bytes(),original)

    def test_report_exposes_timeouts_without_fabricating_scores(self):
        from miningbench.report import report
        text, checks = report([], [{'level':'targets','system':'petta','kinds':['numeric'],
                                  'seed':1,'returncode':124,'completed_steps':7}])
        self.assertIn('Incomplete runs',text)
        self.assertIn('| targets | petta | numeric | 1 | 124 | 7 |',text)
        self.assertNotIn('PASS',text)


class RuleExtractionTest(unittest.TestCase):
    def test_petta_weak_positive_association_and_complement(self):
        from miningbench.petta import Petta
        a = Petta.__new__(Petta)
        a.domains = {'Y':('0','1')}
        a.label_counts = {'Y':[80,20]}
        a.live = {'h':Rule('Y',(Literal('A'),),strength=.3)}
        a.negative_rates = {'h':.1}
        self.assertEqual(a.rules(),list(a.live.values()))
        a.negative_rates = {'h':.6}
        self.assertEqual(a.rules()[0].body,(Literal('A',negative=True),))
        self.assertEqual(a.rules()[0].strength,.6)
