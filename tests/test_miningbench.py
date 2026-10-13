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
            row = s['history'][-1]
            x = next(int(a[0]) for (h,a),v in row.items() if v and h.endswith('_Pos'))
            velocity = next(int(a[0]) for (h,a),v in row.items() if v and h.endswith('_Vel'))
            z = x+velocity
            expected = -z if z < 0 else 64-z if z > 32 else z
            self.assertEqual(s['q']['T0_numeric'][expected],1)
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
            self.assertEqual(len(result['metadata']['petta_commit']),40)
            self.assertTrue(result['petta']['last_verdicts'])
            self.assertFalse(result['petta']['unparsed'])
            self.assertGreater(result['petta']['anticipated_negative_count'],0)


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


class ReviewRegressionTest(unittest.TestCase):
    def test_weak_singles_do_not_absorb_sufficient_conjunction(self):
        a,b = Literal('A'),Literal('B')
        conjunction = Rule('Y',(a,b))
        learned = [Rule('Y',(a,),strength=.54),Rule('Y',(b,),strength=.53),conjunction]
        result = recovery(learned,[conjunction],{'Y':'and2'})
        self.assertEqual(result['all']['recall'],1)
        self.assertEqual(result['all']['precision'],1)
        self.assertEqual(recovery(learned[:2],[conjunction],{'Y':'and2'})['all']['recall'],0)

    def test_exact_association_calibration(self):
        from miningbench.truth import conditional
        from miningbench.scoring import association_scores
        w = World(Config(kinds=('and2',)))
        rule = Rule('T0_and2',(Literal('T0_and2_F0'),),strength=.54)
        self.assertEqual(conditional(w,rule),.5)
        score = association_scores([rule],w.rules[0],w,0)
        self.assertAlmostEqual(score['mean_squared_error'],.04**2)

    def test_numeric_reachable_equivalence_and_memorization_limit(self):
        w = World(Config(kinds=('numeric',)))
        # At wall zero, velocities of +/-1 yield one. That grounded rule is
        # credited on its reachable contexts, not rejected for lacking a formula.
        rules = [Rule('T0_numeric',(Literal('T0_numeric_Pos',('0',)),
                                   Literal('T0_numeric_Vel',(str(v),))), '1') for v in (-1,1)]
        score = recovery(rules,w.rules[0],w.types,w)
        self.assertEqual(score['all']['precision'],1)
        self.assertAlmostEqual(score['all']['recall'],2/198)
        self.assertLess(score['all']['recall'],.05)
        self.assertEqual(recovery(w.rules[0],w.rules[0],w.types,w)['all']['recall'],1)

    def test_oracle_independent_interpreter_and_broken_oracle(self):
        from miningbench.truth import interpret
        for phase in (False,True):
            w = World(Config(samples=64,strength=.7,observation_noise=.2,drift=phase))
            for sample in w.stream():
                actual = interpret(w,sample['history'],sample['phase'])
                for t,q in sample['q'].items():
                    self.assertLess(max(abs(a-b) for a,b in zip(actual[t],q)),1e-12)
        # Deliberately broken numeric formula is observable by the independent check.
        w = World(Config(kinds=('numeric',)))
        w.rules[0] = [replace(r,formula='x1') for r in w.rules[0]]
        self.assertTrue(any(interpret(w,s['history'],0) != s['q'] for s in w.stream()))

    def test_fixed_budget_lag_priority_and_entity_holdout(self):
        from miningbench.systems import BudgetedMarginal
        w = World(Config(samples=64,kinds=('binding',),entity_holdout=True,deep_lags=12,distractors=10))
        learner = BudgetedMarginal(w.features,w.domains,budget=10)
        for sample in w.stream():
            learner.predict(sample['history'])
            learner.learn(sample['history'],sample['y'])
            self.assertLessEqual(learner.work['table_updates'],10)
            self.assertEqual(set(learner.work['by_lag']),{'1'})
            if sample['step'] >= 32:
                self.assertTrue(sample['holdout'])
        early = list(w.stream())[0]['history'][-1]
        late = list(w.stream())[-1]['history'][-1]
        self.assertTrue(all(a[0] in ('e0','e1','e2','e3') for (h,a),v in early.items() if v and a))
        self.assertTrue(all(a[0] in ('e4','e5','e6','e7') for (h,a),v in late.items() if v and a))

    def test_cost_slope_rejects_growing_work(self):
        from miningbench.scoring import CostMeter
        constant,growing = CostMeter(1024),CostMeter(1024)
        for i in range(1024):
            constant.add(1)
            growing.add(i+1)
        self.assertTrue(constant.result()['constant_cost_screen'])
        self.assertFalse(growing.result()['constant_cost_screen'])

    def test_skill_and_every_gate_negative_control(self):
        from miningbench.report import gate
        # A population predictor has zero skill at all strengths, even though
        # its expected label Brier decreases. Oracle skill remains one.
        for p in (.5,.35,.1):
            base = losses([1-p,p],[1.,0.])['brier']
            self.assertEqual(1-base/base,0)
        for direction,good,bad in [('above',[1.]*10,[0.]*10),('below',[0.]*10,[1.]*10)]:
            result = gate('fixture',good,.5,direction,bad,'broken predictor')
            self.assertEqual(result['status'],'PASS')
            self.assertTrue(result['discriminates'])
            self.assertEqual(gate('fixture',bad,.5,direction,bad,'broken predictor')['status'],'FAIL')
