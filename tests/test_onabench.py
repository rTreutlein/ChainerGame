import os
import sys
import types
import unittest
from collections import namedtuple
from unittest import mock

from onabench import mts

Answer = namedtuple("Answer", "proof statement tv")


class FakeChainer:
    """Records what the agent stores and asks; answers (Reward op t) from `answers`."""

    def __init__(self):
        self.added = []
        self.queries = []
        self.answers = {}

    def __getattr__(self, name):
        if name.startswith("set_"):
            return lambda *args: None
        raise AttributeError(name)

    def timing(self):
        return {"inference": 0.0, "overhead": 0.0}

    def add_atoms_no_check(self, atoms):
        self.added.extend(atoms)

    def query_many_refs(self, statements, steps):
        self.queries.extend(statements)
        return [self.answers.get(s, []) for s in statements]


def agent(**kwargs):
    fake = FakeChainer()
    with mock.patch.dict(sys.modules, {"pettachainer": types.SimpleNamespace(PeTTaChainer=lambda: fake)}):
        return mts.PLNAgent(seed=1, steps=200, **kwargs), fake


class OutcomeEvidenceTest(unittest.TestCase):
    def test_none_stores_only_the_chosen_outcome(self):
        a, fake = agent()
        a.trial("A1", "B1", "B2")
        a.feedback("^right", False)
        self.assertIn("(: o1 (Reward right t1) (STV 0 1))", fake.added)
        self.assertFalse([s for s in fake.added if "Reward left" in s])

    def test_rule_states_that_exactly_one_operation_pays(self):
        a, fake = agent(outcome_evidence="rule")
        self.assertEqual(fake.added[:2], [
            "(: one-of-left (Implication (Reward left $t) (Reward right $t)) (CTV (STV 0 1) (STV 1 1)))",
            "(: one-of-right (Implication (Reward right $t) (Reward left $t)) (CTV (STV 0 1) (STV 1 1)))"])
        a.trial("A1", "B1", "B2")
        fake.queries.clear()
        a.feedback("^right", False)
        self.assertFalse([s for s in fake.added if "Reward left" in s and "Implication" not in s])
        self.assertEqual(fake.queries, ["(: $p (Reward left t1) $tv)"])

    def test_facts_store_the_unchosen_outcome_and_its_hypotheses(self):
        a, fake = agent(outcome_evidence="facts")
        a.trial("A1", "B1", "B2")
        a.feedback("^right", False)
        self.assertIn("(: o1 (Reward right t1) (STV 0 1))", fake.added)
        self.assertIn("(: c1 (Reward left t1) (STV 1 1))", fake.added)
        self.assertEqual({op for _, op in a.hyps}, {"left", "right"})
        a.trial("A2", "B2", "B1")
        a.feedback("^left", True)
        self.assertIn("(: c2 (Reward right t2) (STV 0 1))", fake.added)


class DesireTest(unittest.TestCase):
    def test_matching_reads_only_operations_with_a_matching_hypothesis(self):
        a, fake = agent(desire="matching")
        a.trial("A1", "B1", "B2")
        a.feedback("^right", True)
        fake.queries.clear()
        fake.answers["(: $p (Reward right t2) $tv)"] = [Answer("p", "s", ["STV", 1.0, 0.8])]
        a.trial("A1", "B1", "B2")
        self.assertNotIn("(: $p (Reward left t2) $tv)", fake.queries)
        self.assertEqual(a.timing[-1]["desire"], {"left": 0.5, "right": 0.9})

    def test_all_reads_the_population_prior_and_records_it(self):
        a, fake = agent()
        fake.answers["(: $p (Reward left t1) $tv)"] = [
            Answer(["population-prior", ["Reward", "left", "$_0"]], "s", ["STV", 1.0, 0.5])]
        a.trial("A1", "B1", "B2")
        self.assertEqual(a.timing[-1]["desire"], {"left": 0.75, "right": 0.5})
        self.assertEqual(a.timing[-1]["prior"], ["left"])

    def test_matching_desire_needs_harness_hypotheses(self):
        argv = ["mts", "reversal", "--agent", "pln", "--mined", "8", "16", "--desire", "matching", "--out", os.devnull]
        with mock.patch.object(sys, "argv", argv), mock.patch("sys.stderr"), self.assertRaises(SystemExit):
            mts.main()


@unittest.skipUnless(os.environ.get("SUPPLYNET_LIVE_PETTACHAINER") == "1", "set SUPPLYNET_LIVE_PETTACHAINER=1")
class LiveCounterfactualTest(unittest.TestCase):
    """Outcomes the task rule derives count as the hypotheses' samples and,
    once a query has derived them, as members of the population prior."""

    def chainer(self, mode):
        path = os.environ.get("PETTACHAINER_PYTHONPATH")
        if path:
            sys.path.insert(0, path)
        from pettachainer import PeTTaChainer
        h = PeTTaChainer()
        h.set_evidence_confidence_k(1)
        h.set_rule_refinement(True)
        for p in ("Sample", "LeftStim", "RightStim"):
            h.set_positive_only_predicate(p)
        if mode == "rule":
            h.add_atoms_no_check([f"(: one-of-{a} (Implication (Reward {a} $t) (Reward {b} $t)) (CTV (STV 0 1) (STV 1 1)))"
                                  for a, b in (("left", "right"), ("right", "left"))])
        h.add_atoms_no_check(["(: hl (Implication (And (Sample A1 $t) (LeftStim B1 $t)) (Reward left $t)) (STV 0.5 0.01))"])
        for t in range(1, 7):
            h.add_atoms_no_check([f"(: s{t} (Sample A1 t{t}) (STV 1 1))", f"(: l{t} (LeftStim B1 t{t}) (STV 1 1))",
                                  f"(: r{t} (RightStim B2 t{t}) (STV 1 1))", f"(: o{t} (Reward right t{t}) (STV 0 1))"]
                                 + ([f"(: c{t} (Reward left t{t}) (STV 1 1))"] if mode == "facts" else []))
        return h

    def truth(self, h, statement):
        (rows,) = h.query_many_refs([f"(: $p {statement} $tv)"], steps=200)
        return [(round(float(r.tv[1]), 3), round(float(r.tv[2]), 3)) for r in rows]

    def test_derived_outcomes_count_like_stored_ones(self):
        rule_truth = "(RuleTruth (Implication (And (Sample A1 $t) (LeftStim B1 $t)) (Reward left $t)))"
        facts, rule = self.chainer("facts"), self.chainer("rule")
        self.assertEqual(self.truth(rule, rule_truth), self.truth(facts, rule_truth))
        self.assertEqual(self.truth(rule, rule_truth), [(0.999, 0.857)])
        for h in (facts, rule):
            h.add_atoms_no_check(["(: s9 (Sample A3 t9) (STV 1 1))", "(: l9 (LeftStim B3 t9) (STV 1 1))",
                                  "(: r9 (RightStim B4 t9) (STV 1 1))"])
        self.assertEqual(self.truth(rule, "(Reward left t9)"), self.truth(facts, "(Reward left t9)"))

    def test_a_derived_outcome_counts_once_a_query_derives_it(self):
        rule_truth = "(RuleTruth (Implication (And (Sample A1 $t) (LeftStim B1 $t)) (Reward left $t)))"
        h = self.chainer("rule")
        self.assertEqual(self.truth(h, rule_truth), [(0.999, 0.857)])
        h.add_atoms_no_check([f"(: {p}{t} ({q} t{t}) (STV {v} 1))" for t in (7, 8, 9)
                              for p, q, v in (("s", "Sample A1", 1), ("l", "LeftStim B1", 1), ("o", "Reward right", 0))])
        (rows,) = h.query_many_refs([f"(: $p {rule_truth} $tv)"], steps=0)
        self.assertEqual(round(float(rows[0].tv[2]), 3), 0.857)
        h.query_many_refs([f"(: $p (Reward left t{t}) $tv)" for t in (7, 8, 9)], steps=0)
        (rows,) = h.query_many_refs([f"(: $p {rule_truth} $tv)"], steps=0)
        self.assertEqual(round(float(rows[0].tv[2]), 3), 0.9)


if __name__ == "__main__":
    unittest.main()
