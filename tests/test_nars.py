import os
import subprocess
import unittest

from supplynet import nars
from supplynet.game import GameConfig, run_game


class TranslationTest(unittest.TestCase):
    def test_statements(self):
        self.assertEqual(nars.statement(nars.parse("(Storm north $p)")), "<(north * $p) --> storm>")
        self.assertEqual(nars.statement(nars.parse("(StockedFuel t5)")), "<t5 --> stockedfuel>")
        self.assertEqual(nars.statement(nars.parse("(Not (Degraded power-plant t5))")), "(! <(power_plant * t5) --> degraded>)")
        self.assertEqual(nars.conjunction(["a", "b", "c"]), "((a && b) && c)")

    def test_ctv_rule_states_both_branches_for_both_polarities(self):
        lines = nars.rule("(: block-r1 (Implication (Storm north $p) (Blocked r1 $p)) (CTV (STV 0.8 1) (STV 0.03 1)))")
        self.assertEqual(
            lines,
            [
                "<<(north * $p) --> storm> ==> <(r1 * $p) --> blocked>>. %0.8;0.99%",
                "<<(north * $p) --> storm> ==> (! <(r1 * $p) --> blocked>)>. %0.2;0.99%",
                "<(! <(north * $p) --> storm>) ==> <(r1 * $p) --> blocked>>. %0.03;0.99%",
                "<(! <(north * $p) --> storm>) ==> (! <(r1 * $p) --> blocked>)>. %0.97;0.99%",
            ],
        )

    def test_a_learned_rule_takes_its_branches_from_the_labelled_samples(self):
        """k of n samples per branch, as revision sums NARS's induction; a
        branch with no sample is left out."""
        text = "(: block-r1 (Implication (Storm north $p) (Blocked r1 $p)) (CTV (STV 0.5 0.02) (STV 0.5 0.02)))"
        self.assertEqual(
            nars.rule(text, {("Blocked", "r1"): ((3, 4), (0, 0))}),
            [
                "<<(north * $p) --> storm> ==> <(r1 * $p) --> blocked>>. %0.75;0.8%",
                "<<(north * $p) --> storm> ==> (! <(r1 * $p) --> blocked>)>. %0.25;0.8%",
            ],
        )

    def test_persistence_keeps_the_period_link_in_the_negative_branch(self):
        lines = nars.rule(
            "(: persist-north (Implication (And (NextPeriod $p $t) (Storm north $p)) (Storm north $t)) (CTV (STV 0.7 1) (STV 0.3 1)))"
        )
        self.assertIn("<(<($p * $t) --> nextperiod> && (! <(north * $p) --> storm>)) ==> <(north * $t) --> storm>>. %0.3;0.99%", lines)

    def test_certain_or_and_negated_and_split(self):
        fuelled = nars.rule(
            "(: fuelled (Implication (Or (StockedFuel $t) (FuelArrived $t)) (Fuelled power-plant $t)) (CTV (STV 1 1) (STV 0 1)))"
        )
        self.assertEqual(
            fuelled,
            [
                "<<$t --> stockedfuel> ==> <(power_plant * $t) --> fuelled>>. %1;0.99%",
                "<<$t --> fuelarrived> ==> <(power_plant * $t) --> fuelled>>. %1;0.99%",
                "<((! <$t --> stockedfuel>) && (! <$t --> fuelarrived>)) ==> (! <(power_plant * $t) --> fuelled>)>. %1;0.99%",
            ],
        )
        runs = nars.rule(
            "(: runs-mine (Implication (And (Producing power-plant $t) (Not (Degraded mine $t))) (Producing mine $t)) (CTV (STV 1 1) (STV 0 1)))"
        )
        self.assertIn("<(! <(power_plant * $t) --> producing>) ==> (! <(mine * $t) --> producing>)>. %1;0.99%", runs)
        self.assertIn("<<(mine * $t) --> degraded> ==> (! <(mine * $t) --> producing>)>. %1;0.99%", runs)

    def test_facts_take_their_true_polarity(self):
        self.assertEqual(nars.fact("(: late-s1-1-t3 (Late s1-1 t3) (STV 1 1))"), "<(s1_1 * t3) --> late>. %1;0.99%")
        self.assertEqual(nars.fact("(: late-s1-1-t3 (Late s1-1 t3) (STV 0 1))"), "(! <(s1_1 * t3) --> late>). %1;0.99%")

    def test_base_rate_is_induction_over_the_labelled_periods(self):
        self.assertEqual(
            nars.base_rate("Storm", "north", 3, 30),
            [
                "<<$t --> period> ==> <(north * $t) --> storm>>. %0.1;0.967742%",
                "<<$t --> period> ==> (! <(north * $t) --> storm>)>. %0.9;0.967742%",
            ],
        )

    def test_answers(self):
        output = "noise\nAnswer: None.\nAnswer: <a --> b>. creationTime=2 Stamp=[2,1] Truth: frequency=0.800000, confidence=0.648000\n"
        self.assertEqual(nars.answers(output), [None, (0.8, 0.648)])


@unittest.skipUnless(os.access(nars.DEFAULT_NAR, os.X_OK), "ONA is not built")
class OnaTest(unittest.TestCase):
    def ask(self, lines):
        text = "\n".join(["*volume=0", *lines, ""])
        return nars.answers(subprocess.run([nars.DEFAULT_NAR, "shell"], input=text, capture_output=True, text=True, timeout=60).stdout)

    def test_deduction_and_abduction_through_a_translated_rule(self):
        rule = nars.rule("(: block-r1 (Implication (Storm north $p) (Blocked r1 $p)) (CTV (STV 0.8 1) (STV 0.03 1)))")
        (forward,) = self.ask([*rule, nars.fact("(: s (Storm north t5) (STV 1 1))"), "5", "<(r1 * t5) --> blocked>?"])
        self.assertAlmostEqual(forward[0], 0.8, places=3)
        (backward,) = self.ask([*rule, nars.fact("(: b (Blocked r1 t5) (STV 1 1))"), "5", "<(north * t5) --> storm>?"])
        self.assertGreater(backward[0], 0.5)

    def test_stage_one_round_answers_every_query(self):
        summary = run_game(GameConfig(seed=3, rounds=1, budget=0, history=5), nars.NarsBackend(cycles_per_step=0))
        self.assertEqual(summary["backend"], "nars")
        self.assertGreater(summary["coverage"], 0.5)


if __name__ == "__main__":
    unittest.main()
