import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from benchmarks.learned.rule_rates import main


class RuleRateReportTests(unittest.TestCase):
    def test_reports_only_available_branches_in_current_and_historical_runs(self):
        goal = "(: $prf (RuleTruth (Implication (Storm north $p) (Blocked r1 $p))) $tv)"
        positive = [0.7, 0.4]
        historical = [0.6, 0.3, 0.05, 0.2]
        for truths in ([positive], [historical], [positive, historical]):
            with self.subTest(truths=truths), TemporaryDirectory(dir=Path(__file__).parent) as directory:
                root = Path(directory)
                for seed, truth in enumerate(truths, 1):
                    (root / f"pettachainer-s1-seed{seed}.json").write_text(json.dumps({
                        "seed": seed, "stage": 1, "rule_truths": {goal: truth},
                    }))
                out = root / "report.json"
                with patch("builtins.print"):
                    main(root, out)
                report = json.loads(out.read_text())
                self.assertEqual([row["pln"] for row in report["rows"]], truths)
                self.assertEqual(report["summary"]["given"]["rules"], len(truths))
                if historical in truths:
                    self.assertEqual(report["summary"]["without"]["rules"], 1)
                else:
                    self.assertNotIn("without", report["summary"])
