import io
import json
import unittest
from contextlib import redirect_stderr, redirect_stdout

from stationops.cli import main


class StreamingRunTests(unittest.TestCase):
    def test_v2_run_streams_each_shift_before_summary(self):
        output = io.StringIO()
        with redirect_stdout(output):
            main([
                "run",
                "--benchmark", "v2",
                "--backend", "reference",
                "--shifts", "2",
                "--modules", "2",
                "--initial-history-per-cohort", "2",
                "--stream",
            ])

        rows = [json.loads(line) for line in output.getvalue().splitlines()]
        self.assertEqual([row["type"] for row in rows], [
            "shift", "shift", "run-summary",
        ])
        self.assertEqual(
            [row["result"]["shift"] for row in rows[:-1]],
            [1, 2],
        )
        self.assertEqual(len(rows[-1]["result"]["rounds"]), 2)

    def test_stream_rejects_non_v2_run(self):
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            main(["run", "--benchmark", "v1", "--stream"])


if __name__ == "__main__":
    unittest.main()
