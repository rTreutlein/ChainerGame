"""Analysis-only reproduction of the historical fixed-order D100 workload.

This intentionally reproduces the old ingestion artefact to locate its retained
memory. It is not a learner arm or corrected distractor prediction result.
Run via capped with the 171b2820 Python path; no engine stores are modified.
"""

import argparse
import json
import os
from miningbench.petta import Petta
from miningbench.runner import run
from miningbench.world import Config


def legacy_inputs(self, row, period):
    self.input_count += len(row)
    self.h.add_atoms_no_check(
        [
            f"(: in-{period}-{i} ({head} {' '.join(args)} p{period}) (STV {value} 1))"
            for i, ((head, args), value) in enumerate(row.items())
        ]
    )


def main():
    from pathlib import Path

    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--deep-lags", type=int, default=3)
    parser.add_argument("--seed", type=int, default=9)
    args = parser.parse_args()
    Petta.add_inputs = legacy_inputs
    Petta.input_order = "historical-fixed-order-repro"
    os.environ["MININGBENCH_STORE_CENSUS"] = f"64,{args.samples}"
    result = run(
        Config(
            seed=args.seed,
            samples=args.samples,
            kinds=("single",),
            distractors=100,
            deep_lags=args.deep_lags,
        ),
        "petta",
        args.out,
        budget=40,
    )
    result["metadata"].update(
        input_order="historical-fixed-order-repro", level="memory_census_only"
    )
    (args.out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    world_path = args.out / "world.json"
    world = json.loads(world_path.read_text())
    world.update(input_order="historical-fixed-order-repro", level="memory_census_only")
    world_path.write_text(json.dumps(world, indent=2) + "\n")


if __name__ == "__main__":
    main()
