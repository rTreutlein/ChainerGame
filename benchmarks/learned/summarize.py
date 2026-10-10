"""summary.json for the learned-rules runs: one flat row per stage and reasoner.

    python3 benchmarks/learned/summarize.py RESULTS_DIR META.json [GIVEN_SUMMARY.json]

RESULTS_DIR holds ``runs/`` (every reasoner but PLN) and one ``pln-<build>/``
per PeTTaChainer build; META.json is merged into ``meta``; GIVEN_SUMMARY.json
(the given-rules comparison's summary) adds its rows as ``meta.given_rules``
for comparison."""

import json
import sys
from collections import defaultdict
from pathlib import Path

CONFIGS = (("s1", "timed", "Stage 1"), ("s2", "timed", "Stage 2"), ("s3", "timed", "Stage 3 timed"), ("s3", "untimed", "Stage 3 untimed"))
LABEL = {"learned-reference": "Exact, learned rates", "problog": "ProbLog", "pettachainer": "PLN", "nars": "NARS", "prior": "Base rates"}
ROUNDS = 30


def load(directory: Path) -> dict:
    runs = defaultdict(list)
    for path in sorted(directory.glob("*-s*-*-seed*.json")):
        backend, stage, cycle, _ = path.stem.rsplit("-", 3)
        runs[(stage, cycle, backend)].append(json.loads(path.read_text()))
    return runs


def mean(rows: list[dict], field: str) -> float:
    return sum(r[field] for r in rows) / len(rows)


def row(label: str, backend: str, runs: list[dict], build: str | None) -> dict:
    assert all(r["learned_rules"] for r in runs), "learned-rules runs only"
    kinds = sorted({kind for r in runs for kind in r["kinds"]})
    return {
        "stage": label,
        "reasoner": LABEL[backend],
        "error": mean(runs, "posterior_error"),
        "brier": mean(runs, "brier"),
        "log_loss": mean(runs, "log_loss"),
        "coverage": mean(runs, "coverage"),
        "seconds_per_round": None if backend in ("prior", "learned-reference") else mean(runs, "seconds") / ROUNDS,
        "pln_build": build,
        "seeds": sorted(r["seed"] for r in runs),
        "error_by_kind": {kind: mean([r["kinds"][kind] for r in runs if kind in r["kinds"]], "posterior_error") for kind in kinds},
        **({"review_seconds_per_round": mean(runs, "review_seconds") / ROUNDS} if backend == "pettachainer" else {}),
    }


def main(root: Path, meta: dict, given: Path | None) -> dict:
    sources = [(load(root / "runs"), None)] + [(load(path), path.name.removeprefix("pln-")) for path in sorted(root.glob("pln-*")) if path.is_dir()]
    rows, exact = [], {}
    for stage, cycle, label in CONFIGS:
        for backend in LABEL:
            for runs, build in sources:
                found = runs.get((stage, cycle, backend))
                if found and (backend == "pettachainer") == (build is not None):
                    rows.append(row(label, backend, found, build))
        for runs, _ in sources:
            reference = runs.get((stage, cycle, "reference"))
            if reference:
                exact[label] = {"brier": mean(reference, "brier"), "log_loss": mean(reference, "log_loss")}
    out = {"rows": rows, "meta": {**meta, "exact_posterior_true_rates": exact}}
    if given:
        out["meta"]["given_rules"] = [
            {key: r[key] for key in ("stage", "reasoner", "error", "brier", "coverage", "seconds_per_round")}
            for r in json.loads(given.read_text())["supplynet"]
        ]
    (root / "summary.json").write_text(json.dumps(out, indent=1))
    return out


def markdown(summary: dict) -> str:
    """The rows as tables: overall with the given-rules numbers alongside, and error by query kind."""
    given = {(r["stage"], r["reasoner"].replace(" (PeTTaChainer)", "")): r for r in summary["meta"].get("given_rules", [])}
    number = lambda value, digits=3: "" if value is None else f"{value:.{digits}f}"  # noqa: E731
    lines = [
        "| stage | reasoner | error | error, given rules | Brier | Brier, given rules | coverage | s per round | s per round, given rules |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in summary["rows"]:
        name = r["reasoner"] + (f" ({r['pln_build']})" if r["pln_build"] else "")
        g = given.get((r["stage"], r["reasoner"]), {})
        lines.append(
            f"| {r['stage']} | {name} | {number(r['error'])} | {number(g.get('error'))} | {number(r['brier'])} | {number(g.get('brier'))} "
            f"| {r['coverage']:.2f} | {number(r['seconds_per_round'], 2)} | {number(g.get('seconds_per_round'), 2)} |"
        )
    for stage in dict.fromkeys(r["stage"] for r in summary["rows"]):
        rows = [r for r in summary["rows"] if r["stage"] == stage]
        kinds = sorted({kind for r in rows for kind in r["error_by_kind"]})
        lines += ["", f"{stage}, error by query kind:", "", "| reasoner | " + " | ".join(kinds) + " |", "|---|" + "---|" * len(kinds)]
        for r in rows:
            name = r["reasoner"] + (f" ({r['pln_build']})" if r["pln_build"] else "")
            lines.append(f"| {name} | " + " | ".join(number(r["error_by_kind"].get(kind)) for kind in kinds) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    print(markdown(main(Path(sys.argv[1]), json.loads(Path(sys.argv[2]).read_text()), Path(sys.argv[3]) if len(sys.argv) > 3 else None)))
