"""Check seed-identical prequential prefixes across independent base/long runs."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    runs = {}
    for path in sorted(args.root.rglob("result.json")):
        result = json.loads(path.read_text())
        meta = result["metadata"]
        config = meta["config"]
        if (
            meta.get("level") not in ("base", "long")
            or len(config["kinds"]) != 1
            or config["targets"] != 1
        ):
            continue
        if path.parent.parent.name == "pure" and config["kinds"] == ["numeric"]:
            continue
        status = path.parent / "status.json"
        if status.exists() and json.loads(status.read_text())["returncode"]:
            continue
        key = (
            meta["system"],
            config["seed"],
            tuple(config["kinds"]),
            meta.get("petta_commit"),
        )
        runs.setdefault(key, {})[meta["level"]] = path.parent
    checks = []
    for key, pair in runs.items():
        if set(pair) != {"base", "long"}:
            continue
        differences = []
        with (
            (pair["base"] / "steps.jsonl").open() as base,
            (pair["long"] / "steps.jsonl").open() as long,
        ):
            count = 0
            for a, b in zip(base, long):
                count += 1
                left, right = json.loads(a), json.loads(b)
                for field in ("predicted", "exact", "label", "scores"):
                    if left[field] != right[field]:
                        differences.append({"step": count - 1, "field": field})
        checks.append(
            {
                "system": key[0],
                "seed": key[1],
                "kinds": key[2],
                "prefix_steps": count,
                "differences": differences,
                "identical": not differences,
            }
        )
    with args.out.open("x") as out:
        json.dump(
            {
                "checks": checks,
                "all_identical": bool(checks) and all(c["identical"] for c in checks),
            },
            out,
            indent=2,
        )
        out.write("\n")
    if not checks or any(not c["identical"] for c in checks):
        raise SystemExit(1)
    print(f"{len(checks)} independently executed prefixes are identical")


if __name__ == "__main__":
    main()
