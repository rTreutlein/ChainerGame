"""Reproducible stage 5 grid, each process admitted through capped.

python3 benchmarks/supplynet_stage5/sweep.py [--jobs 4] [--group main|growth|all]
Writes streamed rounds, summaries, exit status and provenance. Existing
successful files are reused only with the same source hash and arguments.
"""
import argparse
import concurrent.futures
import fnmatch
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = Path("/nexus/Dev/OpenCog/bench/results/supplynet-stage5/review-fixes")
CHAINER_REPO = Path(os.environ.get("PETTACHAINER_REPO", "/nexus/Dev/OpenCog/NL2PLN_Project/PeTTaChainer"))
PETTA = os.environ.get("CHAINER_PY", str(CHAINER_REPO / ".venv/bin/python"))
MM2_PYTHON = os.environ.get("MM2_CHAINER_PYTHON", "/nexus/Dev/OpenCog/NL2PLN_Project/mm2-chainer/.venv/bin/python")
CAPPED = "/nexus/Dev/OpenCog/bench/capped"


def jobs(group):
    if group in ("all", "main"):
        for seed in range(1, 5):
            for backend in ("reference", "myopic", "greedy", "prior", "exact-pairs", "beam", "problog"):
                yield f"{backend}-s{seed}", PETTA, ["--backend", backend, "--seed", str(seed)]
            for budget in (50, 100, 200, 400):
                yield f"pettachainer-extend-b{budget}-s{seed}", PETTA, ["--backend", "pettachainer", "--budget", str(budget), "--seed", str(seed)]
            yield f"pettachainer-open-b200-s{seed}", PETTA, ["--backend", "pettachainer", "--proposer", "open", "--budget", "200", "--seed", str(seed)]
        for seed in range(1, 5):
            yield f"mm2-s{seed}", PETTA, ["--backend", "mm2", "--mm2-python", MM2_PYTHON, "--mm2-path", "/nexus/Dev/OpenCog/NL2PLN_Project/mm2-chainer/python", "--seed", str(seed)]
    if group in ("all", "growth"):
        for cap in (16, 32, 64, 128, 256):
            for proposer in ("extend", "open", "exhaustive"):
                yield f"growth-{proposer}-k{cap}", PETTA, ["--backend", "pettachainer", "--proposer", proposer, "--pair-cap", str(cap), "--budget", "200", "--seed", "1", "--rounds", "5"]
        for plants in (0, 2):
            for backend in ("reference", "problog"):
                yield f"growth-{backend}-p{plants}", PETTA, ["--backend", backend, "--plants", str(plants), "--seed", "1", "--rounds", "5"]


def main():
    global OUT
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--group", choices=("main", "growth", "all"), default="all")
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--case-pattern", default="*")
    parser.add_argument("--exclude-case-pattern", default="")
    args = parser.parse_args()
    OUT = args.out
    OUT.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    for path in sorted((ROOT/"src/supplynet").glob("*.py")):
        digest.update(path.name.encode()+path.read_bytes())
    source = digest.hexdigest()
    chainer_digest = hashlib.sha256()
    for path in sorted((CHAINER_REPO / "pettachainer").rglob("*")):
        if path.suffix in (".py", ".metta", ".pl"):
            chainer_digest.update(str(path.relative_to(CHAINER_REPO)).encode() + path.read_bytes())
    chainer_source = chainer_digest.hexdigest()
    provenance = {"source_sha256": source, "chainer_source_sha256": chainer_source, "chainer_commit": subprocess.check_output(["git", "-C", str(CHAINER_REPO), "rev-parse", "HEAD"], text=True).strip(),
                  "game_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                  "group": args.group, "jobs": args.jobs, "case_pattern": args.case_pattern, "exclude_case_pattern": args.exclude_case_pattern, "started": time.time()}
    (OUT/f"manifest-{args.group}.json").write_text(json.dumps(provenance, indent=2)+"\n")

    def run(job):
        name, python, arguments = job
        cli = [python, "-m", "supplynet.cli", "run", "--stage", "5", "--rounds", "30", "--problog-timeout", "15", "--stream", *arguments]
        fingerprint = {"source_sha256": source, "chainer_source_sha256": chainer_source, "command": cli, "chainer_commit": provenance["chainer_commit"]}
        status_path = OUT/f"{name}.status.json"
        if (OUT/f"{name}.json").exists() and status_path.exists():
            old = json.loads(status_path.read_text())
            if old.get("fingerprint") == fingerprint and old.get("returncode") == 0:
                print(name, "reused", flush=True)
                return
        env = {**os.environ, "PYTHONPATH": str(ROOT/"src")+":"+str(ROOT/"benchmarks/supplynet_stage5/scratch/deps"), "PYTHONDONTWRITEBYTECODE": "1", "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}
        env.pop("DISPLAY", None)
        command = [CAPPED, "--priority", "normal", "--slots", "1", "--peak-mb", "1024", "bash", "-c", 'ulimit -v 16000000; exec timeout --kill-after=10s 3600 "$@"', "stage5", *cli]
        started = time.perf_counter()
        with (OUT/f"{name}.jsonl").open("w") as stdout, (OUT/f"{name}.err").open("w") as stderr:
            process = subprocess.run(command, cwd=ROOT, env=env, stdout=stdout, stderr=stderr)
        status = {"fingerprint": fingerprint, "returncode": process.returncode, "wall_seconds": time.perf_counter()-started}
        status_path.write_text(json.dumps(status, indent=2)+"\n")
        if process.returncode == 0:
            last = json.loads((OUT/f"{name}.jsonl").read_text().splitlines()[-1])
            (OUT/f"{name}.json").write_text(json.dumps(last, indent=2)+"\n")
        else:
            (OUT/f"{name}.json").unlink(missing_ok=True)
        print(name, process.returncode, round(status["wall_seconds"], 1), flush=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        list(pool.map(run, (job for job in jobs(args.group) if fnmatch.fnmatchcase(job[0],args.case_pattern) and not fnmatch.fnmatchcase(job[0],args.exclude_case_pattern))))


if __name__ == "__main__":
    main()
