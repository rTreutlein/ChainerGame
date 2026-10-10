#!/usr/bin/env bash
# Usage: run_pln_mts.sh TASK SEED SPACE [STEPS]  -- one PLN run of an ONA matching-to-sample task.
set -u
ulimit -v 16000000; unset DISPLAY
task=$1; seed=$2; space=$3; steps=${4:-200}
R=/nexus/Dev/OpenCog/bench/results/ona-benchmarks/raw/mts
cd /nexus/Dev/OpenCog/ChainerGame-ona
PYTHONPATH=src timeout 7200 /nexus/Dev/OpenCog/NL2PLN_Project/PeTTaChainer/.venv/bin/python -u -m onabench.mts $task --agent pln --seed $seed --space $space --steps $steps \
  --out $R/${task}_pln-${space}_s${seed}.json < /dev/null > $R/${task}_pln-${space}_s${seed}.log 2>&1
