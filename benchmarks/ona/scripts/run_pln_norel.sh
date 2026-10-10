#!/usr/bin/env bash
# Identity matching, PLN without relational (shared-variable) hypotheses.
set -u
ulimit -v 16000000; unset DISPLAY
R=/nexus/Dev/OpenCog/bench/results/ona-benchmarks/raw/mts_norel
cd /nexus/Dev/OpenCog/ChainerGame-ona
for seed in 42 2 3; do
  PYTHONPATH=src timeout 1800 /nexus/Dev/OpenCog/NL2PLN_Project/PeTTaChainer/.venv/bin/python -u -m onabench.mts identitymatching --agent pln --seed $seed --no-relational \
    --out $R/identitymatching_pln-norel_s$seed.json < /dev/null > $R/identitymatching_pln-norel_s$seed.log 2>&1
done
echo DONE > $R/done
