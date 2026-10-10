#!/usr/bin/env bash
# Usage: run_ona_mts.sh TASK SEED -- ONA (v0.9.3, its own NAR.py) through the shared protocol.
set -u
ulimit -v 16000000; unset DISPLAY
R=/nexus/Dev/OpenCog/bench/results/ona-benchmarks/raw/mts
cd /nexus/Dev/OpenCog/ChainerGame-ona
PYTHONPATH=src timeout 600 python3 -m onabench.mts $1 --agent ona --seed $2 --out $R/$1_ona_s$2.json > $R/$1_ona_s$2.log 2>&1
