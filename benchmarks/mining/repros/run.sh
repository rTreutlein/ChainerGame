#!/usr/bin/env bash
set -euo pipefail
cd /nexus/Dev/OpenCog/ChainerGame-mining
unset DISPLAY
ulimit -v 16000000
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=src
export TMPDIR=/nexus/Dev/OpenCog/ChainerGame-mining/.scratch
export MININGBENCH_CAPPED=1
mkdir -p "$TMPDIR"
case "${1:-}" in
  time-link-order)
    exec /nexus/Dev/OpenCog/bench/capped --slots 1 timeout --kill-after=10 90 \
      /nexus/Dev/OpenCog/NL2PLN_Project/PeTTaChainer/.venv/bin/python \
      benchmarks/mining/repros/time_link_order.py ;;
  single|and2|and3|and4|negation|lag|xor|or|numeric|binding)
    exec /nexus/Dev/OpenCog/bench/capped --slots 1 timeout --kill-after=10 90 \
      /nexus/Dev/OpenCog/NL2PLN_Project/PeTTaChainer/.venv/bin/python -m miningbench run \
      --system petta --kinds "$1" --samples 64 --budget 40 \
      --out "/nexus/Dev/OpenCog/bench/results/mining-bench/repro-$1" ;;
  *) echo 'usage: run.sh time-link-order|single|and2|and3|and4|negation|lag|xor|or|numeric|binding' >&2; exit 2 ;;
esac
