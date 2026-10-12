#!/usr/bin/env bash
# Usage: run_pong.sh JOBFILE  -- one onabench.pong run per line of JOBFILE:
#   NAME ARGS...   (e.g. "pln_pong_s11 pong --agent pln --seed 11")
# Runs PONG_JOBS (default 4) at a time through the admission pool;
# NAME.json/.log go to PONG_RESULTS (default
# /nexus/Dev/OpenCog/bench/results/pong-onestep/raw). Existing results are
# kept. PONG_CG runs another ChainerGame checkout's onabench (default: this
# one), PETTACHAINER_PYTHON another PeTTaChainer venv, PONG_PEAK_MB tells the
# pool a run's expected peak (the one-step agent grows about 1 MB a decision).
set -u
R=${PONG_RESULTS:-/nexus/Dev/OpenCog/bench/results/pong-onestep/raw}; mkdir -p $R
CG=${PONG_CG:-$(cd "$(dirname "$0")/../../.." && pwd)}
PY=${PETTACHAINER_PYTHON:-/nexus/Dev/OpenCog/NL2PLN_Project/PeTTaChainer/.venv/bin/python}
one() {
  name=$1; shift
  [ -s $R/$name.json ] && return
  ( ulimit -v 16000000; unset DISPLAY; cd $CG
    PYTHONPATH=src timeout 14400 /nexus/Dev/OpenCog/bench/capped --peak-mb ${PONG_PEAK_MB:-1500} $PY -u -m onabench.pong "$@" --out $R/$name.json \
      < /dev/null > $R/$name.log 2>&1 )
  echo "$name $(tail -1 $R/$name.log | cut -c1-160)"
}
export -f one; export R CG PY PONG_PEAK_MB
grep -v '^\s*\(#\|$\)' "$1" | xargs -P ${PONG_JOBS:-4} -L 1 bash -c 'one "$@"' _
echo PONG_BATCH_DONE
