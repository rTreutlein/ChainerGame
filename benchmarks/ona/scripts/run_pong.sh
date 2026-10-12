#!/usr/bin/env bash
# Usage: run_pong.sh JOBFILE  -- one onabench.pong run per line of JOBFILE:
#   NAME ARGS...   (e.g. "pln_pong_s11 pong --agent pln --seed 11")
# Runs 4 at a time through the admission pool; NAME.json/.log go to
# /nexus/Dev/OpenCog/bench/results/pong/raw/. Existing results are kept.
set -u
R=/nexus/Dev/OpenCog/bench/results/pong/raw; mkdir -p $R
CG=$(cd "$(dirname "$0")/../../.." && pwd)
PY=${PETTACHAINER_PYTHON:-/nexus/Dev/OpenCog/NL2PLN_Project/PeTTaChainer/.venv/bin/python}
one() {
  name=$1; shift
  [ -s $R/$name.json ] && return
  ( ulimit -v 16000000; unset DISPLAY; cd $CG
    PYTHONPATH=src timeout 7200 /nexus/Dev/OpenCog/bench/capped $PY -u -m onabench.pong "$@" --out $R/$name.json \
      < /dev/null > $R/$name.log 2>&1 )
  echo "$name $(tail -1 $R/$name.log | cut -c1-160)"
}
export -f one; export R CG PY
grep -v '^\s*\(#\|$\)' "$1" | xargs -P 4 -L 1 bash -c 'one "$@"' _
echo PONG_BATCH_DONE
