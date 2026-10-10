#!/usr/bin/env bash
# Run the conflicting-sources grid: backends x sizes x seeds x steps per query,
# one JSON summary per run, at most $JOBS runs at a time, each in the
# memory-capped bench slice under ulimit -v 16000000, started only while 8 GB
# are available. Budget-independent backends (the Python references and
# ProbLog) run once per size and seed. ProbLog runs in its own venv
# ($PROBLOG_PYTHON), the others in PeTTaChainer's ($PYTHON).
#
#   benchmarks/conflict/run_grid.sh OUT_DIR "exact pln-raw ..." "s m l" "1 2 3 4" "1 4 16"
set -euo pipefail
out=$1 backends=$2 sizes=$3 seeds=$4 budgets=$5
here=$(cd "$(dirname "$0")/../.." && pwd)
python=${PYTHON:-/nexus/Dev/OpenCog/NL2PLN_Project/PeTTaChainer/.venv/bin/python}
problog_python=${PROBLOG_PYTHON:-/nexus/Dev/OpenCog/ChainerGame-problog/.venv-problog/bin/python}
petta=${PETTACHAINER_PATH:-/nexus/Dev/OpenCog/NL2PLN_Project/PeTTaChainer}
capped=${CAPPED:-/nexus/Dev/OpenCog/bench/capped}
jobs=${JOBS:-4}
timeout=${TIMEOUT:-3600}
mkdir -p "$out"
unset DISPLAY

run() {
  local backend=$1 size=$2 seed=$3 spq=$4
  local name="$backend-$size-seed$seed-spq$spq" interpreter=$python
  [[ -s "$out/$name.json" ]] && return 0
  [[ $backend == problog-* ]] && interpreter=$problog_python
  while (( $(free -g | awk '/^Mem:/ {print $7}') < 8 )); do sleep 5; done
  if "$capped" timeout "$timeout" bash -c "ulimit -v 16000000; cd '$out' && PYTHONPATH='$here/src' '$interpreter' -m conflict.cli run \
      --backend $backend --size $size --seed $seed --steps-per-query $spq --pettachainer-path '$petta'" \
      < /dev/null > "$out/$name.json.tmp" 2> "$out/$name.err"; then
    mv "$out/$name.json.tmp" "$out/$name.json"
  else
    echo "{\"type\": \"run-failed\", \"backend\": \"$backend\", \"size\": \"$size\", \"seed\": $seed, \"steps_per_query\": $spq, \"exit\": $?}" > "$out/$name.failed"
    rm -f "$out/$name.json.tmp"
  fi
}

for size in $sizes; do
  for backend in $backends; do
    case $backend in pln-*|nars-*) spqs=$budgets ;; *) spqs=${budgets%% *} ;; esac
    for spq in $spqs; do
      for seed in $seeds; do
        while (( $(jobs -rp | wc -l) >= jobs )); do sleep 1; done
        run $backend $size $seed $spq &
        sleep 0.5
      done
    done
  done
done
wait
