#!/usr/bin/env bash
# StationOps temporal replay (the bench suite's configuration, budget 50) for
# the given backends and seeds, one --stream file per run, at most $JOBS at a
# time in the memory-capped bench slice under ulimit -v 16000000. ProbLog runs
# in its own venv ($PROBLOG_PYTHON), the others in PeTTaChainer's.
#
#   benchmarks/problog/run_stationops.sh OUT_DIR "reference pettachainer problog" "1 2 3 4"
set -euo pipefail
out=$1 backends=$2 seeds=$3
here=$(cd "$(dirname "$0")/../.." && pwd)
python=${PYTHON:-/nexus/Dev/OpenCog/NL2PLN_Project/PeTTaChainer/.venv/bin/python}
problog_python=${PROBLOG_PYTHON:-$here/.venv-problog/bin/python}
petta=${PETTACHAINER_PATH:-/nexus/Dev/OpenCog/NL2PLN_Project/PeTTaChainer}
recordings=${RECORDINGS:-/nexus/Dev/OpenCog/bench/recordings}
capped=${CAPPED:-/nexus/Dev/OpenCog/bench/capped}
jobs=${JOBS:-4}
mkdir -p "$out"
unset DISPLAY

run() {
  local backend=$1 seed=$2 name="$1-seed$2" interpreter=$python
  [[ $backend == problog ]] && interpreter=$problog_python
  [[ -s "$out/$name.jsonl" ]] && return 0
  while (( $(free -g | awk '/^Mem:/ {print $7}') < 8 )); do sleep 5; done
  "$capped" bash -c "ulimit -v 16000000; cd '$out' && PYTHONPATH='$here/src' '$interpreter' -m stationops.cli run \
      --benchmark v2 --backend $backend --seed $seed --shifts 20 --modules 10 --initial-history-per-cohort 20 \
      --sensor-knowledge mixed --independent-modules --budget 50 --action-budget 1 --shortfall-budget 25 \
      --temporal-model --replay '$recordings/fv_s$seed.jsonl' --pettachainer-path '$petta' --stream" \
    > "$out/$name.jsonl.tmp" 2> "$out/$name.err" && mv "$out/$name.jsonl.tmp" "$out/$name.jsonl"
}

for backend in $backends; do
  for seed in $seeds; do
    while (( $(jobs -rp | wc -l) >= jobs )); do sleep 1; done
    run $backend $seed &
    sleep 1
  done
done
wait
