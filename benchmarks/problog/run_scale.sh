#!/usr/bin/env bash
# ProbLog on SupplyNet stage scale: one JSON summary per size, seed and
# engine, at most $JOBS runs at a time in the memory-capped bench slice under
# ulimit -v $MEMORY_KB (default 8000000: exact compilation can take all
# memory it is given, and the slice is shared), each round killed after
# $TIMEOUT seconds (default 120).
#
#   benchmarks/problog/run_scale.sh OUT_DIR "s m l xl" "1 2" ["ddnnf sdd"]
set -euo pipefail
out=$1 sizes=$2 seeds=$3 engines=${4:-ddnnf}
here=$(cd "$(dirname "$0")/../.." && pwd)
problog_python=${PROBLOG_PYTHON:-$here/.venv-problog/bin/python}
capped=${CAPPED:-/nexus/Dev/OpenCog/bench/capped}
jobs=${JOBS:-2} timeout=${TIMEOUT:-120} memory=${MEMORY_KB:-8000000}
mkdir -p "$out"
unset DISPLAY

run() {
  local engine=$1 size=$2 seed=$3
  local name="problog-$engine-$size-seed$seed"
  [[ -s "$out/$name.json" ]] && return 0
  while (( $(free -g | awk '/^Mem:/ {print $7}') < 8 )); do sleep 5; done
  "$capped" bash -c "ulimit -v $memory; cd '$out' && PYTHONPATH='$here/src' '$problog_python' -m supplynet.cli run \
      --stage scale --size $size --seed $seed --backend problog --problog-engine $engine --problog-timeout $timeout" \
    > "$out/$name.json.tmp" 2> "$out/$name.err" && mv "$out/$name.json.tmp" "$out/$name.json"
}

for engine in $engines; do
  for size in $sizes; do
    for seed in $seeds; do
      while (( $(jobs -rp | wc -l) >= jobs )); do sleep 1; done
      run $engine $size $seed &
      sleep 1
    done
  done
done
wait
