#!/usr/bin/env bash
# Run SupplyNet stages 1-3 for the given backends and seeds, one JSON summary
# per run, at most $JOBS runs at a time, each in the memory-capped bench slice
# under ulimit -v 16000000 and started only while 8 GB are available.
#
#   benchmarks/nars/run_grid.sh OUT_DIR "reference prior pettachainer nars" "1 2 3 4" [extra args...]
set -euo pipefail
out=$1 backends=$2 seeds=$3
shift 3
here=$(cd "$(dirname "$0")/../.." && pwd)
python=${PYTHON:-/nexus/Dev/OpenCog/NL2PLN_Project/PeTTaChainer/.venv/bin/python}
petta=${PETTACHAINER_PATH:-/nexus/Dev/OpenCog/NL2PLN_Project/PeTTaChainer}
capped=${CAPPED:-/nexus/Dev/OpenCog/bench/capped}
jobs=${JOBS:-4}
mkdir -p "$out"
unset DISPLAY

run() {
  local backend=$1 stage=$2 cycle=$3 seed=$4
  shift 4
  local name="$backend-s$stage-$cycle-seed$seed"
  [[ -s "$out/$name.json" ]] && return 0
  while (( $(free -g | awk '/^Mem:/ {print $7}') < 8 )); do sleep 5; done
  "$capped" bash -c "ulimit -v 16000000; cd '$out' && PYTHONPATH='$here/src' '$python' -m supplynet.cli run \
      --stage $stage --cycle $cycle --backend $backend --seed $seed --rounds 30 --budget 100 \
      --pettachainer-path '$petta' $*" > "$out/$name.json.tmp" 2> "$out/$name.err" \
    && mv "$out/$name.json.tmp" "$out/$name.json"
}

for backend in $backends; do
  for config in "1 timed" "2 timed" "3 timed" "3 untimed"; do
    for seed in $seeds; do
      while (( $(jobs -rp | wc -l) >= jobs )); do sleep 1; done
      run $backend $config $seed "$@" &
      sleep 1
    done
  done
done
wait
