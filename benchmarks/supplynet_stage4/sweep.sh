#!/usr/bin/env bash
# SupplyNet stage 4 measurements (docs/supply_network.md, "Stage 4 results").
# Each run is one CLI subprocess through the admission pool, writing its
# summary as JSON to $OUT/<name>.json; a run whose file exists is skipped.
# Usage: sweep.sh OUT JOBS
set -euo pipefail
OUT=${1:?output directory}; JOBS=${2:-8}
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
CHAINER_PY=${CHAINER_PY:-/nexus/Dev/OpenCog/NL2PLN_Project/PeTTaChainer/.venv/bin/python}
PROBLOG_PY=${PROBLOG_PY:-/nexus/Dev/OpenCog/ChainerGame-problog/.venv-problog/bin/python}
ROUNDS=${ROUNDS:-30}
mkdir -p "$OUT"
jobs_file=$(mktemp)
add() { echo "$1|$2|$3" >> "$jobs_file"; }  # name | python | cli arguments
for seed in 1 2 3 4; do
  for backend in reference greedy beam prior; do add "$backend-s$seed" python3 "--backend $backend --seed $seed"; done
  add "problog-s$seed" "$PROBLOG_PY" "--backend problog --seed $seed"
  for budget in ${BUDGETS:-50 100 200 400}; do
    add "pettachainer-extend-b$budget-s$seed" "$CHAINER_PY" "--backend pettachainer --proposer extend --budget $budget --seed $seed"
  done
  add "pettachainer-exhaustive-b200-s$seed" "$CHAINER_PY" "--backend pettachainer --proposer exhaustive --budget 200 --seed $seed"
done
# Decision cost against the number of single actions (plants) and capacity.
for plants in 2 3 4 5; do
  for capacity in 2 3; do
    for proposer in extend exhaustive; do
      add "growth-$proposer-p$plants-c$capacity" "$CHAINER_PY" "--backend pettachainer --proposer $proposer --budget 200 --seed 1 --plants $plants --capacity $capacity --rounds 10"
    done
    for backend in reference beam; do
      add "growth-$backend-p$plants-c$capacity" python3 "--backend $backend --seed 1 --plants $plants --capacity $capacity --rounds 10"
    done
    add "growth-problog-p$plants-c$capacity" "$PROBLOG_PY" "--backend problog --seed 1 --plants $plants --capacity $capacity --rounds 10"
  done
done
run() {
  IFS='|' read -r name py args <<< "$1"
  [ -s "$OUT/$name.json" ] && return 0
  rounds="--rounds $ROUNDS"; [[ "$args" == *--rounds* ]] && rounds=""
  ( cd "$ROOT"; unset DISPLAY; ulimit -v 16000000
    /nexus/Dev/OpenCog/bench/capped env PYTHONPATH=src "$py" -m supplynet.cli run --stage 4 $rounds $args < /dev/null \
      > "$OUT/$name.json.tmp" 2> "$OUT/$name.err" ) && mv "$OUT/$name.json.tmp" "$OUT/$name.json"
}
export -f run; export OUT ROOT ROUNDS
xargs -P "$JOBS" -I{} bash -c 'run "$@"' _ {} < "$jobs_file"
rm -f "$jobs_file"
