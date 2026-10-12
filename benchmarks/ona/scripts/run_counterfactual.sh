#!/usr/bin/env bash
# Usage: run_counterfactual.sh OUT_DIR [JOBS] [MINED]
# PLN on conditioning, identitymatching, reversal, seeds 42, 1-9, change detection off and 1e4, for
# each evidence mode: none (a), rule and facts (b), none with desire only through matching hypotheses (c).
# MINED=1 runs --mined 8 16 instead (modes none, rule, facts). Runs through bench/capped; existing outputs are kept.
set -u
out=$1; jobs=${2:-4}; mined=${3:-0}
mkdir -p "$out"
PY=/nexus/Dev/OpenCog/NL2PLN_Project/PeTTaChainer/.venv/bin/python
PC=${PETTACHAINER_PATH:-/nexus/Dev/OpenCog/bench/chainers/82ac84cc}
CG=$(cd "$(dirname "$0")/../../.." && pwd)
if [[ $mined == 1 ]]; then modes="none:all rule:all facts:all"; extra="--mined 8 16"; tag=mined-; else modes="none:all rule:all facts:all none:matching"; extra=""; tag=""; fi
for task in conditioning identitymatching reversal; do
  for cd in off 10000; do
    for mode in $modes; do
      for seed in 42 1 2 3 4 5 6 7 8 9; do
        ev=${mode%:*}; de=${mode#*:}
        echo "$task $cd $ev $de $seed"
      done
    done
  done
done | xargs -P "$jobs" -L 1 bash -c '
  task=$0 cd=$1 ev=$2 de=$3 seed=$4
  f='"$out"'/${task}_'"$tag"'${ev}-${de}_cd${cd}_s${seed}
  [[ -s $f.json ]] && exit 0
  cdarg=""; [[ $cd != off ]] && cdarg="--change-detection $cd"
  cd '"$CG"' && /nexus/Dev/OpenCog/bench/capped timeout 3600 bash -c "ulimit -v 16000000; unset DISPLAY; PYTHONPATH=src '"$PY"' -u -m onabench.mts $task --agent pln --seed $seed --space sample-pairs --steps 200 '"$extra"' $cdarg --outcome-evidence $ev --desire $de --pettachainer-path '"$PC"' --out $f.json" < /dev/null > $f.log 2>&1
  echo "done $task $cd $ev $de $seed $(tail -c 300 $f.log | tail -1)"'
echo "GRID DONE"
