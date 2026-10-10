#!/usr/bin/env bash
# Run every ONA and PLN job of the matching-to-sample prototype, 4 at a time.
cd /nexus/Dev/OpenCog/bench/results/ona-benchmarks
{
for task in conditioning identitymatching reversal; do
  for seed in 42 1 2 3 4 5 6 7 8 9; do
    echo "./run_ona_mts.sh $task $seed"
    echo "./run_pln_mts.sh $task $seed sample-pairs"
  done
done
for seed in 1 2; do echo "./run_pln_mts.sh conditioning $seed all"; done
} | xargs -P 4 -I{} sh -c '{}'
echo BATCH_DONE > raw/mts/batch.done
