#!/usr/bin/env bash
# Run every examples/nal/*.nal through ONA's shell exactly as evaluation.py does.
set -u
ulimit -v 16000000; unset DISPLAY
OUT=/nexus/Dev/OpenCog/bench/results/ona-benchmarks/raw/ona/nal
mkdir -p $OUT
cd /nexus/Dev/OpenCog/ONA
for f in examples/nal/*.nal; do
  b=$(basename $f .nal)
  s=$(date +%s.%N)
  timeout 600 ./NAR shell < $f > $OUT/$b.out 2>&1
  e=$(date +%s.%N)
  echo "$b wall=$(echo "$e - $s" | bc)" >> $OUT/timing.txt
done
echo NAL_DONE >> $OUT/timing.txt
