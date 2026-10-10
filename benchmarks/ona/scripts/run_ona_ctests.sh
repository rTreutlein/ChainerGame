#!/usr/bin/env bash
# ONA's procedure-learning system tests with evaluation.py's iteration counts, plus its C system tests.
set -u
ulimit -v 16000000; unset DISPLAY
OUT=/nexus/Dev/OpenCog/bench/results/ona-benchmarks/raw/ona/ctests; mkdir -p $OUT
cd /nexus/Dev/OpenCog/ONA
s=$(date +%s.%N); timeout 900 ./NAR > $OUT/systemtests.out 2>&1; echo "systemtests exit=$? wall=$(echo "$(date +%s.%N) - $s"|bc)" >> $OUT/summary.txt
for spec in "pong 10000" "pong2 10000" "alien 20000" "cartpole 10000" "robot 1200"; do
  set -- $spec
  s=$(date +%s.%N); timeout 900 ./NAR $1 $2 > $OUT/$1.out 2>&1; rc=$?
  echo "$1 $2 exit=$rc wall=$(echo "$(date +%s.%N) - $s"|bc) $(grep -a 'ratio=\|eaten=' $OUT/$1.out | tail -1)" >> $OUT/summary.txt
  tail -c 3000 $OUT/$1.out > $OUT/$1.tail; rm $OUT/$1.out
done
echo CTESTS_DONE >> $OUT/summary.txt
