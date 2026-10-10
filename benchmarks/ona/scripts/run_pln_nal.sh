#!/usr/bin/env bash
# Run the PeTTaChainer translations of ONA's declarative NAL tests.
set -u
ulimit -v 16000000; unset DISPLAY
OUT=/nexus/Dev/OpenCog/bench/results/ona-benchmarks/raw/nal_pln
mkdir -p $OUT; : > $OUT/timing.txt
cd /nexus/Dev/OpenCog/ChainerGame-ona/benchmarks/ona/nal_pln
for f in ${@:-*.metta}; do
  b=${f%.metta}
  s=$(date +%s.%N)
  timeout 300 /nexus/Dev/OpenCog/NL2PLN_Project/PeTTaChainer/.venv/bin/petta $f --silent < /dev/null > $OUT/$b.out 2>&1
  rc=$?
  e=$(date +%s.%N)
  echo "$b wall=$(echo "$e - $s"|bc) exit=$rc" >> $OUT/timing.txt
done
echo NAL_PLN_DONE >> $OUT/timing.txt
