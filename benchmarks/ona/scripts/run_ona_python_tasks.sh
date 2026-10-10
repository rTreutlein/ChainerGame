#!/usr/bin/env bash
# Run ONA's own Python evaluation tasks (as evaluation.py does, plus extra seeds).
# Usage: run_ona_python_tasks.sh  (run through bench/capped)
set -u
ulimit -v 16000000; unset DISPLAY
OUT=/nexus/Dev/OpenCog/bench/results/ona-benchmarks/raw/ona
cd /nexus/Dev/OpenCog/ONA/misc/Python
for task in conditioning discriminativefunction sortingtask identitymatching; do
  for seed in 42 1 2 3 4 5 6 7 8 9; do
    s=$(date +%s.%N)
    timeout 600 python3 $task.py silent seed=$seed > $OUT/${task}_s$seed.txt 2>&1
    e=$(date +%s.%N)
    echo "$task seed=$seed wall=$(echo "$e - $s" | bc)" >> $OUT/python_tasks_timing.txt
  done
done
s=$(date +%s.%N); timeout 600 python3 count_sheep.py > $OUT/count_sheep.txt 2>&1; e=$(date +%s.%N)
echo "count_sheep wall=$(echo "$e - $s" | bc)" >> $OUT/python_tasks_timing.txt
echo PY_TASKS_DONE >> $OUT/python_tasks_timing.txt
