#!/usr/bin/env bash
# ONA_SEEDED: a copy of ONA's src with main.c calling mysrand(atol(getenv("ONA_SEED"))) after NAR_INIT(),
# built as build.sh builds ONA (stage-2 flags), and a copy of misc/Python/NAR.py in its misc/Python.
# ONA's own C Pong/Pong2, 10000 iterations: the default generator ("default", as ./NAR pong 10000)
# and mysrand(seed) after NAR_INIT for seeds 1-20. Writes cumulative hits/misses every 100 steps.
ulimit -v 16000000; unset DISPLAY
R=/nexus/Dev/OpenCog/bench/results/pong/ona_c
cd "${ONA_SEEDED:-/nexus/Dev/OpenCog/ChainerGame-pong/.scratch/ona-seeded}"
for v in pong pong2; do for s in default $(seq 1 20); do
  if [ $s = default ]; then env -u ONA_SEED ./NAR $v 10000; else ONA_SEED=$s ./NAR $v 10000; fi \
   | grep -a -E '^Hits=' | sed -E 's/Hits=([0-9]+) misses=([0-9]+) ratio=\S+ time=([0-9]+)/\3 \1 \2/' \
   | awk '$1 % 100 == 0 || $1 == 10001' > $R/${v}_${s}.curve
  echo "$v $s $(tail -1 $R/${v}_${s}.curve)" >> $R/summary.txt
done; done
echo DONE >> $R/summary.txt
