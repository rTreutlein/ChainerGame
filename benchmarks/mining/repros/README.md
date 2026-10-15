# Frozen miner capability and runtime probes

The original probes use PeTTaChainer `4e6b059b`, without modifications. They require the
shared interpreter and the capped wrapper, DISPLAY unset and virtual memory
limited to 16000000 KiB. `run.sh` supplies these settings. Use a fresh output path
for repeated generator runs; the benchmark refuses to overwrite evidence.

## Temporal links arriving after endpoint observations

`run.sh time-link-order` supplies identical facts in two orders. `F(entity,t)`
alternates true/false; `Y(entity,t+1)` copies it. Both predicates have layer zero,
so the miner must use the temporal link. Link-first proposes the intended rule
with strength 0.9993756555741329; link-last proposes no Y hypothesis.

The observed-predicate API requires observations before the closing link. At
`4e6b059b`, the miner records the link but does not revisit already ingested endpoints.
This limits natural event-time mining. The benchmark's context/outcome layer
encoding enables event lag 1 without changing the world; deeper event lags still
expose temporal restrictions. Raw log:
`/nexus/Dev/OpenCog/bench/results/mining-bench/time-link-order.log`.

On `7925f8d5`, both orders propose the intended rule with strength
0.9993756555741329 and confidence 0.8889924978641507. Late-link replay fixes
this candidate-generation repro. This does not by itself demonstrate firing
before a link arrives or mining conjunctions spanning several links. Raw log:
`/nexus/Dev/OpenCog/bench/results/mining-bench/review-v2-time-link-order-792.log`.

## Correct learned rates, incorrect predictive reads

`prediction_read.py` trains on 64 deterministic single-literal cases. Its final
live rule has CTV rates 0.9998485307482582 and 0.00016123831022250886. Fresh test
inputs yield strengths 0.45439112679589333 (true) and 0.4536851293455913 (false),
unchanged by subsequent queries at budgets 40, 400 and 2000. The returned confidence
is approximately 0.99647, and the proof is a revision of multiple views.

The cause is retained skeleton rows applying withdrawn hypotheses: removing a
hypothesis from rule indexes did not invalidate its existing skeleton applications.
`expand_proof` exposes the dropped hypothesis names inside the returned revision.
The upstream agent owns this fix. The harness still queries the chainer directly;
it does not filter proofs, reset the chainer, or substitute learned rates.
The expanded-proof rerun on `4e6b059b` confirms withdrawn `hypothesis-0`,
`-1`, `-2`, `-3`, `-4`, `-8`, and `-9` in both returned revisions, at all three
query budgets. The complete names, withdrawal verdicts and expansions are in
`/nexus/Dev/OpenCog/bench/results/mining-bench/review-v2-prediction-read.log`.
A separate rerun on master `6ffb1404` reproduces exactly the same two strengths
and the same seven withdrawn hypotheses in every expanded revision. That commit
changes implication-query semantics but does not fix hypothesis withdrawal. Its
raw evidence is
`/nexus/Dev/OpenCog/bench/results/mining-bench/review-v2-prediction-read-6ff.log`.
The exact sequential-read repro is retained rather than changing the benchmark
to read learned rule rates outside the chainer. Raw log:
`/nexus/Dev/OpenCog/bench/results/mining-bench/prediction-read.log`.

The upstream withdrawal and late-link replay fix landed during the powered
rerun. On master `7925f8d5`, the unchanged repro now returns
0.9998485307482582 for true input and 0.00016123831022250886 for false input,
at all three budgets. None of the nine withdrawn hypothesis names appears in
the expanded returned proofs. This fixes the reproduced predictive-read defect;
it does not establish recovery of rule forms the miner never proposes.
Evidence: `review-v2-prediction-read-792.log` and
`review-v2-prediction-read-792-check.json` under the same results directory.

```bash
unset DISPLAY
ulimit -v 16000000
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src TMPDIR="$PWD/.scratch" \
/nexus/Dev/OpenCog/bench/capped --slots 1 timeout --kill-after=10 90 \
/nexus/Dev/OpenCog/NL2PLN_Project/PeTTaChainer/.venv/bin/python \
benchmarks/mining/repros/prediction_read.py
```

## Candidate-language limits

`run.sh and3`, `run.sh and4`, `run.sh negation`, `run.sh lag`, `run.sh xor` and
`run.sh numeric` are independently seeded end-to-end repros. They save the
planted rules, predictions, current rates, raw review verdicts and normalized
learned rules. These limitations concern what the **miner proposes**, not what
PeTTa's rule language can express when a client supplies a rule:

- At most two observed antecedents at one period; no three-/four-input proposal.
- Antecedent features come from true facts. A CTV can learn a negative branch,
  but this does not create arbitrary mixed-polarity conjunctions for XOR.
- Across a link, current candidate generation reads one earlier feature rather
  than a conjunction spread over several periods; lag-3 search is absent.
- Numeric coordinates are symbols to the miner. No order, offset, velocity or
  wall-exception formulas are proposed. Coordinate-specific implications receive reachable-context behavioural
  credit in v2; they do not establish discovery of a general affine formula.
- The API supplies no directed target priority queue, action-relevance filtering,
  residual-split telemetry or auditable bounded-work counters. The benchmark
  records missing instrumentation rather than certifying the proposed design.

A same-period positive pair and shared-variable equality are supported proposal
forms. Failure to recover an individual pair in the fixed-seed run is therefore
a search/review outcome, not proof of an expressivity limit. The baseline binding
run recovered the shared-variable rule, which validates that the benchmark does
not categorically exclude that existing capability.


### Final-review diagnoses

`../review_diagnoses.py` extracts AND2 seed 2's exact pair proposal (step 1) and
retirement (step 7), plus the identical lag prediction hashes in the two arms.
The AND2 failure is review screening followed by permanent tried-rule exclusion,
not failure to propose a pair. At retirement, the pair is .99875 and F1 .66637;
F1's near-one value occurred earlier. The lag experiment is blocked by the
“present first” candidate policy and does not distinguish the two ingestion arms.

The old D100 harness always inserted signal predicates before distractors. With
only the newest 24 true facts per constant considered, that was an ordering
artefact. Production ingestion now shuffles each period deterministically from
the seed. The finite 24-fact window remains a real capacity limit.

`store_retention.py` intentionally reproduces the historical fixed-order D100
seed 9 solely for a read-only store census at 64 and 256 steps.
With `--deep-lags 12 --seed 1 --samples 409`, it profiles the historical deep-lag
workload at 64 and 409 steps. It does not
contribute to prediction scores. The census uses SWI predicate clause counts and
allocated predicate bytes without enumerating row terms or invoking GC. These
bytes explain store retention but are not an exact decomposition of process RSS.
