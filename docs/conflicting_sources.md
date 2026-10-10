# Conflicting sources: contradictory reports of different reliability and volume

Status: built (`src/conflict`), October 2026. Runs:
`benchmarks/conflict/run_grid.sh`; report: `benchmark-runs/conflict-2026-10-10/report.md`;
raw runs and `summary.json`: `/nexus/Dev/OpenCog/bench/results/conflict/`.

## Question

Several sources report on the same hidden statements. They differ in how
reliable they are, in which way they are wrong (noisy, biased towards "up",
towards "down", or inverted), and in how much they say: some report on most
statements every round, some repeat each claim ten times, some speak a handful
of times in the whole game. They contradict each other. The past rounds are
labelled, so each source's reliability can be learned.

What does each reasoner make of this? Specifically: what does PLN's
count-weighted revision give for free when it never sees who said what, how
much does it gain when the source identity reaches the knowledge base and
the reliabilities are learned by PLN itself, and how does that compare with
an exact probabilistic program (ProbLog) with the true or the learned model,
with a naive probabilistic program, and with NARS?

## World

`src/conflict/world.py`. Each round is a fresh day of one plant:

- **Components** `c1..cK`, each up independently with a prior drawn from
  [0.55, 0.9].
- **Systems** `w1..wS`, each on two random components: up with P(up | both
  up) ∈ [0.85, 0.97], P(up | not both) ∈ [0.03, 0.15]. These rates are given
  to every reasoner, as SupplyNet's rules are (`(Implication (And (Up c1 $t)
  (Up c2 $t)) (Up w1 $t))` with a certain CTV).
- **Sources** report on nodes (components and systems). Source s reports on
  a node with probability `coverage`; its claim is "up" with probability
  P(claim up | up) when the node is up, P(claim up | down) when it is down; it
  repeats the claim `repeats` times. The copies are the same claim, so they
  carry no evidence beyond the first: volume is not reliability.

Source archetypes (each parameter drawn uniformly from its range per seed):

| archetype | P(claim up \| up) | P(claim up \| down) | coverage | repeats | what it is |
|---|---|---|---|---|---|
| expert | 0.88–0.95 | 0.05–0.12 | 0.10–0.20 | 1 | accurate, rarely speaks |
| crowd | 0.60–0.70 | 0.30–0.40 | 0.50–0.70 | 1 | weak, speaks about most nodes |
| optimist | 0.93–0.99 | 0.50–0.65 | 0.40–0.60 | 2–4 | biased: "up" means little, "down" means a lot |
| contrarian | 0.10–0.25 | 0.75–0.90 | 0.20–0.40 | 1 | inverted: informative once its sign is known |
| echo | 0.55–0.70 | 0.40–0.50 | 0.70–0.90 | 8–15 | nearly uninformative, very loud |
| alarmist | 0.35–0.50 | 0.02–0.08 | 0.30–0.50 | 1–2 | biased the other way |
| rare | 0.85–0.95 | 0.05–0.15 | 0.005–0.02 | 1 | accurate, a few claims per game |

Over a default game (30 labelled rounds + 10 rounds) a crowd source makes
hundreds of claims, an echo thousands of claim copies, a rare source a few.
On a typical node the expert and the contrarian say the same word and mean
opposite things, and the echo outvotes both.

| size | components | systems | sources (roster order) | nodes = queries per round |
|---|---|---|---|---|
| s | 2 | 2 | expert, crowd, optimist, contrarian, echo | 4 |
| m | 4 | 4 | + alarmist, rare, crowd | 8 |
| l | 8 | 8 | + expert, optimist, crowd, contrarian, echo, alarmist | 16 |
| xl | 16 | 16 | + rare, crowd | 32 |

**Game** (`game.py`): 30 labelled rounds (`--history`), then 10 rounds
(`--rounds`). In a round the reasoner gets the claims (the states hidden),
states P(up) for every node, is scored, and the round's states are revealed:
it joins the labelled history.

**Exact posterior.** Given the round's claims, one per source and node,
P(node up | claims) under the true model, by summing over the 2^K component
states (systems and claims factor given them). Tested against enumerating
every node's state (`test_posterior_matches_brute_force`). `exact-learned`
is the same computation with the component priors and every source's two
rates estimated from the labelled rounds (Laplace counts, one claim per
source and node).

**Score.** Per query, |belief − exact posterior| (an unanswered query counts
as 0.5), averaged over the round's queries, the rounds and the seeds; also
split by node kind (component, system) and by whether the node's claims
disagree (contested) or not (agreed). Time is the wall time of the backend's
round (beliefs + resolve, reviews included). Budget: steps per query (1, 4,
16), the scale stage's settings; the round's budget is that times its
queries. It is PLN's backward-step budget and NARS's inference cycles; the
Python, exact and ProbLog contenders have none.

## Contenders

### PLN, raw (`pln-raw`)

Every node is the statement `(Up c1 t5)`. A source's claims on a node become
one fact about the node, with the claim as strength and the copy count n as
confidence n/(n + k), k = 5:

    (: rep-s5-c1-t5 (Up c1 t5) (STV 1 0.75))     ; the echo, 15 copies
    (: rep-s1-c1-t5 (Up c1 t5) (STV 0 0.166667)) ; the expert, once

plus the system rules and the labelled rounds as certain facts. The facts of
one statement revise, so the answer is the copy-weighted share of "up", moved
by the system rules (and their inversions) through the other nodes. No
source identity reaches the KB. This is what count-weighted revision gives for
free; `vote` is its Python ideal without the system rules.

### PLN, sources encoded (`pln-sources`): learned trust rules

The claim is a fact about the source, one per source and node however often
it was repeated, strength 1 for "up" and 0 for "down":

    (: claim-s4-w2-t5 (Claims s4 (Up w2 t5)) (STV 0 1))

and every source has one hypothesis rule with a weak CTV prior:

    (: trust-s4 (Implication (Claims s4 (Up $n $t)) (Up $n $t)) (CTV (STV 0.5 0.02) (STV 0.5 0.02)))

With `set-rule-refinement` on, the rule's truth is learned from the labelled
rounds (PeTTaChainer `docs/metta/hypothesis_rules.md`): P(up | s4 claims up)
from the instances, P(up | s4 claims down) from the base rates (since the
PeTTaChainer fixes below, from the down-claims). Before each round's queries,
and before its claims are added, every trust rule is reviewed by its own
`RuleTruth` query (20 steps, or `--review-steps`), so the last round's labels
are folded in. Reviewed with the round's claims in the fold, a claim whose
node the round's other views derive is a soft sample of its rule, a view of
that node may not read a rule truth depending on it, and every view read the
rule's prior. The answer for a node is
the revision of the views of the sources that spoke about it, with the system
rules' views.

### PLN, stated trust (`pln-stated`): learned once, stated as given

The same claims, but each source's CTV is learned once, from the initial 30
rounds, by an implication query `(: $prf (Implication (Claims s4 (Up $n $t))
(Up $n $t)) $tv)` in a KB of its own, and stated as a given rate at
confidence 1 before the history is added:

    (: trust-s4 (Implication (Claims s4 (Up $n $t)) (Up $n $t)) (CTV (STV 0.0667 1) (STV 0.9917 1)))

A source with no labelled claim gets no rule. This is how SupplyNet and
CombiNet hand the chainer rates (certain CTVs). It is the encoding that works
around two chainer problems (see "Chainer problems"): views of one rule on
different claims are treated as overlapping evidence unless the rule is
certain, and refined rules carry their samples, so their views always overlap.
Stated before the history, the labelled outcomes can also score PeTTaChainer's
combination modes (`docs/metta/combination_modes.md`). The rates are not
updated after the initial history.

### PLN, latent reliability (`pln-reliable`): one rule per claim

No claim facts. Each source has two latent statements, `(Reliable s4 up)` and
`(Reliable s4 down)`: its up-claims, its down-claims, are correct. Every
claim, history and current, is one certain rule from the source's
reliability for that polarity to the claimed state:

    (: (no_inverse claim-s4-c1-t5) (Implication (Reliable s4 up) (Up c1 t5)) (CTV (STV 1 1) (STV 0 1)))
    (: (no_inverse claim-s4-w2-t5) (Implication (Reliable s4 down) (Up w2 t5)) (CTV (STV 0 1) (STV 1 1)))

so P(Up c1 t5) = P(Reliable s4 up) and P(Up w2 t5) = 1 - P(Reliable s4
down): the reliabilities are the trust CTV's two branches, P(up | claim up)
and P(down | claim down), as statements. Each resolved round adds, per source
and polarity, the share of the source's claims that were correct, at the
confidence n/(n + k) of their count:

    (: reliability-s4-up-h3 (Reliable s4 up) (STV 0.75 0.444444))

and revision of the rounds' facts is the pooled share with count-based
confidence, over a prior of one correct and one wrong claim per polarity
(`(STV 0.5 2/(2+k))`, Laplace's rule, as exact-learned counts). Rules and
evidence of a round go in before its outcomes, which so score the
combination modes (odds wins).

Semantics, and the choices against the alternatives:

- **Two polarities, not one reliability.** A single `(Reliable s)` read as
  "s's claims are correct" expresses an inverted source (reliability below
  0.5, no `Inverts s` needed) but not a biased one: an optimist's up-claims
  are weak and its down-claims strong. Read as "s tells the truth, otherwise
  its claim is noise" (P(X | not R) = the node's prior), it cannot express
  an inverted source at all. Two polarities with the claim decided by the
  reliability express every archetype.
- **Certain per-claim rules.** Given the reliability, a claim decides its
  node; the source's uncertainty is all in the reliability. A per-claim CTV
  would have to be learned from one instance.
- **`no_inverse`.** Inverting a claim rule over a labelled outcome makes the
  outcome a certain view of the source-wide statement, true for some claims
  and false for others; the history would be contradictory certain evidence
  of one statement. Learning the reliability from the outcomes needs a
  statement per claim instead.
- **Evidence facts rather than a learned implication.** The client labels
  each resolved claim correct or not, as it labels the states. Learning it
  inside the chainer needs a per-claim latent `(Reliable s4 c1 t5)` with a
  prior per claim, which only a per-source rule (that is `pln-sources`) or
  a base rate read as a premise's prior (rows read no population prior)
  could give.

What PLN computes then differs from the world in one way. `(Reliable s4 up)`
is one statement shared by all of s4's up-claims: when two views of a node
share it (s4 claimed a system and one of its components), overlap
conditioning conditions on it, P(R) P(w | R) + (1 - P(R)) P(w | not R), so
s4's errors are fully correlated where the world's are independent per
claim, and with a certain rule the given branch lets s4's claim decide the
node, other sources aside. In `pln-sources` the per-source quantity is a
rule's truth, a parameter, and given it the claims' errors are independent,
as in the world.

### PLN, likelihood rules (`pln-likelihood`): the generating direction

The claims of `pln-sources`, but each source's refined rule runs from the
node's state to the claim, the way the world generates it:

    (: report-s4 (Implication (Up $n $t) (Claims s4 (Up $n $t))) (CTV (STV 0.5 0.02) (STV 0.5 0.02)))

Refined, it learns P(claim up | up) from the labelled nodes that were up and
P(claim up | down) from those that were down, exactly the rates exact-learned
counts; a current claim updates its node by inverting the rule over the base
rate of `Up` (Bayes' rule), several sources' claims combine as updates of one
prior (the factored merge), and a system's forward view takes the base
rate's place under its own claims. It is the encoding whose semantics match
the generating model: a likelihood per source and polarity, naive Bayes
given the node, independent errors given the rates. Reviewed like
`pln-sources`, before the round's claims.

### Encodings tried and rejected

| encoding | result |
|---|---|
| `(Implication (Reports s1 $x) $x)`, the claim is the statement | no answer at all: neither the rule truth nor the claimed statement (`examples/pettachainer_trust_rule_forms_repro.py`, part 1) |
| two refined STV rules per source, `(Asserts s1 (Up $n $t))` → `(Up $n $t)` and `(Denies s1 ...)` → `(Up $n $t)` | both rule truths are learned right from the instances (16 of 20 → 0.798), but a view applied to a certain claim is pulled to the base rate (0.666), and in the game some views came out at confidence 10⁻⁶ (part 2) |
| flat `(Asserts s1 c1 t5)` | the same as the nested STV form |
| `pln-stated` with the learned confidences instead of 1 | 0.334 against 0.237 at s seed 1 (before the history-order fix below): a node's own claims dropped in favour of a system rule's view that uses the same source (`examples/pettachainer_shared_rule_overlap_repro.py`) |
| `pln-stated` with the trust rules added after the history | 0.237 against 0.147 at s seed 1: the history's outcomes are stored before the rules, so they do not score the combination modes and every merge is plain revision |

### ProbLog (`problog-naive`, `problog-oracle`, `problog-learned`)

`src/conflict/problog_backend.py`, one program per round, ground, compiled
(d-DNNF) and evaluated in a forked child (`supplynet.problog_backend.solve`;
ProbLog 2.3.0 in `ChainerGame-problog/.venv-problog`).

- **oracle**: the generating model written out, conditioned on the claims:

      0.597::up(c1).  0.847::up(c2).
      both(w1) :- up(c1), up(c2).
      0.864::up(w1) :- both(w1).   0.121::up(w1) :- \+both(w1).
      0.882::claim(7) :- up(c1).   0.109::claim(7) :- \+up(c1).
      evidence(claim(7), false).
      query(up(c1)).  ...

  one `claim` per source and node (copies dropped), the two clauses of a
  claim or a system having exclusive bodies, so each is an exact CTV.
- **learned**: the same program with the Laplace estimates of the priors and
  of every source's two rates from the labelled rounds, re-estimated each
  round.
- **naive**: every claim copy is a probabilistic fact about its node with the
  pooled share t of correct claim copies in the history. A node is up when
  something supports it (its prior, or any up-claim: noisy-OR) and no
  down-claim denies it (noisy-OR of denials); a system is also supported
  through its rule:

      0.6::support(c1).  0.58::support(c1).  ...  0.58::denied(c1).
      up(c1) :- support(c1), \+denied(c1).

  No identity, copies counted: what "each report as a probabilistic fact"
  gives.

`test_oracle_is_exact_and_learned_is_exact_learned`: per key, ProbLog oracle
equals `exact` and ProbLog learned equals `exact-learned` to 10⁻⁹ (sizes s
and m); over the grid ProbLog oracle's mean error to `exact` is at most 5·10⁻¹¹ and ProbLog learned equals `exact-learned` to 10⁻¹¹.

### NARS (`nars-raw`, `nars-sources`)

`src/conflict/nars.py`, through the SupplyNet translation
(`supplynet.nars`, `docs/nars_backend.md`), ONA v0.9.3, a fresh `NAR shell`
per round: the system rules (positive branch `<(<(c1 * $t) --> up> &&
<(c2 * $t) --> up>) ==> <(w1 * $t) --> up>>`, the negative one stated from
each parent's negation), each node's base rate as induction over the labelled
rounds, the round's claims, budget × 1 cycles, one question per node.

- **raw**: a source's claims on a node as one judgement, `<(c1 * t5) --> up>.
  %1;0.75%` (`(! ...)` for "down"), confidence n/(n + k).
- **sources**: `<(s4 * c1 * t5) --> claims>. %1;0.99%` (negated for "down"),
  one per source and node, and per source
  `<<(s4 * $n * $t) --> claims> ==> <($n * $t) --> up>>. %f;c%` and the same
  from the negated claim (and both with `(! up)` heads), f the share of up
  nodes among the source's labelled claims of that polarity and c = n/(n+1):
  what NARS induction plus revision over the labelled rounds would sum to,
  computed by the adapter as SupplyNet's NARS backend computes base rates.

### Python references

| contender | what it is |
|---|---|
| `exact` | the posterior under the true model |
| `exact-learned` | the same with Laplace estimates from the labelled rounds (= ProbLog learned) |
| `prior` | each node's frequency of being up in the labelled rounds |
| `vote` | share of claim copies saying up (prior when silent): count-weighted revision |
| `last-wins` | the last claim to arrive, read with the pooled share of correct claims |
| `loudest` | only the source with the most claim copies in the history, read with its own accuracy |
| `trust-mean` | the mean of the speaking sources' P(up \| claim), each estimated from the labelled rounds: revision of learned trust views, no system rules; what the sources encoding reaches by revision alone |

## Results

October 10, 2026: ChainerGame `conflicting-sources`, PeTTaChainer master
b43ad3de, ProbLog 2.3.0 (d-DNNF), ONA v0.9.3. Seeds 1–4, 30 labelled rounds
then 10 rounds, evidence k = 5, steps per query 1, 4 and 16. Each run in its
own process under `ulimit -v 16000000`; other agents' benchmarks shared the
machine, so seconds are indicative. Full per-size tables (component, system,
contested and agreed error, coverage, Brier):
`benchmark-runs/conflict-2026-10-10/report.md`; machine-readable rows:
`/nexus/Dev/OpenCog/bench/results/conflict/summary.json`.

### Error to the exact posterior

At 4 steps per query, mean ± std over seeds (budget-free contenders have
one value). Lower is better; 0.30–0.40 is where a reasoner that ignores the
claims sits (`prior`).

| contender | s | m | l | xl |
|---|---|---|---|---|
| exact, problog-oracle | 0.000 | 0.000 | 0.000 | 0.000 |
| exact-learned, problog-learned | 0.053 ± 0.020 | 0.054 ± 0.017 | 0.020 ± 0.007 | 0.017 ± 0.005 |
| **pln-stated** | **0.123 ± 0.036** | **0.197 ± 0.072** | **0.097 ± 0.037** | **0.092 ± 0.034** |
| pln-given (ablation) | 0.118 ± 0.035 | 0.144 ± 0.043 | 0.068 ± 0.009 | 0.073 ± 0.013 |
| trust-mean (Python) | 0.243 ± 0.042 | 0.276 ± 0.025 | 0.337 ± 0.015 | 0.333 ± 0.010 |
| **pln-raw** | **0.277 ± 0.036** | **0.268 ± 0.037** | **0.318 ± 0.017** | **0.313 ± 0.020** |
| **pln-sources** | **0.272 ± 0.047** | **0.324 ± 0.022** | **0.373 ± 0.022** | **0.366 ± 0.016** |
| nars-raw | 0.261 ± 0.013 | 0.295 ± 0.025 | 0.336 ± 0.006 (1 step/query) | — |
| nars-sources | 0.277 ± 0.050 | 0.307 ± 0.041 | 0.410 ± 0.017 (1 step/query; coverage 0.70) | — |
| prior | 0.278 ± 0.050 | 0.308 ± 0.040 | 0.380 ± 0.026 | 0.385 ± 0.024 |
| vote | 0.343 ± 0.027 | 0.314 ± 0.037 | 0.324 ± 0.010 | 0.333 ± 0.017 |
| last-wins | 0.324 ± 0.034 | 0.320 ± 0.022 | 0.401 ± 0.019 | 0.402 ± 0.012 |
| loudest | 0.310 ± 0.020 | 0.330 ± 0.032 | 0.401 ± 0.016 | 0.413 ± 0.015 |
| problog-naive | 0.320 ± 0.030 | 0.343 ± 0.038 | 0.491 ± 0.045 | 0.501 ± 0.036 |

NARS at size l ran at 1 step per query only (about a minute per round); not
at xl.

The exact posterior is sharp (its Brier score is 0.10 / 0.09 / 0.055 / 0.054
at s / m / l / xl), because the expert, the contrarian and the biased sources
are very informative once read correctly. Everything that does not read them
correctly lands near the prior.

**Budget.** No contender's error moves with the budget by more than 0.01
(pln-stated at l: 0.097 / 0.097 / 0.093 at 1 / 4 / 16 steps per query;
pln-sources at xl gets worse at 16, 0.383 against 0.366, as more views are
found and dropped). The searches are shallow: a node's views are one rule
away. The budget buys time only.

### Cost

Seconds per round at 4 steps per query (round = the backend's whole work for
the round's 4 / 8 / 16 / 32 queries: reviews, queries, adding facts; ProbLog:
ground + compile + evaluate in a child; NARS: one ONA run).

| contender | s | m | l | xl |
|---|---|---|---|---|
| exact (Python enumeration) | 0.000 | 0.000 | 0.001 | 0.51 |
| problog-oracle / learned | 0.09 | 0.10 | 0.26 | 1.5–1.6 |
| problog-naive | 0.09 | 0.11 | 0.47 | 13.8 (2 rounds timed out at 60 s) |
| pln-stated | 0.05 | 0.15 | 0.57 | 1.66 (+ setup 0.7 / 2 / 9 / 28 s: the trust queries) |
| pln-raw | 0.05 | 0.16 | 0.79 | 40.2 (seed 2: 144 s, 10 GB; at 16 steps it timed out at 1 h) |
| pln-sources | 0.74 | 2.81 | 11.0 | 32.1 (reviews are most of it) |
| nars-raw / nars-sources | 0.74 / 1.07 | 11.1 / 19.1 | 44 / 148 (1 step/query) | — |

ProbLog's exact compilation stays cheap here: components are independent a
priori and each system has two parents, so the circuits are small; at xl
(16 components, 32 queries) a round takes 1.5 s, three times Python's
enumeration over 2^16 states. This stage has no size at which exact inference
blows up the way SupplyNet's scale stage does; xl shows the cost growth of
the reasoners, not a regime where exact fails.

## Discussion

### What PLN gets for free (raw)

Count-weighted revision is a vote by copies. Without source identity it
cannot tell the expert from the echo or read the contrarian's "up" as "down",
and the echo's 8–15 copies make it the loudest voice: `vote` is at or above
the prior's error at every size, and its error on agreed nodes is low (0.08–
0.18 at m–xl) while contested nodes stay at 0.34–0.37. `pln-raw` is `vote`
plus the system rules and their inversions, which buy 0.01–0.07 (0.268
against 0.314 at m). The same holds for NARS raw (revision by evidence weight
is the same vote) and for ProbLog naive, which is worse: noisy-OR over copies
saturates at 1 − (1 − t)^n, so a source repeating itself ten times is
near-certain, and one denial copy vetoes it (0.49–0.50 at l–xl).

### Where PLN wins: sources encoded, rates stated (pln-stated)

With the claim attached to its source and one trust CTV per source learned
from the labelled claims, the chainer reads the contrarian inverted, the
optimist's "up" as weak and its "down" as strong, and the echo as noise. The
error falls to a third to a half of raw (0.092–0.123 at s, l, xl; 0.197 at m), it is
the best non-exact contender at every size, it improves with size (more
claims per source to learn from and more sources per node), and it costs
about as much as raw (0.05–1.7 s per round). Two chainer mechanisms do the
work:

- **combination modes**: the labelled outcomes stored after the trust rules
  score revision against odds, and odds wins: views of different sources on
  one node merge as `(combination odds ...)` (naive Bayes over the trust
  views), not by averaging. `trust-mean`, the Python revision of the same
  kind of views, stays at 0.24–0.34, near the prior: averaging P(up | claim)
  views throws away exactly the agreement that makes the exact posterior
  sharp. Stating the rules after the history (no outcomes to score, plain
  revision) gave 0.237 instead of 0.147 at s seed 1;
- **factored revision with antecedent completion**: a system's claims update
  its components and back, through the given system rules.

### Where PLN loses, and why

1. **Against exact with learned rates: 0.07–0.14 behind.** `exact-learned`
   (= ProbLog learned) reaches 0.017–0.054 from the same labelled data. The
   `pln-given` ablation, which hands the chainer the Laplace-counted CTVs
   instead of its own, separates the causes:
   - *learning the rates*: 0.02–0.05 (pln-stated − pln-given: 0.005 at s,
     0.053 at m, 0.029 at l, 0.019 at xl). The negative branch of a learned
     CTV comes from base rates, not from the claims of that polarity: P(up |
     s claims down) is (P(up) − P(claim up) P(up | claim up)) / (1 − P(claim
     up)), at confidence about 0.2 whatever the sample. At s seed 1 the
     expert's "down" reads 0.00 (its 3 labelled "down" claims say 0.33), the
     contrarian's 0.99 (0.86); the positive branches equal the empirical
     frequencies. Stated as certain, a hard 0 or 1 makes one claim decide a
     node. The rates are also learned once and not updated with the 10
     resolved rounds;
   - *combining*: 0.05–0.09 (pln-given − exact-learned). Odds is applied
     only between forward views with disjoint evidence; views that meet a
     completion or a system rule's view are revised, and revision of certain
     views weights each pair equally whatever the evidence behind them
     (CombiNet "Chainer problems" 1). One mode per predicate: components and
     systems share `Up`.
2. **The hypothesis-rule encoding (`pln-sources`) is no better than raw, and
   at l–xl worse than the prior (0.37).** Its trust rules learn the right
   positive branches, but every view of a refined rule carries the rule's
   samples, so two applications of one source's rule to different nodes are
   overlapping evidence: a node's own claims are dropped whenever the search
   also reaches it through a system rule whose proof uses one of the same
   sources (see Chainer problems 1). The merge then keeps the most confident
   single view, typically a system rule's view or a completion that ignores
   the node's own claims. It also costs 10–30 s per round at l–xl, mostly in
   the per-source reviews, and gets combination modes on none of its views
   (refined rules are not scored). This is the encoding the chainer's
   design intends for learned rules, and today it is the wrong one for this
   problem.
3. **Rare sources.** A source with a few labelled claims gets a CTV stated at
   confidence 1 from those few (or no rule at all with none), where
   exact-learned's Laplace estimate stays near uninformative.

### NARS

NARS raw is a vote by evidence weight, like PLN raw, and lands at the same
place (0.26 / 0.30 / 0.34 at s / m / l). Given the same identity and trust
counts, NARS sources does not improve on it (0.28 / 0.31 / 0.41), for the
reasons `docs/nars_backend.md` found in SupplyNet: deduction through a trust
implication gives f = f_claim · f_rule at a confidence that shrinks with
each factor, revision then averages the sources' conclusions by evidence
weight instead of multiplying likelihood ratios, and conclusions from the
negated-claim implications are rarely selected. At l the concept table is
full: 30% of the questions go unanswered (scored 0.5) and a round takes
2.5 minutes. More cycles change nothing (s and m: identical error at 1, 4 and
16 steps per query) and only cost time (m: 11 s → 32 s per round for raw).

### What the comparison says

- Identity matters more than inference power. Every contender without the
  source identity (vote, last-wins, loudest, PLN raw, NARS raw, ProbLog naive)
  is at or above the prior's error; ProbLog naive, an exact engine, is the
  worst of all, because its model is wrong.
- With identity and learned reliabilities, the exact model (ProbLog learned)
  is the ceiling at 0.02–0.05, and cheap here (0.1–1.6 s per round).
- PLN with stated learned trust is the best of the approximate reasoners,
  0.09–0.20, and recovers most of the gap from raw (0.27–0.32). The remaining
  gap is half rate learning (negative branches from base rates) and half
  combination (odds only for disjoint forward views; equal-weight revision of
  certain views).
- PLN's own learned-rule machinery (refined hypothesis rules) does not work
  for this problem today because of how refined views overlap.

## After the PeTTaChainer fixes (branch `conflict-fixes`)

October 10, 2026. PeTTaChainer branch `conflict-fixes` at 46d6fd50 against
master 66a684f1; ChainerGame `pln-reliable` 83bbee5 (the client changes
below apply to both columns). Seeds 1–4, 4 steps per query, raw runs in
`/nexus/Dev/OpenCog/bench/results/conflict-fixes/` (`master/`, `fixes/`,
`fixes-review400/`). Error to the exact posterior, mean ± std over seeds.

| contender | s | m | l | xl |
|---|---|---|---|---|
| exact-learned | 0.053 | 0.054 | 0.020 | 0.017 |
| pln-stated, master | 0.123 ± 0.036 | 0.196 ± 0.070 | 0.097 ± 0.037 | 0.092 ± 0.034 |
| **pln-stated, fixes** | **0.100 ± 0.022** | **0.147 ± 0.055** | **0.079 ± 0.028** | 0.103 ± 0.062 |
| pln-sources, master (reviews before claims) | 0.279 ± 0.046 | 0.323 ± 0.029 | 0.374 ± 0.014 | 0.374 ± 0.019 |
| pln-sources, fixes | 0.167 ± 0.086 | 0.300 ± 0.021 | 0.349 ± 0.013 | 0.326 ± 0.041 |
| pln-sources, fixes, 400 review steps | 0.105 ± 0.026 | 0.169 ± 0.088 | — | — |
| pln-reliable (master = fixes) | 0.153 ± 0.084 | 0.219 ± 0.074 | 0.103 ± 0.008 | 0.109 ± 0.016 |
| **pln-likelihood, fixes, 400 review steps** | 0.117 ± 0.004 | **0.105 ± 0.018** | **0.035 ± 0.010** | — |

Seconds per round (fixes): pln-stated 0.05 / 0.15 / 0.63 / 1.95, pln-reliable
0.10 / 0.57 / 0.73 / 1.60, pln-sources 0.9 / 4.5 / 14.8 / 78 (master 0.7 /
2.0 / 7.3 / 19), with 400 review steps 2.0 / 10.5; pln-likelihood (400) 3.1 /
17.7 / 155. The xl runs at 400 review steps and pln-sources at l were not run
(the l runs failed on a broken intermediate build; xl would exceed the hour).

- **pln-stated** gains 0.02–0.05 at s–l from the negative branches learned
  from the down-claims (problem 2). At xl seed 4 it loses (0.143 → 0.195):
  two rare sources' negative branches are now their one or two labelled
  down-claims, 0.0, stated at confidence 1, so one claim decides a node
  (the "rare sources" weakness below, sharper now that the branch is
  empirical).
- **pln-sources** was broken by more than problem 1. With it fixed (no view
  dropped), three more causes showed: reviews made after the round's claims
  read the prior (fixed in the client: review first); refined rules were
  never scored for the combination modes (fixed in PeTTaChainer 46d6fd50),
  so the views averaged; and 20 review steps fold only part of a source's
  samples beyond size s (ten rows per step; an echo has hundreds of claims).
  With 400 review steps it matches pln-stated at s and m, and keeps learning
  from the resolved rounds; it costs 2–10 s per round.
- **pln-reliable**, the per-claim design, sits between: better than raw and
  than pln-sources at 20 review steps, behind pln-stated at s–l, level at xl.
  Its reliabilities are the trust CTV's branches as counted statements, so
  it learns what pln-stated learns, from every resolved round, cheaply; what
  it loses is the shared latent (errors correlated through `Reliable s`, see
  its section) and the component priors.
- **pln-likelihood** is the best approximate reasoner at m and l (0.035
  against exact-learned's 0.020 at l, a third of pln-stated's error): it
  learns the likelihoods exact-learned counts and combines claims by Bayes'
  rule over the base rate, with the systems' forward views as priors. At s
  it trails pln-stated (components get the pooled base rate of `Up`, not
  their own prior; 0.173 component error against 0.061 system). It is slow:
  every claim is an inversion over a base rate, 155 s per round at l.

## Chainer problems

Found while building the backends. PeTTaChainer master b43ad3de; status on
branch `conflict-fixes` (66a684f1 + 5 commits) after each item.

1. **Applications of one rule to different facts are treated as overlapping
   evidence** unless the rule is certain, so a merge keeps only the more
   confident view. `examples/pettachainer_shared_rule_overlap_repro.py`: w's
   own claim by source s is dropped from the answer for `(Up w n)` because
   the forward view through the system rule uses `trust-s` on c1; the same
   claim by source q is revised in; at trust confidence 1 both revise. For
   refined rules this is by design (their views carry the samples), and it is
   what breaks `pln-sources`. In the game it also replaces a node's own
   claims by an antecedent completion through a system rule when the budget
   is larger (c2 at s seed 1 round 1: 0.138 from its claims, exact 0.132, at
   4 steps; 0.820 from the completion alone at 50 steps).
   **Fixed (5e4fd377).** Given the rule's truth its applications are
   independent, as rows already assumed; a shared rule, given or estimated,
   no longer makes views dependent, and a refined rule's `RuleTruth` key is
   a parameter, not a statement to condition on. The repro gives 0.568 for
   both claimants. Not conditioning on the rule (a mix over its Beta) is
   deliberate: given the rule, views of certain claims are certain and
   revision weighs them equally (PeTTaChainer
   `docs/metta/overlap_conditioning.md`, "Shared rules"). The rule's
   uncertainty is counted once per view, so a merge of many views through
   one estimated rule is overconfident.
2. **The negative branch of a learned CTV comes from base rates**, not from
   the instances whose antecedent is false: biased (0.00 against 0.33, 0.99
   against 0.86 above) and at confidence about 0.2 regardless of the sample
   size (47 "down" claims of the echo still give 0.23). A claim fact with
   strength 0 is a sample of the negative branch; it is not used as one.
   **Fixed (8c4adc3e).** The instance fold accumulates both branches, each
   sample weighted by the antecedent's strength for P(B | A) and by its
   complement for P(B | not A); the base rates give the negative branch only
   while no antecedent-false instance exists. 10 down-claims with 3 up now
   give (0.30, 0.67).
3. **A variable consequent is not supported.** `(Implication (Asserts s $x)
   $x)`, the most direct way to say "what s asserts holds", gives no answer,
   neither for its rule truth nor for a claimed statement
   (`examples/pettachainer_trust_rule_forms_repro.py`, part 1).
   Not to be fixed: bare-variable consequents are not wanted.
4. **Views of STV rules applied to a certain antecedent are pulled towards
   the consequent's base rate**: a refined STV rule learned at 0.798 gives a
   view of 0.666, a stated 0.8 gives 0.775 (same file, part 2). In the game
   some refined STV views came out at confidence 10⁻⁶.
   **Cause and fix (5203822a).** Neither problem 1 nor 2, nor the Bayes
   prior: `Asserts` stores only asserted claims, so its base rate is 1, and
   the STV-to-CTV conversion reconciles P(B | A) with P(A) = 1 and P(B) =
   0.5 (ConsistentTriple), which forces P(B | A) to P(B). Declaring the
   predicate positive-only (`set_positive_only_predicate`) now keeps its rate
   out of the conversion, and the rule applies at 0.798; undeclared, the
   data still says P(A) = 1. The confidence 10⁻⁶ views were factored merges
   over a refined rule's prior (0.5, 0.02), read when the review had folded
   the round's own claims (see 6).
5. **pln-raw at xl, seed 2**: 144 s per round and 10 GB at 4 steps per query
   (the other seeds 1.7–11 s, 0.2–0.9 GB), and a timeout at 1 h at 16 steps.
   Not reduced to a repro: the KB holds one uncertain fact per source and
   node beside the system rules.
6. **A refined rule reviewed with the round's claims in its fold is unusable
   for those claims** (found with the fixes). A claim whose node the round's
   other views derive is a soft sample of its rule, so the fold depends on
   the node; a view of the node may not read a value depending on itself, so
   it falls back to an older version of the rule truth, the prior when the
   review was the first. Worked around in the client (reviews first). Same
   root as ChainerGame `benchmarks/ona/repros/stale_after_early_review.py`.
   A fix would fold a rule's samples for a reader without the reader's own
   sample (leave one out) instead of rejecting the whole version.
7. **Refined rules were never scored for the combination modes** (found with
   the fixes). The truth premise is an open statement, which gave no view
   when scoring an outcome, so modes tied and the views revised. **Fixed
   (46d6fd50):** the premise reads the truth key's current value; pln-sources
   at s seed 1 0.322 → 0.164.

## Limits

- **One relation between sources and truth.** Claims are conditionally
  independent given the node; sources do not copy each other (beyond
  repeats of their own claim) and their reliability does not depend on the
  node. A copying source would make odds over-count, where revision is safer.
- **Rates are learned once in pln-stated and pln-given** (from the 30 labelled
  rounds); the other learned contenders re-learn every round. Re-stating a
  rule needs the deprecated `remove_statement`.
- **Exact never becomes expensive** in this stage (see Cost). A world where it
  does (sources whose reliability depends on a hidden per-round context, or
  components that persist across rounds) would show the cost side.
- **NARS gets its trust implications as counts computed by the adapter**, as
  SupplyNet's NARS backend gets base rates; ONA would not form them itself.
  NARS ran at l only at 1 step per query and not at xl.
- 4 seeds; time measurements shared the machine with other agents.

## Rerun

    cd ChainerGame-conflict
    out=/nexus/Dev/OpenCog/bench/results/conflict
    G=benchmarks/conflict/run_grid.sh
    JOBS=2 $G $out/runs "exact exact-learned prior vote last-wins loudest trust-mean problog-oracle problog-learned problog-naive pln-raw pln-stated pln-given pln-sources" "s m l xl" "1 2 3 4" "1 4 16"
    JOBS=2 $G $out/runs "nars-raw nars-sources" "s m" "1 2 3 4" "1 4 16"
    JOBS=2 $G $out/runs "nars-raw nars-sources" "l" "1 2 3 4" "1"
    PYTHONPATH=src python3 -m conflict.cli table $out/runs > benchmark-runs/conflict-2026-10-10/report.md
    PYTHONPATH=src python3 -m conflict.cli summary $out/runs --out $out/summary.json

After the PeTTaChainer fixes (`W` the PeTTaChainer checkout, `M` master's):

    out=/nexus/Dev/OpenCog/bench/results/conflict-fixes
    PYTHON=$W/.venv/bin/python PETTACHAINER_PATH=$W JOBS=4 $G $out/fixes "exact-learned pln-reliable pln-stated pln-sources" "s m l xl" "1 2 3 4" "4"
    PYTHON=$M/.venv/bin/python PETTACHAINER_PATH=$M JOBS=4 $G $out/master "pln-reliable pln-stated pln-sources" "s m l xl" "1 2 3 4" "4"
    REVIEW_STEPS=400 PYTHON=$W/.venv/bin/python PETTACHAINER_PATH=$W JOBS=4 $G $out/fixes-review400 "pln-likelihood pln-sources" "s m l" "1 2 3 4" "4"

A grid imports the PeTTaChainer checkout it is given when each run starts:
do not edit that checkout while it runs.

Tests: `PYTHONPATH=src <problog venv>/bin/python -m pytest tests/test_conflict.py`
(ProbLog checks skip without ProbLog); the live PeTTaChainer and ONA tests run
in PeTTaChainer's venv with `PETTACHAINER_PYTHONPATH` set:
`PYTHONPATH=src:tests .../PeTTaChainer/.venv/bin/python -m unittest test_conflict`.
