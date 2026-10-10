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
from the instances, P(up | s4 claims down) from the base rates. Before each
round's queries every trust rule is reviewed by its own `RuleTruth` query (20
steps), so the last round's labels are folded in. The answer for a node is
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
and m); over the grid the largest difference is below 10⁻¹².

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

RESULTS
