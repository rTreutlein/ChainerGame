# Actions in the belief knowledge base

Status: built, October 2026 (`actions.py`: `BELIEF_ACTION_RULES`, `leak_belief_link`,
`generate_decision_statements`; `PeTTaChainerBackend.propose_actions` and
`observe_inspection`).

## Problem

The PeTTaChainer action reasoner (`PeTTaChainerBackend.propose_actions`) builds
a new `PeTTaChainer()`, and so a new knowledge base, for every decision
context `decision-sNN-stepMM`: about three per shift. PeTTaChainer has no way
to drop a knowledge base, so every context's facts, skeleton rows, views and
forward agenda stay in the shared runtime for the rest of the game.

Measured on a live temporal-model game (seed 1, query budget 50, forward slice
150):
- After 20 shifts the 60 action knowledge bases hold 70% of all skeleton rows
  (4,985 of 7,125) and 82% of all `&kb` facts.
- Over 60 shifts, memory grows 241 MB → 1,161 MB (about 16 MB per shift).
  Replaying the same game without the action reasoner, it grows 190 MB → 508 MB.
- Belief forward time per shift rises 0.27 s → 0.52 s with the action
  contexts, and 0.25 s → 0.34 s without them.
- The action reasoner itself takes 1.3–2.6 s per shift, two thirds of all
  reasoning time. Each context adds and compiles the same six rules again,
  then forward-chains about 57 facts.

## Why the separate knowledge base exists

`actions.py` gives the reason:

> ordinary chainer premises do not expose a proof's strength as a numeric
> variable. The adapter therefore asserts the current, context-scoped
> probability as an explicit term before asking one open `ActionProposal`
> query.

So the game queries the beliefs and copies each incident's probability and
confidence into certain facts (`LeakProbability`, `BeliefConfidence`). It then
runs arithmetic rules over the copies. After an inspection the game sets that
incident's belief to 1 or 0 in Python and builds the next context from the
changed numbers. A fresh knowledge base per context was the simple way to start
each decision from its own copies.

The premise of that docstring no longer holds. `FoldAllTruth` reads a
statement's truth value as data (`tests/metta/test_foldall_truth.metta` in
PeTTaChainer reads one belief's strength this way), and the temporal model
already uses it for the shortfall posterior.

## Design

One knowledge base holds the beliefs and the decisions. The action rules read
the beliefs; nothing is copied.

### Outcomes are logical consequences of the beliefs

An incident's belief is queried as a statement whose form depends on the game
mode: `(SealLeak c t u)` in the temporal model, `(LocalProblem …)` with
module dependencies, or a class-level `(Inheritance …)` otherwise. One rule
per incident names that statement as its leak belief:

    (: (no_inverse leak-belief-u) (Implication <belief of u> (LeakBelief u)) …)

An action's outcomes are statements whose truth follows from the leak belief
through ordinary connectives:

    (Implication (And (DecisionCandidate $step $u) (LeakBelief $u))
                 (Outcome $step (Repair $u) protected))
    (Implication (And (DecisionCandidate $step $u) (Not (LeakBelief $u)))
                 (Outcome $step (Repair $u) unnecessary))

Their probabilities are P(leak) and 1 − P(leak), with the belief's
confidence. The chainer derives them
from `SealLeak` as it derives any belief: a belief that changes leaves its
outcome rows stale, and the next decision query recomputes them.

### Amounts are certain facts

What an outcome is worth comes from the game's numbers, stated once per
decision step:

    (OutcomeValue $step (Repair u) protected 120)
    (OutcomeValue $step (Repair u) unnecessary -30)
    (ActionCost $step (Repair u) 25)

These are plain facts, not copies of beliefs.

### Expected utility is a fold over truth values

    (Implication
       (And (DecisionCandidate $step $c $t $u)
            (FoldAllTruth (And (Outcome $step (Repair $u) $o)
                               (OutcomeValue $step (Repair $u) $o $value))
               (STV $s $_conf) (* $s $value) 0.0
               (|-> ($acc $elem) (+ $acc $elem)) -> $gross)
            (ActionCost $step (Repair $u) $cost)
            (Compute - ($gross $cost) -> $utility))
       (ActionValue $step (Repair $u) $utility))

The fold weights each outcome's amount by the outcome's truth value. That is
the expected utility, with no new construct: a weighted sum over statements
is what `FoldAllTruth` does. The certain `OutcomeValue` conjunct leaves each
outcome's strength unchanged.

Inspection is valued the same way. Its outcomes are "a leak is found and
repaired" (worth risk − repair cost) and "no leak is found" (worth nothing).
The value of the information is that, minus the best value without the
inspection (max(0, repair utility), a `Compute`), minus the inspection cost.
The learning probe needs only certain facts and stays as it is.

The truth value of `ActionValue` is certain; its utility is the number the
proposal carries, as `ActionProposal` does now. The proposal's confidence is the
confidence of the incident's belief, read by a `FoldAllTruth` over
`(SealLeak $c $t $u)`, as in the PeTTaChainer test.

### Decision state is indexed by step

What changes during a shift is stated as new facts under the step id:
- `(DecisionCandidate step c t u)`
- `(InspectionEligible step u)`
- the outcome amounts and costs

Nothing is replaced: a later step adds its own facts, as shifts do for beliefs.
Rules, including the action rules, are added once per game.

### An inspection result settles the leak belief

Today an inspection result only changes a number in Python
(`effective_beliefs`). Here it becomes an observation in the knowledge base,
`(: inspection-u (LeakBelief u) (STV 1 1))` or the same with `(STV 0 1)`
(`observe_inspection`, called by the action loop). The certain observation
outweighs the derived belief, and the incident's outcomes and the next
step's values follow from it.

The observation is stated on `LeakBelief`, not on the belief statement: in
the class-level mode that statement is about every incident of the class.
The belief model learns the result at the shift's end from the resolved
leaks, as it does for every resolution (ChainerGame fb45e44), so beliefs and
their metrics are unchanged.

### The choice stays in the game

The game queries `(ActionValue step $action $utility)`, or keeps
`ActionProposal` as the conclusion. It then picks the best action as now, with
the same ordering and tie-breaks.

## Forward chaining and budgets

- Decision facts are not forward seeds: the game queues only belief facts, as
  `_forward_fact` selects them now. Action values are derived by the action
  query only, within its budget.
- The rules' heads (`Outcome`, `ActionValue`) are read by no belief rule, so
  dormant rules keep belief forward runs off them.
- PeTTaChainer compiles an inverted rule for every implication, from the
  conclusion back to a premise, unless the rule is named `(no_inverse …)`:
  all action rules and links are. A rule whose premise is a pure conjunction
  still gets antecedent-completion rules, which derive one conjunct from the
  conclusion and the others (PeTTaChainer keeps them under `no_inverse` on
  purpose, `test_antecedent_completion.metta`). For the action rules they
  conclude `LeakBelief` or `DecisionCandidate` from an `Outcome` that is
  derived only from those same premises, so they can only support
  themselves and are dormant (PeTTaChainer `docs/metta/dormant_rules.md`).

## What goes

- The per-context action handler, its knowledge base and
  `_action_atoms_by_name`.
- `LeakProbability`, `BeliefConfidence` and the Python-side belief override
  after inspections.
- `_new_handler` settings applied to a second handler. Rule refinement skips
  the action rules in the shared knowledge base, since their confidence is 1.
- The docstring's premise in `actions.py`.

`ACTION_RULES` and `generate_action_statements` stay for the MM2 backend.
`reference_action_proposals` stays: it is the oracle the action rules are
tested against (`test_live_pettachainer_actions_match_reference_when_available`:
equal utilities and confidences for repair, inspection and learning probes,
before and after an inspection).

## Risks

- **`Not (SealLeak …)`.** It must give 1 − P(leak) with the belief's
  confidence. Check the compiled negation's truth-value formula on a belief
  that is a merged view with a prior.
- **Self-support.** The only path from an action predicate back to a belief
  is the dormant antecedent completions above, and they end at `LeakBelief`,
  which no belief rule reads.
- **Cost per decision.** Each step's query computes the outcome rows and folds
  for its candidates (about 10). That should be well below the 0.4 s per
  context spent now on compiling rules and forward-chaining copies; measure it.

## Evaluation

1. Unit tests in ChainerGame: for fixed beliefs, the action values equal
   `reference_action_proposals`, for repair, inspection and learning probes,
   including an incident inspected in an earlier step.
2. Live games: the same 4 seeds as the comparison of the `forward-units`
   branch, 20 shifts.
   - Regret and production should match or improve. Inspections becoming
     evidence may change later shifts.
   - Compute and peak memory should drop.
3. 60-shift live game: memory per shift close to the replay without actions
   (about 5 MB per shift), and action time per shift flat.
