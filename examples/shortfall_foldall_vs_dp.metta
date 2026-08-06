;; StationOps aggregate-loss inference: eager FoldAll versus an ordinary-rule
;; dynamic-programming chain.
;;
;; This file contains all data, rules, queries, and checks.  Its only dependency
;; is PeTTaChainer.  Adjust this import if the checkout is located elsewhere.
!(import! &self ../../NL2PLN_Project/PeTTaChainer/lib_pettachainer)

;; One exact observation has two possible explanations:
;;   pump alone       0.02 * 0.90 * 0.95 = 0.0171
;;   valve and motor  0.98 * 0.10 * 0.05 = 0.0049
;; Evidence mass is 0.022, so the normalized marginals are
;;   pump 0.7772727, valve 0.2227273, motor 0.2227273.


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Option 1: derive complete explanations, then FoldAll all their proof values.
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

!(compileadd foldKb (: foldPump (Candidate pump 5 0.02) (STV 1 1)))
!(compileadd foldKb (: foldValve (Candidate valve 3 0.10) (STV 1 1)))
!(compileadd foldKb (: foldMotor (Candidate motor 2 0.05) (STV 1 1)))
!(compileadd foldKb (: foldTarget (ObservedLoss 5) (STV 1 1)))

;; These two rules stand for the complete explanations found by normal proof
;; search.  Their weights include both failed and intact candidate branches.
!(compileadd foldKb
   (: foldPumpExplanation
      (Implication
         (And
            (Candidate pump 5 $pump-prior)
            (Candidate valve 3 $valve-prior)
            (Candidate motor 2 $motor-prior)
            (ObservedLoss 5)
            (Compute - (1.0 $valve-prior) -> $valve-ok)
            (Compute - (1.0 $motor-prior) -> $motor-ok)
            (Compute * ($pump-prior $valve-ok) -> $partial)
            (Compute * ($partial $motor-ok) -> $weight))
         (ExactExplanation pump-only $weight))
      (CTV (STV 1 1) (STV 0 1))))

!(compileadd foldKb
   (: foldValveMotorExplanation
      (Implication
         (And
            (Candidate pump 5 $pump-prior)
            (Candidate valve 3 $valve-prior)
            (Candidate motor 2 $motor-prior)
            (ObservedLoss 5)
            (Compute - (1.0 $pump-prior) -> $pump-ok)
            (Compute * ($pump-ok $valve-prior) -> $partial)
            (Compute * ($partial $motor-prior) -> $weight))
         (ExactExplanation valve-motor $weight))
      (CTV (STV 1 1) (STV 0 1))))

;; One fold accumulator retains the evidence denominator and all three
;; numerators.  It is still a global barrier, but avoids running two separate
;; exhaustive folds per requested module.
(= (AddWorldMass
      (Masses $evidence $pump $valve $motor)
      (WorldWeight pump-only $weight))
   (Masses
      (+ $evidence $weight)
      (+ $pump $weight)
      $valve
      $motor))
(= (AddWorldMass
      (Masses $evidence $pump $valve $motor)
      (WorldWeight valve-motor $weight))
   (Masses
      (+ $evidence $weight)
      $pump
      (+ $valve $weight)
      (+ $motor $weight)))

(= (NormalizeMasses (Masses $evidence $pump $valve $motor))
   (Marginals
      (/ $pump $evidence)
      (/ $valve $evidence)
      (/ $motor $evidence)))

!(compileadd foldKb
   (: foldMarginalsRule
      (Implication
         (And
            (FoldAll
               (ExactExplanation $selection $weight)
               (WorldWeight $selection $weight)
               (Masses 0.0 0.0 0.0 0.0)
               AddWorldMass
               -> $masses)
            (Compute NormalizeMasses ($masses) -> $marginals))
         (FoldMarginals $marginals))
      (CTV (STV 1 1) (STV 0 1))))


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Option 3: an ordinary-rule DP chain, queried backwards from the marginal.
;;
;; There is no FoldAll below.  Each proof carries one compact sparse DP row and
;; the next rule deterministically constructs the next row.  This is runnable
;; in the current chainer and exposes the extra recursive proof depth directly.
;;
;; The more granular target is one answer per (stage, accumulated-loss), merged
;; by numeric `sum` answer subsumption.  PeTTaChainer does not expose that merge
;; yet, so this example carries the small row as one term instead.  Importantly,
;; it still never constructs complete fault-set proofs.
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

!(compileadd dpKb (: dpPump (Candidate pump 5 0.02) (STV 1 1)))
!(compileadd dpKb (: dpValve (Candidate valve 3 0.10) (STV 1 1)))
!(compileadd dpKb (: dpMotor (Candidate motor 2 0.05) (STV 1 1)))
!(compileadd dpKb (: dpTarget (ObservedLoss 5) (STV 1 1)))
!(compileadd dpKb (: dpStart (DPRow stage0 (At0 1.0)) (STV 1 1)))

;; Pump maps {0:1} to {0:0.98, 5:0.02}.
!(compileadd dpKb
   (: dpPumpRow
      (Implication
         (And
            (DPRow stage0 (At0 $mass0))
            (Candidate pump 5 $prior)
            (Compute - (1.0 $prior) -> $not-prior)
            (Compute * ($mass0 $not-prior) -> $sum0)
            (Compute * ($mass0 $prior) -> $sum5))
         (DPRow stage1 (At0And5 $sum0 $sum5)))
      (CTV (STV 1 1) (STV 0 1))))

;; Valve maps that row to {0:0.882, 3:0.098, 5:0.018}.
!(compileadd dpKb
   (: dpValveRow
      (Implication
         (And
            (DPRow stage1 (At0And5 $mass0 $mass5))
            (Candidate valve 3 $prior)
            (Compute - (1.0 $prior) -> $not-prior)
            (Compute * ($mass0 $not-prior) -> $sum0)
            (Compute * ($mass0 $prior) -> $sum3)
            (Compute * ($mass5 $not-prior) -> $sum5))
         (DPRow stage2 (At0And3And5 $sum0 $sum3 $sum5)))
      (CTV (STV 1 1) (STV 0 1))))

;; For target 5, the motor needs only stage2 sums 5 (exclude motor) and 3
;; (include motor).  The row values for final sums 0, 2, and 3 are never built.
!(compileadd dpKb
   (: dpTargetRow
      (Implication
         (And
            (DPRow stage2 (At0And3And5 $_mass0 $mass3 $mass5))
            (Candidate motor 2 $prior)
            (ObservedLoss 5)
            (Compute - (1.0 $prior) -> $not-prior)
            (Compute * ($mass5 $not-prior) -> $pump-mass)
            (Compute * ($mass3 $prior) -> $pair-mass)
            (Compute + ($pump-mass $pair-mass) -> $evidence))
         (DPTargetMass 5 $evidence $pump-mass $pair-mass))
      (CTV (STV 1 1) (STV 0 1))))

!(compileadd dpKb
   (: dpMarginalsRule
      (Implication
         (And
            (DPTargetMass 5 $evidence $pump-mass $pair-mass)
            (Compute / ($pump-mass $evidence) -> $pump)
            (Compute / ($pair-mass $evidence) -> $pair))
         (DPMarginals (Marginals $pump $pair $pair)))
      (CTV (STV 1 1) (STV 0 1))))


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Correctness comparison.
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

!(test
   (let (: $_proof (ExactExplanation pump-only $weight) $_tv)
      (query 100 foldKb
         (: $_proof (ExactExplanation pump-only $weight) $_tv))
      $weight)
   0.0171)

!(test
   (let (: $_proof
            (FoldMarginals
               (Marginals $pump $valve $motor)) $_tv)
      (query 200 foldKb
         (: $_proof
            (FoldMarginals
               (Marginals $pump $valve $motor)) $_tv))
      (and
         (and (> $pump 0.7772) (< $pump 0.7773))
         (and
            (and (> $valve 0.2227) (< $valve 0.2228))
            (and (> $motor 0.2227) (< $motor 0.2228)))))
   true)

!(test
   (let (: $_proof
            (DPRow stage2 (At0And3And5 $sum0 $sum3 $sum5)) $_tv)
      (query 200 dpKb
         (: $_proof
            (DPRow stage2 (At0And3And5 $sum0 $sum3 $sum5)) $_tv))
      (and
         (and (> $sum0 0.8819) (< $sum0 0.8821))
         (and
            (and (> $sum3 0.0979) (< $sum3 0.0981))
            (and (> $sum5 0.0179) (< $sum5 0.0181)))))
   true)

!(test
   (let (: $_proof
            (DPMarginals (Marginals $pump $valve $motor)) $_tv)
      (query 200 dpKb
         (: $_proof
            (DPMarginals (Marginals $pump $valve $motor)) $_tv))
      (and
         (and (> $pump 0.7772) (< $pump 0.7773))
         (and
            (and (> $valve 0.2227) (< $valve 0.2228))
            (and (> $motor 0.2227) (< $motor 0.2228)))))
   true)

!(let (: $_proof (FoldMarginals $marginals) $_tv)
   (query 200 foldKb (: $_proof (FoldMarginals $marginals) $_tv))
   (ExactResult FoldAll $marginals))

!(let (: $_proof (DPMarginals $marginals) $_tv)
   (query 200 dpKb (: $_proof (DPMarginals $marginals) $_tv))
   (ExactResult DP $marginals))

;; Compact fresh-query budget probes.  Every query starts with an empty
;; backward proof store, so these show how much best-first search each shape
;; currently needs before its marginal vector appears.
(: has-answer? (-[det]-> (List $value) Bool))
(= (has-answer? ()) false)
(= (has-answer? (cons $_head $_tail)) true)

!(let $answers
   (collapse (query 10 foldKb (: $p (FoldMarginals $x) $tv)))
   (BudgetProbe FoldAll 10 (has-answer? $answers)))
!(let $answers
   (collapse (query 20 foldKb (: $p (FoldMarginals $x) $tv)))
   (BudgetProbe FoldAll 20 (has-answer? $answers)))
!(let $answers
   (collapse (query 30 foldKb (: $p (FoldMarginals $x) $tv)))
   (BudgetProbe FoldAll 30 (has-answer? $answers)))
!(let $answers
   (collapse (query 40 foldKb (: $p (FoldMarginals $x) $tv)))
   (BudgetProbe FoldAll 40 (has-answer? $answers)))
!(let $answers
   (collapse (query 80 foldKb (: $p (FoldMarginals $x) $tv)))
   (BudgetProbe FoldAll 80 (has-answer? $answers)))

!(let $answers
   (collapse (query 10 dpKb (: $p (DPMarginals $x) $tv)))
   (BudgetProbe DP 10 (has-answer? $answers)))
!(let $answers
   (collapse (query 20 dpKb (: $p (DPMarginals $x) $tv)))
   (BudgetProbe DP 20 (has-answer? $answers)))
!(let $answers
   (collapse (query 40 dpKb (: $p (DPMarginals $x) $tv)))
   (BudgetProbe DP 40 (has-answer? $answers)))
!(let $answers
   (collapse (query 50 dpKb (: $p (DPMarginals $x) $tv)))
   (BudgetProbe DP 50 (has-answer? $answers)))
!(let $answers
   (collapse (query 60 dpKb (: $p (DPMarginals $x) $tv)))
   (BudgetProbe DP 60 (has-answer? $answers)))
!(let $answers
   (collapse (query 80 dpKb (: $p (DPMarginals $x) $tv)))
   (BudgetProbe DP 80 (has-answer? $answers)))

;; Approximate/top-k exploration boundary:
;;
;; A small query budget can return only some ExactExplanation proofs, but those
;; partial answers are not yet a probability approximation: the public chainer
;; ranks proof work by PLN search priority, not by the numeric `$weight` stored
;; inside ExactExplanation, and the missing probability mass is unknown.
;; A real top-k mode should make branch log-probability part of agenda priority
;; and return (found-mass, unexplored-mass-bound, complete?).  A particle mode
;; should sample whole include/exclude paths conditioned on total loss 5, never
;; sample the three marginals independently.
;;
;; On the current runtime the 14-step probe finds valve-motor first even though
;; it has less probability mass than pump-only: its proof happens to contain one
;; fewer Compute premise.  That is useful best-first search, but not weighted
;; top-k inference.  Also, the existing ParticleFromPairs resampling compresses
;; a distribution after its pairs exist; it does not avoid generating exact
;; explanations upstream.  A cheap approximation needs sampling during search.

!(let $answers
   (collapse
      (query 10 foldKb
         (: $proof (ExactExplanation $selection $weight) $tv)))
   (ExplanationBudgetProbe 10 $answers))
!(let $answers
   (collapse
      (query 14 foldKb
         (: $proof (ExactExplanation $selection $weight) $tv)))
   (ExplanationBudgetProbe 14 $answers))
!(let $answers
   (collapse
      (query 16 foldKb
         (: $proof (ExactExplanation $selection $weight) $tv)))
   (ExplanationBudgetProbe 16 $answers))
!(let $answers
   (collapse
      (query 18 foldKb
         (: $proof (ExactExplanation $selection $weight) $tv)))
   (ExplanationBudgetProbe 18 $answers))
!(let $answers
   (collapse
      (query 20 foldKb
         (: $proof (ExactExplanation $selection $weight) $tv)))
   (ExplanationBudgetProbe 20 $answers))
