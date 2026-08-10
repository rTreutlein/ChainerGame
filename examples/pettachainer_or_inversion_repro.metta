;; Minimal PeTTaChainer repro: a directly queryable OR is not recovered through
;; implication inversion after its conclusion has both a prior and sensor
;; evidence.
;;
;; Run from this checkout with:
;;   petta examples/pettachainer_or_inversion_repro.metta
;;
;; The absolute import keeps this file immediately runnable in the current
;; sibling-checkout layout.  When copying it into pettachainer/metta/examples,
;; replace the import path with: ../petta_chainer

!(import! &self
   /nexus/Dev/OpenCog/NL2PLN_Project/PeTTaChainer/pettachainer/metta/petta_chainer)

;; A and B are prior beliefs about the same current machine.  Their direct OR
;; is 0.4 OR 0.2 = 0.52.
!(compileadd orInverseReproKb
   (: localPrior
      (LocalProblem current)
      (STV 0.4 0.9)))
!(compileadd orInverseReproKb
   (: dependencyPrior
      (ProblemDependency current)
      (STV 0.2 0.9)))

;; Problem is definitionally the OR of its local and dependency causes.
!(compileadd orInverseReproKb
   (: problemDefinition
      (Implication
         (Or
            (LocalProblem current)
            (ProblemDependency current))
         (Problem current))
      (CTV (STV 1.0 1.0) (STV 0.0 1.0))))

;; A noisy sensor provides independent current evidence about Problem.
!(compileadd orInverseReproKb
   (: problemRaisesAlarm
      (Implication
         (Problem $unit)
         (Alarm $unit))
      (CTV (STV 0.9 1.0) (STV 0.1 1.0))))
!(compileadd orInverseReproKb
   (: currentAlarm
      (Alarm current)
      (STV 1.0 1.0)))

;; Two resolved historical cases supply the base rates used by sensor
;; inversion.  No set-base-rate shortcut is involved: the inverse rule must
;; obtain these through its compiled FoldAll queries.
!(compileadd orInverseReproKb
   (: h1Problem (Problem h1) (STV 1.0 1.0)))
!(compileadd orInverseReproKb
   (: h1Alarm (Alarm h1) (STV 1.0 1.0)))
!(compileadd orInverseReproKb
   (: h2Problem (Problem h2) (STV 0.0 1.0)))
!(compileadd orInverseReproKb
   (: h2Alarm (Alarm h2) (STV 0.0 1.0)))

;; Checkpoint 1: OR introduction works when the compound is queried directly.
;; Expected: a conjunction of localPrior and dependencyPrior, strength 0.52.
!(println! "CHECK 1: direct OR query")
!(query 2000 orInverseReproKb
   (: $proof
      (Or
         (LocalProblem current)
         (ProblemDependency current))
      $tv))

;; Checkpoint 2: currentAlarm is inverted through problemRaisesAlarm and
;; combined with the forward prior from problemDefinition.
!(println! "CHECK 2: Problem receives sensor evidence")
!(query 5000 orInverseReproKb
   (: $proof (Problem current) $tv))

;; Checkpoint 3: this should include a proof that follows Problem backwards
;; through problemDefinition, projects child 0 of the reconstructed OR, and
;; uses dependencyPrior as the other-child evidence.  At PeTTaChainer e3961f7,
;; only localPrior is returned; no alarm-conditioned proof appears.
!(println! "CHECK 3: missing alarm-conditioned local proof")
!(query 20000 orInverseReproKb
   (: $proof (LocalProblem current) $tv))
