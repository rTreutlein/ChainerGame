;; PeTTaChainer regression repro: a directly queryable OR must remain
;; recoverable through its definitional reverse direction after the equivalent
;; Problem has both prior and sensor evidence.
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

;; Problem is definitionally equivalent to the OR of its local and dependency
;; causes.  BiImplication lowers this declaration into two explicit one-way
;; directions; those directions must not themselves be inverted again.
!(compileadd orInverseReproKb
   (: problemDefinition
      (BiImplication
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

;; Checkpoint 3: this includes a proof that follows Problem backwards through
;; problemDefinition, projects child 0 of the reconstructed OR, and uses
;; dependencyPrior as the other-child evidence.  The forward Problem proof is
;; cyclic relative to this query and is removed before equivalent Problem
;; outputs are merged; the independent alarm proof remains available.
!(println! "CHECK 3: alarm-conditioned local proof")
!(query 20000 orInverseReproKb
   (: $proof (LocalProblem current) $tv))
