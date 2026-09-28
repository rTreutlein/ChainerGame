!(import! &self ../../NL2PLN_Project/PeTTaChainer/pettachainer/metta/petta_chainer)
!(import! &self ../src/stationops/metta/weighted_subset_posterior)

;; An observed loss of 5 has two explanations: pump, or valve + motor.
;; The compact DP merges worlds by reachable loss and projects posterior
;; probabilities without enumerating either fault set.
(: station-candidates (-[det]-> (List WeightedSubsetCandidate)))
(= (station-candidates)
   ((WeightedCandidate pump 5 0.02)
    (WeightedCandidate valve 3 0.10)
    (WeightedCandidate motor 2 0.05)))

(: repeated-unit-candidates (-[det]-> (List WeightedSubsetCandidate)))
(= (repeated-unit-candidates)
   ((WeightedCandidate c01 1 0.5)
    (WeightedCandidate c02 1 0.5)
    (WeightedCandidate c03 1 0.5)
    (WeightedCandidate c04 1 0.5)
    (WeightedCandidate c05 1 0.5)
    (WeightedCandidate c06 1 0.5)
    (WeightedCandidate c07 1 0.5)
    (WeightedCandidate c08 1 0.5)
    (WeightedCandidate c09 1 0.5)
    (WeightedCandidate c10 1 0.5)
    (WeightedCandidate c11 1 0.5)
    (WeightedCandidate c12 1 0.5)))

!(test
   (WeightedSubsetPosteriorMass
      (WeightedSubsetPosteriorDP (station-candidates) 5))
   0.022000000000000002)

!(test
   (WeightedSubsetPosteriorMarginal
      (WeightedSubsetPosteriorDP (station-candidates) 5)
      pump)
   0.7772727272727272)

!(test
   (WeightedSubsetPosteriorMarginal
      (WeightedSubsetPosteriorDP (station-candidates) 5)
      motor)
   0.22272727272727275)

!(test
   (WeightedSubsetPosteriorMass
      (WeightedSubsetPosteriorDP (station-candidates) 4))
   0.0)

;; M03 alone explains the loss; the unclamped quotient rounds to 1.0000000000000002.
!(test
   (WeightedSubsetPosteriorMarginal
      (WeightedSubsetPosteriorDP
         ((WeightedCandidate M02 45 1e-06)
          (WeightedCandidate M03 90 0.0009990000019960037)
          (WeightedCandidate M04 70 1e-06)
          (WeightedCandidate M07 40 0.1666666578276653)
          (WeightedCandidate M10 100 1e-06))
         90)
      M03)
   1.0)

;; Twelve binary choices collapse to the seven reachable loss cells 0..6,
;; rather than retaining the 2^12 underlying configurations.
!(test
   (let (WeightedSubsetDP $_target $_candidates $_prefix-rows
           (cons (WeightedDPRow $all-cells) $_postfix-rest))
      (WeightedSubsetPosteriorDP (repeated-unit-candidates) 6)
      (length $all-cells))
   7)

!(compileadd stationOpsKb
   (: candidates
      (CandidateModules (station-candidates))
      (STV 1.0 1.0)))

!(compileadd stationOpsKb
   (: observed
      (ObservedTotalLoss 5)
      (STV 1.0 1.0)))

!(compileadd stationOpsKb
   (: diagnose
      (Implication
         (And
            (CandidateModules $candidates)
            (ObservedTotalLoss $total)
            (Compute WeightedSubsetPosteriorDP
               ($candidates $total)
               ->
               $posterior))
         (FaultPosterior $posterior))
      (CTV (STV 1.0 1.0) (STV 0.0 1.0))))

!(compileadd stationOpsKb (: pumpModule (Module pump) (STV 1.0 1.0)))

!(compileadd stationOpsKb
   (: projectModuleProbability
      (Implication
         (And
            (FaultPosterior $posterior)
            (Module $module)
            (Compute WeightedSubsetPosteriorMarginal
               ($posterior $module)
               ->
               $probability))
         (FaultProbability $module $probability))
      (CTV (STV 1.0 1.0) (STV 0.0 1.0))))

!(test
   (let (: $_proof (FaultProbability pump $probability) $_tv)
      (query 40 stationOpsKb
         (: $_proof (FaultProbability pump $probability) $_tv))
      (let $number (the Number $probability)
         (and (> $number 0.7772) (< $number 0.7773))))
   true)
