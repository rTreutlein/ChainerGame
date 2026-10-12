!(import! &self ../../../../NL2PLN_Project/PeTTaChainer/pettachainer/metta/petta_chainer)

;; Gap 1 (docs/supply_network.md, "Stage 4 as built"): an Assuming context
;; reads no complete statement as false. S and P are complete; the main
;; context answers (P t) from F alone, the context that does not assume S
;; answers nothing (with assuming_complete.patch: (STV 0.3 0.9)).
!(set-complete-predicate actKb S)
!(set-complete-predicate actKb P)
!(compileadd actKb (: rp (Implication (Or (S $t) (F $t)) (P $t)) (CTV (STV 1 1) (STV 0 1))))
!(compileadd actKb (: f (F t) (STV 0.3 0.9)))
!(query 100 actKb (: $p (P t) $tv))
!(query 100 actKb (: $p (Assuming (Decide t) (P t)) $tv))
!(query 100 actKb (: $p (Assuming (S t) (P t)) $tv))
