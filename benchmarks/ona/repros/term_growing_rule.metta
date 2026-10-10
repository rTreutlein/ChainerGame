;; ONA reasoningbench: a term-growing rule. Cost grows steeply with the budget:
;; budget 20 ~1 s, 40 ~5 s, 60 ~15 s, 100 > 55 s (after ~5 s load).
!(import! &self /nexus/Dev/OpenCog/NL2PLN_Project/PeTTaChainer/pettachainer/metta/petta_chainer)
!(compileadd kb (: r9 (Implication (Num $x) (Num (succ $x))) (STV 1.0 0.9)))
!(compileadd kb (: f10 (Num 1) (STV 1.0 0.9)))
!(query 100 kb (: $p (Num (succ 1)) $tv))
