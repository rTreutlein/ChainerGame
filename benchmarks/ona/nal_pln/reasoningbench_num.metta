;; ONA examples/nal/reasoningbench.nal, the succ part: a term-growing rule.
!(import! &self /nexus/Dev/OpenCog/NL2PLN_Project/PeTTaChainer/pettachainer/metta/petta_chainer)
!(compileadd kb (: r9 (Implication (Num $x) (Num (succ $x))) (STV 1.0 0.9)))
!(compileadd kb (: f10 (Num 1) (STV 1.0 0.9)))
;; expected (NAL): num chain c>=.81/.729/.6561/.5905
!(query 100 kb (: $p (Num (succ 1)) $tv))
!(query 100 kb (: $p (Num (succ (succ 1))) $tv))
!(query 100 kb (: $p (Num (succ (succ (succ 1)))) $tv))
!(query 100 kb (: $p (Num (succ (succ (succ (succ 1))))) $tv))
