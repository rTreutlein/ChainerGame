;; ONA examples/nal/penguin.nal. NAL input confidence 0.9 everywhere.
!(import! &self /nexus/Dev/OpenCog/NL2PLN_Project/PeTTaChainer/pettachainer/metta/petta_chainer)
!(compileadd kb (: f1 (Member pingu Penguin) (STV 1.0 0.9)))
!(compileadd kb (: f2 (Member sam Raven) (STV 1.0 0.9)))
!(compileadd kb (: r1 (Implication (Member $x Penguin) (Fly $x)) (STV 0.0 0.9)))
!(compileadd kb (: f3 (Inheritance Raven Bird) (STV 1.0 0.9)))
!(compileadd kb (: f4 (Inheritance Penguin Bird) (STV 1.0 0.9)))
!(compileadd kb (: r2 (Implication (Member $x Bird) (Fly $x)) (STV 1.0 0.9)))
;; expected (NAL): sam fly f=1.0 c>=0.729; pingu fly f=0.0 c>=0.81
!(query 100 kb (: $p (Fly sam) $tv))
!(query 100 kb (: $p (Fly pingu) $tv))
