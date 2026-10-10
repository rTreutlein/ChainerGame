;; ONA examples/nal/greenplant.nal (first question; the difference-term question (plant ~ ocean) is NAL-specific and disabled in ONA too)
!(import! &self /nexus/Dev/OpenCog/NL2PLN_Project/PeTTaChainer/pettachainer/metta/petta_chainer)
!(compileadd kb (: f1 (Member inst1 Plant) (STV 1.0 0.9)))
!(compileadd kb (: f2 (Green inst1) (STV 0.7 0.9)))
!(compileadd kb (: f3 (Member inst2 Plant) (STV 1.0 0.9)))
!(compileadd kb (: f4 (Green inst2) (STV 0.8 0.9)))
;; expected (NAL): <plant --> [green]> f=0.75 c>=0.618 ; in PLN: how green are plants = the implication Plant -> Green
!(query 100 kb (: $p (Implication (Member $x Plant) (Green $x)) $tv))
