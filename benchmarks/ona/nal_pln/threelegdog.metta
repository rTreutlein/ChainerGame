;; ONA examples/nal/threelegdog.nal
!(import! &self /nexus/Dev/OpenCog/NL2PLN_Project/PeTTaChainer/pettachainer/metta/petta_chainer)
!(compileadd kb (: f1 (Member fido Dog) (STV 1.0 0.9)))
!(compileadd kb (: f2 (Member snoopy Dog) (STV 1.0 0.9)))
!(compileadd kb (: f3 (Inheritance Dog Animal) (STV 1.0 0.9)))
!(compileadd kb (: f4 (HasLeg Dog 4) (STV 1.0 0.9)))
!(compileadd kb (: f5 (HasLeg fido 3) (STV 1.0 0.9)))
!(compileadd kb (: f6 (HasLeg Chicken 2) (STV 1.0 0.9)))
!(compileadd kb (: f7 (Inheritance Chicken Animal) (STV 1.0 0.9)))
;; NAL's (dog * 4) --> has_leg on the class term; the member view of a class property:
!(compileadd kb (: r1 (Implication (And (Member $x $c) (HasLeg $c $n)) (HasLeg $x $n)) (STV 1.0 0.9)))
;; expected (NAL): fido 3 (.9); snoopy 4 (.81); dog 4 (.9); (animal 2) f=1 c>=.4475; (animal 4) f=1 c>=.4475
!(query 100 kb (: $p (HasLeg fido $n) $tv))
!(query 100 kb (: $p (HasLeg snoopy $n) $tv))
!(query 100 kb (: $p (HasLeg Dog $n) $tv))
!(query 100 kb (: $p (HasLeg Animal 2) $tv))
!(query 100 kb (: $p (HasLeg Animal 4) $tv))
