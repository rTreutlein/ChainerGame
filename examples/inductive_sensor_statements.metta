; Resolved states are common subjects.  Both positive and complemented STVs
; are labels; absence remains unknown.  Current unresolved states must not be
; inserted here because they do not yet have a SealLeak label.
(: s1-leak (Inheritance (State s1) (SealLeak old coolant-pump)) (STV 1 1))
(: s1-alarm (Inheritance (State s1) (PressureAlarm old coolant-pump)) (STV 1 1))
(: s2-leak (Inheritance (State s2) (SealLeak old coolant-pump)) (STV 0 1))
(: s2-alarm (Inheritance (State s2) (PressureAlarm old coolant-pump)) (STV 0 1))
(: s2-normal (Inheritance (State s2) (PressureNormal old coolant-pump)) (STV 1 1))
(: s3-leak (Inheritance (State s3) (SealLeak old coolant-pump)) (STV 0 1))
(: s3-alarm (Inheritance (State s3) (PressureAlarm old coolant-pump)) (STV 1 1))
