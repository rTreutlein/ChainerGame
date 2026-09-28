; Resolved states are individuals, members of their group's concepts.  Both
; positive and complemented STVs are labels; absence remains unknown.  Current
; unresolved states must not be inserted here because they do not yet have a
; SealLeak label.
(: s1-leak (Member s1 (SealLeak old coolant-pump)) (STV 1 1))
(: s1-alarm (Member s1 (PressureAlarm old coolant-pump)) (STV 1 1))
(: s2-leak (Member s2 (SealLeak old coolant-pump)) (STV 0 1))
(: s2-alarm (Member s2 (PressureAlarm old coolant-pump)) (STV 0 1))
(: s2-normal (Member s2 (PressureNormal old coolant-pump)) (STV 1 1))
(: s3-leak (Member s3 (SealLeak old coolant-pump)) (STV 0 1))
(: s3-alarm (Member s3 (PressureAlarm old coolant-pump)) (STV 1 1))
