"""Score ONA shell output the way ONA's evaluation.py does, without stopping at
the first failure.

evaluation.py's rule, per `//expected:` comment line (lines starting with
`//--expected` are disabled tests and do not count):
- `Answer: X ... Truth: ... confidence=c`: some `Answer:` line since the
  previous expectation contains X and has confidence >= c.
- `Answer: X` without a truth: an answer containing X exists (answer-ratio test).
- `^op executed with args ...`: the last executed operation before the
  expectation equals that text.
"""
import re
import sys
from pathlib import Path

EXPECT = "Comment: expected: "


def score(text):
    lines = [l.strip() for l in text.split("\n")]
    results = []
    for i, line in enumerate(lines):
        if not line.startswith(EXPECT):
            continue
        body = line[len(EXPECT):]
        if body.startswith("Answer: "):
            exp = body[len("Answer: "):]
            conf = None
            if "Truth:" in exp:
                conf = float(exp.split("confidence=")[1])
                exp = exp.split("Truth:")[0]
            exp = exp.strip() + " "
            best = None
            for j in reversed(range(i)):
                lb = lines[j]
                if lb.startswith(EXPECT):
                    break
                if lb.startswith("Answer:") and exp in lb:
                    c = float(lb.split("confidence=")[1])
                    best = c if best is None else max(best, c)
            ok = best is not None and (conf is None or best >= conf)
            results.append(("answer", body, ok, best, conf))
        elif body.startswith("^"):
            ok = False
            got = None
            for j in reversed(range(i)):
                lb = lines[j]
                if lb.startswith("^"):
                    got = lb
                    ok = lb == body
                    break
            results.append(("exec", body, ok, got, None))
    return results


def main(argv):
    total = passed = 0
    for path in argv[1:]:
        res = score(Path(path).read_text(errors="replace"))
        p = sum(1 for r in res if r[2])
        total += len(res)
        passed += p
        fails = [r[1] for r in res if not r[2]]
        print(f"{Path(path).stem}\t{p}/{len(res)}" + ("" if not fails else "\tFAIL: " + " | ".join(fails)))
    print(f"TOTAL\t{passed}/{total}")


if __name__ == "__main__":
    main(sys.argv)
