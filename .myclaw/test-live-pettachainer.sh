#!/bin/sh
set -eu

PYTHON=/opt/pettachainer/.venv/bin/python
export STATIONOPS_LIVE_PETTACHAINER=1

/usr/local/bin/uv pip install --no-cache --python "$PYTHON" --no-deps --no-build-isolation -e .

"$PYTHON" - <<'PY'
import importlib.metadata
import json
import pathlib
import subprocess
import sys

import janus_swi
from pettachainer import PeTTaChainer

handler = PeTTaChainer()
distribution = importlib.metadata.distribution("petta")
direct_url = json.loads(distribution.read_text("direct_url.json"))
petta_revision = direct_url["vcs_info"]["commit_id"]
pettachainer_revision = subprocess.check_output(
    ["git", "-C", "/opt/pettachainer", "rev-parse", "HEAD"], text=True
).strip()
print(f"python={sys.version.split()[0]}")
print(subprocess.check_output(["swipl", "--version"], text=True).strip())
print(f"janus_swi={importlib.metadata.version('janus-swi')}")
print(f"pettachainer={importlib.metadata.version('PeTTaChainer')}@{pettachainer_revision}")
print(f"petta={distribution.version}@{petta_revision}")
assert pathlib.Path(janus_swi.__file__).is_file()
assert handler is not None
assert pettachainer_revision == "d41c7224ea80695f90c4ca1ffecc5d3a188f3c61"
assert petta_revision == "e1bd9e3fff7ee5caa176bf14a950238b7caf477d"
PY

"$PYTHON" - <<'PY'
import unittest

suite = unittest.defaultTestLoader.loadTestsFromName(
    "tests.test_stationops.V1BenchmarkTests.test_live_pettachainer_v1_conformance_when_available"
)
result = unittest.TextTestRunner(verbosity=2).run(suite)
if result.skipped:
    raise SystemExit(f"live conformance skipped: {result.skipped}")
if not result.wasSuccessful() or result.testsRun != 1:
    raise SystemExit("live conformance failed")
PY

"$PYTHON" -m stationops.cli run --backend reference --budget 100 >/tmp/stationops-v0-reference.json
"$PYTHON" -m stationops.cli run --benchmark v1 --backend reference --budget 100 >/tmp/stationops-v1-reference.json
"$PYTHON" -m stationops.cli run --benchmark v1 --backend pettachainer --budget 200 >/tmp/stationops-v1-pettachainer.json

"$PYTHON" - <<'PY'
import json

checks = (
    ("/tmp/stationops-v0-reference.json", "v0", "reference"),
    ("/tmp/stationops-v1-reference.json", "v1", "reference"),
    ("/tmp/stationops-v1-pettachainer.json", "v1", "pettachainer"),
)
for path, benchmark, backend in checks:
    with open(path, encoding="utf-8") as stream:
        result = json.load(stream)
    assert result["backend"] == backend
    if benchmark == "v1":
        assert result["benchmark"] == "BaseRateTriage-v1"
    print(f"cli-smoke={benchmark}/{backend}:ok")
PY
