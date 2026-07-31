# ChainerGame task environment

This environment installs the local `stationops` package and runs the complete
dependency-independent test suite. It validates ChainerGame itself, including
the reference backend and controlled adapter tests.

Live MM2-Chainer and PeTTaChainer conformance are optional downstream checks.
They are skipped with actionable messages when their external Python bindings
are unavailable; neither integration is installed or validated by this image.
