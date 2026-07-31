# ChainerGame task environment

The managed image pins SWI-Prolog 9.3.33 by digest and installs PeTTaChainer
from `rTreutlein/PeTTaChainer` at commit
`d41c7224ea80695f90c4ca1ffecc5d3a188f3c61`. PeTTaChainer's frozen lock installs
PeTTa at commit `e1bd9e3fff7ee5caa176bf14a950238b7caf477d` and `janus-swi==1.5.2`.

The default managed command remains the complete StationOps unittest suite. It
covers the reference backend and controlled adapters without requiring live
backend behavior to pass. Run it with:

```sh
python /app/project_env.py --task TASK_ID
```

Live PeTTaChainer validation is explicit and fails if conformance skips. It
records runtime identities, constructs the handler, runs focused v1 conformance,
and executes v0/v1 reference and v1 PeTTaChainer CLI smokes in the same image:

```sh
python /app/project_env.py --task TASK_ID -- sh .myclaw/test-live-pettachainer.sh
```

MM2 remains external and is not installed by this image.
