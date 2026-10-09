
### SupplyNet stage 1 (seeds 1-4, 30 rounds, budget 100)

| reasoner | seeds | error to exact | Brier | log loss | coverage | seconds per run |
|---|---|---|---|---|---|---|
| reference | 4 | 0.000 | 0.027 | 0.102 | 1.00 | 0.0 |
| prior | 4 | 0.239 | 0.134 | 0.435 | 1.00 | 0.0 |
| pettachainer | 4 | 0.008 | 0.028 | 0.106 | 1.00 | 5.9 |
| nars (nars-c0) | 4 | 0.242 | 0.139 | 0.455 | 0.97 | 1152.2 |
| nars (nars-c0-expectation) | 4 | 0.286 | 0.144 | 0.460 | 0.97 | 0.1 |
| nars (nars-c1) | 4 | 0.256 | 0.161 | 0.544 | 0.98 | 2815.4 |

Exact posterior's own Brier 0.027, log loss 0.102.

Error to exact by query kind (coverage in brackets):

| reasoner | blocked | storm |
|---|---|---|
| reference | 0.000 [1.00] | 0.000 [1.00] |
| prior | 0.233 [1.00] | 0.254 [1.00] |
| pettachainer | 0.002 [1.00] | 0.021 [1.00] |
| nars (nars-c0) | 0.234 [0.97] | 0.259 [0.99] |
| nars (nars-c0-expectation) | 0.283 [0.97] | 0.292 [0.99] |
| nars (nars-c1) | 0.255 [0.98] | 0.260 [0.99] |

### SupplyNet stage 2 (seeds 1-4, 30 rounds, budget 100)

| reasoner | seeds | error to exact | Brier | log loss | coverage | seconds per run |
|---|---|---|---|---|---|---|
| reference | 4 | 0.000 | 0.154 | 0.477 | 1.00 | 0.0 |
| prior | 4 | 0.183 | 0.213 | 0.615 | 1.00 | 0.0 |
| pettachainer | 4 | 0.000 | 0.154 | 0.477 | 1.00 | 12.1 |
| nars (nars-c0) | 4 | 0.196 | 0.223 | 0.699 | 0.94 | 1318.5 |
| nars (nars-c0-expectation) | 4 | 0.207 | 0.220 | 0.630 | 0.94 | 0.1 |
| nars (nars-c1) | 4 | 0.238 | 0.245 | 0.714 | 0.66 | 2836.0 |

Exact posterior's own Brier 0.154, log loss 0.477.

Error to exact by query kind (coverage in brackets):

| reasoner | blocked | previous_storm | storm |
|---|---|---|---|
| reference | 0.000 [1.00] | 0.000 [1.00] | 0.000 [1.00] |
| prior | 0.120 [1.00] | 0.277 [1.00] | 0.225 [1.00] |
| pettachainer | 0.000 [1.00] | 0.000 [1.00] | 0.000 [1.00] |
| nars (nars-c0) | 0.145 [0.92] | 0.273 [0.98] | 0.231 [0.94] |
| nars (nars-c0-expectation) | 0.172 [0.92] | 0.271 [0.98] | 0.220 [0.94] |
| nars (nars-c1) | 0.218 [0.54] | 0.281 [0.89] | 0.237 [0.67] |

### SupplyNet stage 3 timed (seeds 1-4, 30 rounds, budget 100)

| reasoner | seeds | error to exact | Brier | log loss | coverage | seconds per run |
|---|---|---|---|---|---|---|
| reference | 4 | 0.000 | 0.140 | 0.429 | 1.00 | 0.7 |
| prior | 4 | 0.229 | 0.226 | 0.649 | 1.00 | 0.0 |
| pettachainer | 4 | 0.003 | 0.140 | 0.431 | 1.00 | 16.9 |
| nars (nars-c0) | 4 | 0.214 | 0.227 | 0.688 | 0.98 | 2554.3 |
| nars (nars-c0-expectation) | 4 | 0.218 | 0.218 | 0.623 | 0.98 | 0.1 |

Exact posterior's own Brier 0.140, log loss 0.429.

Error to exact by query kind (coverage in brackets):

| reasoner | blocked | degraded | previous_storm | producing | producing_unstocked | storm |
|---|---|---|---|---|---|---|
| reference | 0.000 [1.00] | 0.000 [1.00] | 0.000 [1.00] | 0.000 [1.00] | 0.000 [1.00] | 0.000 [1.00] |
| prior | 0.131 [1.00] | 0.235 [1.00] | 0.289 [1.00] | 0.399 [1.00] | 0.316 [1.00] | 0.236 [1.00] |
| pettachainer | 0.001 [1.00] | 0.007 [1.00] | 0.002 [1.00] | 0.011 [1.00] | 0.006 [1.00] | 0.001 [1.00] |
| nars (nars-c0) | 0.133 [1.00] | 0.213 [1.00] | 0.286 [0.88] | 0.352 [1.00] | 0.258 [0.99] | 0.231 [0.98] |
| nars (nars-c0-expectation) | 0.145 [1.00] | 0.219 [1.00] | 0.282 [0.88] | 0.327 [1.00] | 0.276 [0.99] | 0.223 [0.98] |

### SupplyNet stage 3 untimed (seeds 1-4, 30 rounds, budget 100)

| reasoner | seeds | error to exact | Brier | log loss | coverage | seconds per run |
|---|---|---|---|---|---|---|
| reference | 4 | 0.000 | 0.135 | 0.414 | 1.00 | 0.7 |
| prior | 4 | 0.214 | 0.212 | 0.616 | 1.00 | 0.0 |
| pettachainer | 4 | 0.002 | 0.136 | 0.416 | 1.00 | 17.5 |
| nars (nars-c0) | 4 | 0.206 | 0.217 | 0.698 | 0.98 | 2284.0 |
| nars (nars-c0-expectation) | 4 | 0.212 | 0.208 | 0.602 | 0.98 | 0.1 |

Exact posterior's own Brier 0.135, log loss 0.414.

Error to exact by query kind (coverage in brackets):

| reasoner | blocked | degraded | previous_storm | producing | producing_unstocked | storm |
|---|---|---|---|---|---|---|
| reference | 0.000 [1.00] | 0.000 [1.00] | 0.000 [1.00] | 0.000 [1.00] | 0.000 [1.00] | 0.000 [1.00] |
| prior | 0.131 [1.00] | 0.225 [1.00] | 0.288 [1.00] | 0.420 [1.00] | 0.212 [1.00] | 0.236 [1.00] |
| pettachainer | 0.001 [1.00] | 0.004 [1.00] | 0.000 [1.00] | 0.008 [1.00] | 0.004 [1.00] | 0.000 [1.00] |
| nars (nars-c0) | 0.138 [1.00] | 0.208 [1.00] | 0.298 [0.92] | 0.395 [1.00] | 0.156 [1.00] | 0.232 [0.98] |
| nars (nars-c0-expectation) | 0.147 [1.00] | 0.215 [1.00] | 0.281 [0.92] | 0.357 [1.00] | 0.216 [1.00] | 0.222 [0.98] |

### StationOps temporal replay (seeds 1-4, 20 shifts, budget 50)

| reasoner | seeds | error to oracle | Brier | log loss | coverage | reasoner-only regret | seconds per run |
|---|---|---|---|---|---|---|---|
| reference | 4 | 0.000 | 0.082 | 0.760 | 1.00 | 327.5 | 0.0 |
| pettachainer | 4 | 0.117 | 0.061 | 0.211 | 1.00 | 366.1 | 16.2 |
| nars | 4 | 0.229 | 0.233 | 1.195 | 0.95 | 747.4 | 830.1 |
