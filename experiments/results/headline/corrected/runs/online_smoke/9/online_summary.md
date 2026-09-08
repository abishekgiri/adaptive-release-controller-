# Online Replay Result: online_smoke

> **Online replay costs are simulation artefacts. CI outcome used as counterfactual proxy. Do not report as causal real-world cost estimates.**

Seed: `9`  |  Trajectories: `2`

| Policy | Steps | Updates | Censored | Cumul. Cost | Mean Cost/Step | Deploy% | Canary% | Block% |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cost_rule | 1150 | 1150 | 0 | 1970.5000 | 1.7135 | 20.1% | 31.3% | 48.6% |
| bayesian_rate | 1150 | 1150 | 0 | 1713.5000 | 1.4900 | 1.9% | 49.0% | 49.1% |
| always_deploy | 1150 | 1150 | 0 | 2900.0000 | 2.5217 | 100.0% | 0.0% | 0.0% |
| always_canary | 1150 | 1150 | 0 | 2020.0000 | 1.7565 | 0.0% | 100.0% | 0.0% |
| always_block | 1150 | 1150 | 0 | 1865.0000 | 1.6217 | 0.0% | 0.0% | 100.0% |
| linucb_bias_only | 1150 | 1150 | 0 | 1739.0000 | 1.5122 | 3.0% | 46.9% | 50.2% |
| static_rules | 1150 | 1150 | 0 | 1884.0000 | 1.6383 | 2.6% | 61.1% | 36.3% |
| heuristic_score | 1150 | 1150 | 0 | 2325.0000 | 2.0217 | 37.4% | 62.6% | 0.0% |
| linucb | 1150 | 1150 | 0 | 1863.0000 | 1.6200 | 1.7% | 26.2% | 72.2% |
| linucb_with_drift | 1150 | 1150 | 0 | 1863.0000 | 1.6200 | 1.7% | 26.2% | 72.2% |
| thompson | 1150 | 1150 | 0 | 1904.0000 | 1.6557 | 2.3% | 4.6% | 93.0% |
