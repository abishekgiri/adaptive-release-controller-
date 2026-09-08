# Online Replay Result: cost_40_1

> **Online replay costs are simulation artefacts. CI outcome used as counterfactual proxy. Do not report as causal real-world cost estimates.**

Seed: `4`  |  Trajectories: `2`

| Policy | Steps | Updates | Censored | Cumul. Cost | Mean Cost/Step | Deploy% | Canary% | Block% |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cost_rule | 1150 | 1150 | 0 | 2425.5000 | 2.1091 | 19.6% | 0.0% | 80.4% |
| bayesian_rate | 1150 | 1150 | 0 | 1865.0000 | 1.6217 | 0.0% | 0.0% | 100.0% |
| always_deploy | 1150 | 1150 | 0 | 5800.0000 | 5.0435 | 100.0% | 0.0% | 0.0% |
| always_canary | 1150 | 1150 | 0 | 3180.0000 | 2.7652 | 0.0% | 100.0% | 0.0% |
| always_block | 1150 | 1150 | 0 | 1865.0000 | 1.6217 | 0.0% | 0.0% | 100.0% |
| linucb_bias_only | 1150 | 1150 | 0 | 1905.0000 | 1.6565 | 0.8% | 0.2% | 99.0% |
| static_rules | 1150 | 1150 | 0 | 2578.0000 | 2.2417 | 2.6% | 61.1% | 36.3% |
| heuristic_score | 1150 | 1150 | 0 | 4115.0000 | 3.5783 | 37.4% | 62.6% | 0.0% |
| linucb | 1150 | 1150 | 0 | 1972.0000 | 1.7148 | 1.0% | 11.0% | 88.0% |
| linucb_with_drift | 1150 | 1150 | 0 | 1972.0000 | 1.7148 | 1.0% | 11.0% | 88.0% |
| thompson | 1150 | 1150 | 0 | 1965.5000 | 1.7091 | 1.1% | 1.7% | 97.2% |
