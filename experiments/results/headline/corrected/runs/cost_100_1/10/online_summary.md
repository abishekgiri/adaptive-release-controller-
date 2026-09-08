# Online Replay Result: cost_100_1

> **Online replay costs are simulation artefacts. CI outcome used as counterfactual proxy. Do not report as causal real-world cost estimates.**

Seed: `10`  |  Trajectories: `2`

| Policy | Steps | Updates | Censored | Cumul. Cost | Mean Cost/Step | Deploy% | Canary% | Block% |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cost_rule | 1150 | 1150 | 0 | 3835.5000 | 3.3352 | 19.6% | 0.0% | 80.4% |
| bayesian_rate | 1150 | 1150 | 0 | 1865.0000 | 1.6217 | 0.0% | 0.0% | 100.0% |
| always_deploy | 1150 | 1150 | 0 | 14500.0000 | 12.6087 | 100.0% | 0.0% | 0.0% |
| always_canary | 1150 | 1150 | 0 | 6660.0000 | 5.7913 | 0.0% | 100.0% | 0.0% |
| always_block | 1150 | 1150 | 0 | 1865.0000 | 1.6217 | 0.0% | 0.0% | 100.0% |
| linucb_bias_only | 1150 | 1150 | 0 | 1989.0000 | 1.7296 | 0.8% | 0.2% | 99.0% |
| static_rules | 1150 | 1150 | 0 | 4660.0000 | 4.0522 | 2.6% | 61.1% | 36.3% |
| heuristic_score | 1150 | 1150 | 0 | 9485.0000 | 8.2478 | 37.4% | 62.6% | 0.0% |
| linucb | 1150 | 1150 | 0 | 2097.5000 | 1.8239 | 1.3% | 0.4% | 98.3% |
| linucb_with_drift | 1150 | 1150 | 0 | 2097.5000 | 1.8239 | 1.3% | 0.4% | 98.3% |
| thompson | 1150 | 1150 | 0 | 2075.5000 | 1.8048 | 0.3% | 1.7% | 98.0% |
