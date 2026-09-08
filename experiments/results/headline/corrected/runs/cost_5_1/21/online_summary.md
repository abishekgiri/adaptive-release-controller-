# Online Replay Result: cost_5_1

> **Online replay costs are simulation artefacts. CI outcome used as counterfactual proxy. Do not report as causal real-world cost estimates.**

Seed: `21`  |  Trajectories: `2`

| Policy | Steps | Updates | Censored | Cumul. Cost | Mean Cost/Step | Deploy% | Canary% | Block% |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cost_rule | 1150 | 1150 | 0 | 1416.0000 | 1.2313 | 51.6% | 39.7% | 8.8% |
| bayesian_rate | 1150 | 1150 | 0 | 1235.0000 | 1.0739 | 51.3% | 48.4% | 0.3% |
| always_deploy | 1150 | 1150 | 0 | 1450.0000 | 1.2609 | 100.0% | 0.0% | 0.0% |
| always_canary | 1150 | 1150 | 0 | 1440.0000 | 1.2522 | 0.0% | 100.0% | 0.0% |
| always_block | 1150 | 1150 | 0 | 2010.0000 | 1.7478 | 0.0% | 0.0% | 100.0% |
| linucb_bias_only | 1150 | 1150 | 0 | 1257.0000 | 1.0930 | 49.3% | 49.3% | 1.4% |
| static_rules | 1150 | 1150 | 0 | 1599.0000 | 1.3904 | 2.6% | 61.1% | 36.3% |
| heuristic_score | 1150 | 1150 | 0 | 1430.0000 | 1.2435 | 37.4% | 62.6% | 0.0% |
| linucb | 1150 | 1150 | 0 | 1466.0000 | 1.2748 | 7.7% | 87.0% | 5.4% |
| linucb_with_drift | 1150 | 1150 | 0 | 1466.0000 | 1.2748 | 7.7% | 87.0% | 5.4% |
| thompson | 1150 | 1150 | 0 | 1450.0000 | 1.2609 | 2.6% | 93.9% | 3.5% |
