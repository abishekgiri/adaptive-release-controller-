# Online Replay Result: robustness_low_block

> **Online replay costs are simulation artefacts. CI outcome used as counterfactual proxy. Do not report as causal real-world cost estimates.**

Seed: `12`  |  Trajectories: `2`

| Policy | Steps | Updates | Censored | Cumul. Cost | Mean Cost/Step | Deploy% | Canary% | Block% |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cost_rule | 1150 | 1150 | 0 | 1273.5000 | 1.1074 | 19.6% | 0.0% | 80.4% |
| bayesian_rate | 1150 | 1150 | 0 | 1005.0000 | 0.8739 | 0.0% | 0.0% | 100.0% |
| always_deploy | 1150 | 1150 | 0 | 2900.0000 | 2.5217 | 100.0% | 0.0% | 0.0% |
| always_canary | 1150 | 1150 | 0 | 2020.0000 | 1.7565 | 0.0% | 100.0% | 0.0% |
| always_block | 1150 | 1150 | 0 | 1005.0000 | 0.8739 | 0.0% | 0.0% | 100.0% |
| linucb_bias_only | 1150 | 1150 | 0 | 1024.0000 | 0.8904 | 0.8% | 0.2% | 99.0% |
| static_rules | 1150 | 1150 | 0 | 1591.0000 | 1.3835 | 2.6% | 61.1% | 36.3% |
| heuristic_score | 1150 | 1150 | 0 | 2325.0000 | 2.0217 | 37.4% | 62.6% | 0.0% |
| linucb | 1150 | 1150 | 0 | 1072.0000 | 0.9322 | 1.5% | 2.1% | 96.4% |
| linucb_with_drift | 1150 | 1150 | 0 | 1072.0000 | 0.9322 | 1.5% | 2.1% | 96.4% |
| thompson | 1150 | 1150 | 0 | 1045.5000 | 0.9091 | 2.3% | 1.7% | 96.1% |
