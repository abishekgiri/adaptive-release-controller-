# Online Replay Result: robustness_long_delay

> **Online replay costs are simulation artefacts. CI outcome used as counterfactual proxy. Do not report as causal real-world cost estimates.**

Seed: `18`  |  Trajectories: `2`

| Policy | Steps | Updates | Censored | Cumul. Cost | Mean Cost/Step | Deploy% | Canary% | Block% |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cost_rule | 1150 | 1150 | 0 | 2077.0000 | 1.8061 | 33.1% | 20.0% | 46.9% |
| bayesian_rate | 1150 | 1150 | 0 | 1713.0000 | 1.4896 | 0.0% | 46.9% | 53.1% |
| always_deploy | 1150 | 1150 | 0 | 2900.0000 | 2.5217 | 100.0% | 0.0% | 0.0% |
| always_canary | 1150 | 1150 | 0 | 2020.0000 | 1.7565 | 0.0% | 100.0% | 0.0% |
| always_block | 1150 | 1150 | 0 | 1865.0000 | 1.6217 | 0.0% | 0.0% | 100.0% |
| linucb_bias_only | 1150 | 1150 | 0 | 1777.5000 | 1.5457 | 1.5% | 52.3% | 46.2% |
| static_rules | 1150 | 1150 | 0 | 1924.0000 | 1.6730 | 2.4% | 63.6% | 34.0% |
| heuristic_score | 1150 | 1150 | 0 | 2319.0000 | 2.0165 | 38.0% | 61.9% | 0.1% |
| linucb | 1150 | 1150 | 0 | 1973.0000 | 1.7157 | 11.5% | 20.4% | 68.1% |
| linucb_with_drift | 1150 | 1150 | 0 | 1973.0000 | 1.7157 | 11.5% | 20.4% | 68.1% |
| thompson | 1150 | 1150 | 0 | 1894.0000 | 1.6470 | 16.8% | 28.2% | 55.0% |
