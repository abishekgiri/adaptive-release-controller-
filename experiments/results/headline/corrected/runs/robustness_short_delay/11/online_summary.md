# Online Replay Result: robustness_short_delay

> **Online replay costs are simulation artefacts. CI outcome used as counterfactual proxy. Do not report as causal real-world cost estimates.**

Seed: `11`  |  Trajectories: `2`

| Policy | Steps | Updates | Censored | Cumul. Cost | Mean Cost/Step | Deploy% | Canary% | Block% |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cost_rule | 1150 | 1150 | 0 | 1949.0000 | 1.6948 | 29.3% | 20.3% | 50.4% |
| bayesian_rate | 1150 | 1150 | 0 | 1714.5000 | 1.4909 | 1.5% | 48.3% | 50.2% |
| always_deploy | 1150 | 1150 | 0 | 2900.0000 | 2.5217 | 100.0% | 0.0% | 0.0% |
| always_canary | 1150 | 1150 | 0 | 2020.0000 | 1.7565 | 0.0% | 100.0% | 0.0% |
| always_block | 1150 | 1150 | 0 | 1865.0000 | 1.6217 | 0.0% | 0.0% | 100.0% |
| linucb_bias_only | 1150 | 1150 | 0 | 1750.0000 | 1.5217 | 4.0% | 48.0% | 48.0% |
| static_rules | 1150 | 1150 | 0 | 1853.5000 | 1.6117 | 2.6% | 60.3% | 37.1% |
| heuristic_score | 1150 | 1150 | 0 | 2294.0000 | 1.9948 | 37.0% | 63.0% | 0.0% |
| linucb | 1150 | 1150 | 0 | 1950.0000 | 1.6957 | 9.1% | 21.7% | 69.1% |
| linucb_with_drift | 1150 | 1150 | 0 | 1950.0000 | 1.6957 | 9.1% | 21.7% | 69.1% |
| thompson | 1150 | 1150 | 0 | 1938.5000 | 1.6857 | 3.3% | 16.2% | 80.5% |
