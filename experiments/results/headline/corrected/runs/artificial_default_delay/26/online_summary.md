# Online Replay Result: artificial_default_delay

> **Online replay costs are simulation artefacts. CI outcome used as counterfactual proxy. Do not report as causal real-world cost estimates.**

Seed: `26`  |  Trajectories: `2`

| Policy | Steps | Updates | Censored | Cumul. Cost | Mean Cost/Step | Deploy% | Canary% | Block% |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cost_rule | 1150 | 1150 | 0 | 2085.0000 | 1.8130 | 32.6% | 19.0% | 48.3% |
| bayesian_rate | 1150 | 1150 | 0 | 1713.0000 | 1.4896 | 0.5% | 48.0% | 51.5% |
| always_deploy | 1150 | 1150 | 0 | 2900.0000 | 2.5217 | 100.0% | 0.0% | 0.0% |
| always_canary | 1150 | 1150 | 0 | 2020.0000 | 1.7565 | 0.0% | 100.0% | 0.0% |
| always_block | 1150 | 1150 | 0 | 1865.0000 | 1.6217 | 0.0% | 0.0% | 100.0% |
| linucb_bias_only | 1150 | 1150 | 0 | 1735.0000 | 1.5087 | 1.0% | 50.8% | 48.2% |
| static_rules | 1150 | 1150 | 0 | 1929.0000 | 1.6774 | 2.4% | 60.0% | 37.6% |
| heuristic_score | 1150 | 1150 | 0 | 2288.0000 | 1.9896 | 38.2% | 61.8% | 0.0% |
| linucb | 1150 | 1150 | 0 | 1877.5000 | 1.6326 | 7.7% | 29.8% | 62.5% |
| linucb_with_drift | 1150 | 1150 | 0 | 1877.5000 | 1.6326 | 7.7% | 29.8% | 62.5% |
| thompson | 1150 | 1150 | 0 | 1860.0000 | 1.6174 | 10.3% | 25.8% | 63.9% |
