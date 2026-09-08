# Online Replay Result: alpha_10.0

> **Online replay costs are simulation artefacts. CI outcome used as counterfactual proxy. Do not report as causal real-world cost estimates.**

Seed: `0`  |  Trajectories: `2`

| Policy | Steps | Updates | Censored | Cumul. Cost | Mean Cost/Step | Deploy% | Canary% | Block% |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cost_rule | 600 | 578 | 22 | 624.5000 | 1.0804 | 63.3% | 8.8% | 27.8% |
| bayesian_rate | 600 | 578 | 22 | 669.0000 | 1.1574 | 55.0% | 23.2% | 21.8% |
| always_deploy | 600 | 578 | 22 | 860.0000 | 1.4879 | 100.0% | 0.0% | 0.0% |
| always_canary | 600 | 578 | 22 | 836.0000 | 1.4464 | 0.0% | 100.0% | 0.0% |
| always_block | 600 | 578 | 22 | 1027.0000 | 1.7768 | 0.0% | 0.0% | 100.0% |
| linucb_bias_only | 600 | 578 | 22 | 648.0000 | 1.1211 | 51.8% | 24.0% | 24.2% |
| static_rules | 600 | 578 | 22 | 640.0000 | 1.1073 | 64.3% | 19.7% | 16.0% |
| heuristic_score | 600 | 578 | 22 | 860.0000 | 1.4879 | 100.0% | 0.0% | 0.0% |
| linucb | 600 | 578 | 22 | 733.0000 | 1.2682 | 43.3% | 29.0% | 27.7% |
| linucb_with_drift | 600 | 578 | 22 | 733.0000 | 1.2682 | 43.3% | 29.0% | 27.7% |
| thompson | 600 | 578 | 22 | 654.0000 | 1.1315 | 51.0% | 15.8% | 33.2% |
