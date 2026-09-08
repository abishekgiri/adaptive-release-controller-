# Online Replay Result: first_per_commit

> **Online replay costs are simulation artefacts. CI outcome used as counterfactual proxy. Do not report as causal real-world cost estimates.**

Seed: `22`  |  Trajectories: `2`

| Policy | Steps | Updates | Censored | Cumul. Cost | Mean Cost/Step | Deploy% | Canary% | Block% |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cost_rule | 156 | 151 | 5 | 199.5000 | 1.3212 | 48.7% | 14.7% | 36.5% |
| bayesian_rate | 156 | 151 | 5 | 219.5000 | 1.4536 | 46.2% | 23.7% | 30.1% |
| always_deploy | 156 | 151 | 5 | 310.0000 | 2.0530 | 100.0% | 0.0% | 0.0% |
| always_canary | 156 | 151 | 5 | 244.0000 | 1.6159 | 0.0% | 100.0% | 0.0% |
| always_block | 156 | 151 | 5 | 255.5000 | 1.6921 | 0.0% | 0.0% | 100.0% |
| linucb_bias_only | 156 | 151 | 5 | 202.0000 | 1.3377 | 56.4% | 8.3% | 35.3% |
| static_rules | 156 | 151 | 5 | 223.0000 | 1.4768 | 51.3% | 23.7% | 25.0% |
| heuristic_score | 156 | 151 | 5 | 305.0000 | 2.0199 | 98.7% | 1.3% | 0.0% |
| linucb | 156 | 151 | 5 | 217.0000 | 1.4371 | 30.1% | 39.7% | 30.1% |
| linucb_with_drift | 156 | 151 | 5 | 217.0000 | 1.4371 | 30.1% | 39.7% | 30.1% |
| thompson | 156 | 151 | 5 | 207.0000 | 1.3709 | 32.1% | 37.8% | 30.1% |
