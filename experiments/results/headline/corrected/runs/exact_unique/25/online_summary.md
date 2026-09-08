# Online Replay Result: exact_unique

> **Online replay costs are simulation artefacts. CI outcome used as counterfactual proxy. Do not report as causal real-world cost estimates.**

Seed: `25`  |  Trajectories: `2`

| Policy | Steps | Updates | Censored | Cumul. Cost | Mean Cost/Step | Deploy% | Canary% | Block% |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cost_rule | 575 | 566 | 9 | 622.5000 | 1.0998 | 61.9% | 9.2% | 28.9% |
| bayesian_rate | 575 | 566 | 9 | 669.5000 | 1.1829 | 53.2% | 23.0% | 23.8% |
| always_deploy | 575 | 566 | 9 | 860.0000 | 1.5194 | 100.0% | 0.0% | 0.0% |
| always_canary | 575 | 566 | 9 | 824.0000 | 1.4558 | 0.0% | 100.0% | 0.0% |
| always_block | 575 | 566 | 9 | 1003.0000 | 1.7721 | 0.0% | 0.0% | 100.0% |
| linucb_bias_only | 575 | 566 | 9 | 648.0000 | 1.1449 | 54.6% | 19.5% | 25.9% |
| static_rules | 575 | 566 | 9 | 639.0000 | 1.1290 | 63.0% | 20.3% | 16.7% |
| heuristic_score | 575 | 566 | 9 | 860.0000 | 1.5194 | 100.0% | 0.0% | 0.0% |
| linucb | 575 | 566 | 9 | 770.0000 | 1.3604 | 24.9% | 30.6% | 44.5% |
| linucb_with_drift | 575 | 566 | 9 | 770.0000 | 1.3604 | 24.9% | 30.6% | 44.5% |
| thompson | 575 | 566 | 9 | 681.0000 | 1.2032 | 49.7% | 7.1% | 43.1% |
