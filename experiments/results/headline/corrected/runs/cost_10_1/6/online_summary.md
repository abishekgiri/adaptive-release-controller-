# Online Replay Result: cost_10_1

> **Online replay costs are simulation artefacts. CI outcome used as counterfactual proxy. Do not report as causal real-world cost estimates.**

Seed: `6`  |  Trajectories: `2`

| Policy | Steps | Updates | Censored | Cumul. Cost | Mean Cost/Step | Deploy% | Canary% | Block% |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cost_rule | 1150 | 1150 | 0 | 1450.5000 | 1.2613 | 51.6% | 24.5% | 23.9% |
| bayesian_rate | 1150 | 1150 | 0 | 1245.5000 | 1.0830 | 51.3% | 47.0% | 1.7% |
| always_deploy | 1150 | 1150 | 0 | 1450.0000 | 1.2609 | 100.0% | 0.0% | 0.0% |
| always_canary | 1150 | 1150 | 0 | 1440.0000 | 1.2522 | 0.0% | 100.0% | 0.0% |
| always_block | 1150 | 1150 | 0 | 1865.0000 | 1.6217 | 0.0% | 0.0% | 100.0% |
| linucb_bias_only | 1150 | 1150 | 0 | 1252.0000 | 1.0887 | 49.4% | 40.3% | 10.3% |
| static_rules | 1150 | 1150 | 0 | 1537.0000 | 1.3365 | 2.6% | 61.1% | 36.3% |
| heuristic_score | 1150 | 1150 | 0 | 1430.0000 | 1.2435 | 37.4% | 62.6% | 0.0% |
| linucb | 1150 | 1150 | 0 | 1477.5000 | 1.2848 | 2.9% | 90.3% | 6.9% |
| linucb_with_drift | 1150 | 1150 | 0 | 1477.5000 | 1.2848 | 2.9% | 90.3% | 6.9% |
| thompson | 1150 | 1150 | 0 | 1481.5000 | 1.2883 | 3.0% | 86.4% | 10.6% |
