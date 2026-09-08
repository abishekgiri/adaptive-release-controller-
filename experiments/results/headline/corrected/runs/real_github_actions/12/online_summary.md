# Online Replay Result: real_github_actions

> **Online replay costs are simulation artefacts. CI outcome used as counterfactual proxy. Do not report as causal real-world cost estimates.**

Seed: `12`  |  Trajectories: `2`

| Policy | Steps | Updates | Censored | Cumul. Cost | Mean Cost/Step | Deploy% | Canary% | Block% |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cost_rule | 600 | 578 | 22 | 624.5000 | 1.0804 | 63.3% | 8.8% | 27.8% |
| bayesian_rate | 600 | 578 | 22 | 669.0000 | 1.1574 | 55.0% | 23.2% | 21.8% |
| always_deploy | 600 | 578 | 22 | 860.0000 | 1.4879 | 100.0% | 0.0% | 0.0% |
| always_canary | 600 | 578 | 22 | 836.0000 | 1.4464 | 0.0% | 100.0% | 0.0% |
| always_block | 600 | 578 | 22 | 1027.0000 | 1.7768 | 0.0% | 0.0% | 100.0% |
| linucb_bias_only | 600 | 578 | 22 | 650.0000 | 1.1246 | 56.3% | 18.7% | 25.0% |
| static_rules | 600 | 578 | 22 | 640.0000 | 1.1073 | 64.3% | 19.7% | 16.0% |
| heuristic_score | 600 | 578 | 22 | 860.0000 | 1.4879 | 100.0% | 0.0% | 0.0% |
| linucb | 600 | 578 | 22 | 779.0000 | 1.3478 | 26.7% | 30.5% | 42.8% |
| linucb_with_drift | 600 | 578 | 22 | 779.0000 | 1.3478 | 26.7% | 30.5% | 42.8% |
| thompson | 600 | 578 | 22 | 675.0000 | 1.1678 | 51.7% | 6.3% | 42.0% |
