# Evaluation Protocol

## Purpose

Phase 16 locks the paper evidence plan for the cost-sensitive delayed contextual bandit. The goal is to test the claim that deployment control is better modeled as sequential cost minimization than as static CI/CD classification.

Primary claim:

> A cost-sensitive contextual bandit with delayed reward updates and simple drift adaptation achieves lower cumulative operational cost than static rules and non-adaptive bandit baselines on CI/CD trajectory data.

## Datasets

| Dataset | Role | Source | Loader | Use in Phase 16 |
| --- | --- | --- | --- | --- |
| TravisTorrent | Primary real-data replay | Local CSV in `data/raw/travistorrent.csv` | `data/loaders.py::TravisTorrentLoader` | Main offline IPS evaluation |
| Synthetic deployment environment | Controlled stress test | `environment/synthetic.py` | Direct environment simulation | Delay, censoring, and drift evidence |
| ApacheJIT | Fallback only | Local extracted files if available | Loader is not implemented yet | Not used in Phase 16 results |
| GitHub Actions export | Deferred | `ingestion/github_client.py` | Not part of current pipeline | Future external validation |

TravisTorrent filtering is fixed to the current config:

| Filter | Value |
| --- | ---: |
| Minimum builds per project | 500 |
| Minimum project history | 365 days |
| Raw data location | `data/raw/` |
| Results location | `experiments/results/<config_name>/<seed>/` |

## Policies Compared

| Policy | Implementation | Status | Purpose |
| --- | --- | --- | --- |
| Static rules | `policies/static_rules.py` | Implemented | Deterministic CI/CD-style baseline |
| Heuristic score | `policies/heuristic_score.py` | TODO placeholder | Fixed risk-score baseline when implemented |
| Offline classifier | `policies/offline_classifier.py` | TODO placeholder | Batch ML baseline when implemented |
| LinUCB | `policies/linucb.py` | Implemented | Standard contextual bandit baseline |
| Cost-sensitive bandit | `policies/cost_sensitive_bandit.py` | Implemented | Research contribution |
| Thompson sampling | `policies/thompson.py` | TODO placeholder | Bayesian bandit baseline when implemented |

The headline Phase 16 comparison uses only implemented policies: static rules, LinUCB, and cost-sensitive bandit. TODO policies must appear in result tables as unavailable rather than silently disappearing.

## Exact Experiment Matrix

### Real-Data Replay

| Experiment ID | Dataset | Policies | Cost matrix | Delay setting | Drift setting | Seeds | Estimator |
| --- | --- | --- | --- | --- | --- | --- | --- |
| RQ1-real-default | TravisTorrent | static-rules, linucb, cost-sensitive-bandit | default | `ceil(tr_duration / 60)` | none | 0-29 | IPS |
| RQ2-cost-low-block | TravisTorrent | static-rules, linucb, cost-sensitive-bandit | low block penalty | `ceil(tr_duration / 60)` | none | 0-29 | IPS |
| RQ2-cost-high-failure | TravisTorrent | static-rules, linucb, cost-sensitive-bandit | high failure penalty | `ceil(tr_duration / 60)` | none | 0-29 | IPS |
| RQ2-cost-canary-heavy | TravisTorrent | static-rules, linucb, cost-sensitive-bandit | high canary overhead | `ceil(tr_duration / 60)` | none | 0-29 | IPS |

### Synthetic Stress Tests

| Experiment ID | Dataset | Policies | Cost matrix | Delay setting | Drift setting | Seeds | Estimator |
| --- | --- | --- | --- | --- | --- | --- | --- |
| RQ3-delay-none | Synthetic | linucb, cost-sensitive-bandit | default | fixed delay = 1 | default schedule | 0-29 | online cost |
| RQ3-delay-moderate | Synthetic | linucb, cost-sensitive-bandit | default | geometric `p = 0.30`, max = 20 | default schedule | 0-29 | online cost |
| RQ3-delay-heavy | Synthetic | linucb, cost-sensitive-bandit | default | geometric `p = 0.10`, max = 50 | default schedule | 0-29 | online cost |
| RQ4-drift-none | Synthetic | linucb, cost-sensitive-bandit | default | geometric `p = 0.30`, max = 20 | stationary single segment | 0-29 | online cost |
| RQ4-drift-abrupt-reset | Synthetic | linucb, cost-sensitive-bandit | default | geometric `p = 0.30`, max = 20 | abrupt segment length = 200, reset on drift | 0-29 | online cost |
| RQ4-drift-abrupt-no-reset | Synthetic | linucb, cost-sensitive-bandit | default | geometric `p = 0.30`, max = 20 | abrupt segment length = 200, no reset | 0-29 | online cost |

## Cost Matrices

The default cost matrix is the current `CostConfig` in `rewards/cost_model.py`.

| Matrix | Deploy success | Deploy failure | Canary success | Canary failure | Block safe | Block bad | Block unknown |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| default | 0.0 | 10.0 | 1.0 | 4.0 | 2.0 | 0.5 | 2.0 |
| low block penalty | 0.0 | 10.0 | 1.0 | 4.0 | 1.0 | 0.5 | 1.0 |
| high failure penalty | 0.0 | 20.0 | 1.0 | 8.0 | 2.0 | 0.5 | 2.0 |
| high canary overhead | 0.0 | 10.0 | 3.0 | 6.0 | 2.0 | 0.5 | 2.0 |

## Delay Settings

| Setting | Definition | Applies to |
| --- | --- | --- |
| TravisTorrent replay delay | `delay_steps = max(1, ceil(tr_duration / 60))` | Real-data replay |
| Fixed delay | `delay_steps = 1` | Synthetic ablation |
| Moderate stochastic delay | `k ~ Geom(0.30)`, max delay 20 | Synthetic main stress |
| Heavy stochastic delay | `k ~ Geom(0.10)`, max delay 50 | Synthetic stress |
| Censoring | rewards beyond max delay are marked censored | Synthetic stress |

Policy updates are valid only when delayed rewards are released by `delayed/buffer.py`.

## Drift Settings

| Setting | Definition | Applies to |
| --- | --- | --- |
| No drift | One stationary segment for the whole horizon | Synthetic ablation |
| Default abrupt drift | Two alternating segments, segment length 200 | Synthetic main stress |
| Drift reset | Reset policy model when detector fires | Contribution setting |
| No reset | Detector disabled or reset disabled | Ablation |

The default synthetic schedule alternates between a low-risk segment and a high-risk segment, matching `environment/synthetic.py`.

## Seeds

All randomized experiments use 30 fixed seeds:

```text
0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
20, 21, 22, 23, 24, 25, 26, 27, 28, 29
```

Development smoke runs may use seeds `0-4`, but paper tables must use `0-29`.

## Metrics

Headline metrics must be cost-first:

| Metric | Definition | Source |
| --- | --- | --- |
| Cumulative operational cost | Sum of finite observed costs | `evaluation/metrics.py` |
| IPS-estimated policy value | Negative mean IPS-weighted cost | `evaluation/replay_eval.py` |
| Matched actions | Count where candidate action equals logged action | `evaluation/replay_eval.py` |
| Effective sample size | IPS weight quality diagnostic | `evaluation/replay_eval.py` |
| Delayed updates applied | Number of matured rewards used for learning | `CostSensitiveBandit.stats` |
| Drift resets | Number of policy resets after detected drift | `CostSensitiveBandit.stats` |
| Action distribution | Fraction of deploy/canary/block decisions | `evaluation/metrics.py` |
| Cumulative regret | Placeholder until oracle-cost path is complete | `evaluation/metrics.py` |

Accuracy, precision, recall, and F1 are not headline metrics. They may appear only as appendix diagnostics if a classifier baseline is implemented.

## Statistical Tests

| Test | Use | Configuration |
| --- | --- | --- |
| Percentile bootstrap CI | 95% confidence intervals over seeds/projects | 10,000 resamples, fixed seed |
| Paired bootstrap | Pairwise policy comparison on matched seeds | two-sided |
| Holm-Bonferroni correction | Multiple policy comparisons | family-wise alpha = 0.05 |
| ESS reporting | IPS reliability diagnostic | report with every replay result |

The primary null hypothesis is:

```text
H0: cost-sensitive-bandit has the same or higher cumulative operational cost than the best baseline.
H1: cost-sensitive-bandit has lower cumulative operational cost than the best baseline.
```

## Threats to Validity

| Threat | Risk | Mitigation |
| --- | --- | --- |
| Build failure is not production failure | TravisTorrent observes CI outcomes, not incidents | State this as proxy validity; use synthetic environment for deployment semantics |
| Unknown logging propensities | IPS assumes propensities; TravisTorrent lacks true logging policy | Report ESS, matched actions, propensity assumption, and clipping |
| Limited action coverage | Real logs mostly represent deploy/run-build behavior | Treat canary/block replay as conservative and emphasize synthetic support |
| Synthetic assumptions | Stress tests may encode unrealistic drift/delay patterns | Keep synthetic results as mechanism evidence, not external validity |
| Cost matrix subjectivity | Operational costs vary by company | Sweep cost matrices and report robustness |
| Non-stationarity confounding | Project evolution may mix with policy learning effects | Evaluate per project and aggregate with bootstrap CIs |
| Censored rewards | Missing outcomes can bias learning and evaluation | Skip censored costs consistently and report censored count |
| Multiple comparisons | Many policies/settings can create false positives | Use Holm-Bonferroni correction |

## Exit Criteria

Phase 16 is complete when the paper can report:

1. A real-data replay table comparing static rules, LinUCB, and cost-sensitive bandit.
2. A cost-sensitivity table showing conclusions are not tied to one cost matrix.
3. A delay-stress figure showing performance degrades gracefully under delayed feedback.
4. A drift-stress figure showing whether reset adaptation reduces post-drift cost.
5. A threat-to-validity section that clearly separates real-data proxy evidence from synthetic mechanism evidence.
