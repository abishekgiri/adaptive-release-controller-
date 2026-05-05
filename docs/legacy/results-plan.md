# Results Plan

## Research Questions

| ID | Question | Evidence |
| --- | --- | --- |
| RQ1 | Does the cost-sensitive delayed bandit reduce operational cost on real CI/CD logs? | TravisTorrent IPS replay |
| RQ2 | Are conclusions robust to different operational cost assumptions? | Cost matrix sweep |
| RQ3 | How much does delayed feedback hurt learning? | Synthetic delay stress tests |
| RQ4 | Does drift adaptation improve recovery after non-stationary shifts? | Synthetic drift stress tests |
| RQ5 | What tradeoff does the policy make between deploy, canary, and block? | Action distribution and cost breakdown |

## Expected Result Tables

### Table 1: Main Real-Data Replay Result

| Policy | Cumulative cost | IPS value | 95% CI | Matched actions | ESS | Censored skipped |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| static-rules | fill from run | fill from run | fill from run | fill from run | fill from run | fill from run |
| linucb | fill from run | fill from run | fill from run | fill from run | fill from run | fill from run |
| cost-sensitive-bandit | fill from run | fill from run | fill from run | fill from run | fill from run | fill from run |

Interpretation target: cost-sensitive-bandit should have the lowest cumulative cost or a statistically indistinguishable cost with a better delay/drift profile.

### Table 2: Cost Matrix Sensitivity

| Cost matrix | Best baseline cost | Cost-sensitive cost | Delta | 95% CI | Corrected p-value |
| --- | ---: | ---: | ---: | ---: | ---: |
| default | fill from run | fill from run | fill from run | fill from run | fill from run |
| low block penalty | fill from run | fill from run | fill from run | fill from run | fill from run |
| high failure penalty | fill from run | fill from run | fill from run | fill from run | fill from run |
| high canary overhead | fill from run | fill from run | fill from run | fill from run | fill from run |

Interpretation target: the contribution should remain competitive when the exact cost matrix changes.

### Table 3: Delay Ablation

| Delay setting | LinUCB cost | Cost-sensitive cost | Delayed updates applied | Pending/censored rewards | Delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| fixed delay = 1 | fill from run | fill from run | fill from run | fill from run | fill from run |
| geometric p = 0.30 | fill from run | fill from run | fill from run | fill from run | fill from run |
| geometric p = 0.10 | fill from run | fill from run | fill from run | fill from run | fill from run |

Interpretation target: delay should increase cost, but the delayed reward buffer should prevent invalid immediate-learning behavior.

### Table 4: Drift Adaptation

| Drift setting | Policy | Cumulative cost | Post-drift cost | Drift resets | Recovery steps |
| --- | --- | ---: | ---: | ---: | ---: |
| no drift | linucb | fill from run | n/a | n/a | n/a |
| no drift | cost-sensitive-bandit | fill from run | n/a | fill from run | n/a |
| abrupt drift, no reset | cost-sensitive-bandit | fill from run | fill from run | 0 | fill from run |
| abrupt drift, reset | cost-sensitive-bandit | fill from run | fill from run | fill from run | fill from run |

Interpretation target: reset adaptation should lower post-drift cost or shorten recovery compared with no reset.

### Table 5: Action Distribution

| Policy | Deploy % | Canary % | Block % | Mean cost per deploy | Mean cost per canary | Mean cost per block |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| static-rules | fill from run | fill from run | fill from run | fill from run | fill from run | fill from run |
| linucb | fill from run | fill from run | fill from run | fill from run | fill from run | fill from run |
| cost-sensitive-bandit | fill from run | fill from run | fill from run | fill from run | fill from run | fill from run |

Interpretation target: the contribution should use canary/block when failure cost is high, not merely mimic deploy-only behavior.

## Expected Figures

| Figure | File target | Description |
| --- | --- | --- |
| Figure 1 | `experiments/results/figures/main_cost_comparison.png` | Bar chart: cumulative cost by policy with 95% CI |
| Figure 2 | `experiments/results/figures/cost_sweep.png` | Line chart: cost delta across cost matrices |
| Figure 3 | `experiments/results/figures/delay_stress.png` | Cost vs. delay severity |
| Figure 4 | `experiments/results/figures/drift_recovery.png` | Cumulative cost around drift boundary |
| Figure 5 | `experiments/results/figures/action_distribution.png` | Stacked bar: deploy/canary/block mix |
| Figure 6 | `experiments/results/figures/ips_diagnostics.png` | ESS and matched-action diagnostics |

All figures must be generated with matplotlib only.

## Success Criteria

The paper claim is strong if the following conditions hold:

1. Cost-sensitive-bandit reduces cumulative operational cost by at least 10% versus static-rules on the default TravisTorrent replay.
2. Cost-sensitive-bandit beats or matches LinUCB on cumulative cost while preserving valid delayed updates.
3. Under high failure penalty, cost-sensitive-bandit reduces failed-deploy cost more than static-rules and LinUCB.
4. Under heavy delay, cost-sensitive-bandit does not use rewards before their scheduled observation step.
5. Under abrupt drift, reset adaptation reduces post-drift cost or recovery time compared with no reset.

The claim is weak if:

1. IPS effective sample size is too low to compare cautious policies.
2. The cost-sensitive bandit only wins under one hand-picked cost matrix.
3. Synthetic drift results improve but real-data replay does not.
4. The policy collapses to always block or always deploy.

## Paper Evidence Narrative

The results section should make three distinct points:

1. Real CI/CD replay shows practical cost reduction under the available logged-data assumptions.
2. Synthetic stress tests isolate why delayed feedback and drift handling matter.
3. Cost sweeps show the result is about operational tradeoffs, not a single arbitrary penalty table.

The paper should not claim production outage reduction from TravisTorrent alone. TravisTorrent supports CI/CD trajectory evidence; synthetic experiments support mechanism evidence for deployment delay, canary, and drift.

## Artifact Checklist

Before writing the paper results section, the repository should contain:

| Artifact | Required? | Status |
| --- | --- | --- |
| Config for main TravisTorrent replay | yes | `experiments/configs/first_real_result.json` exists |
| Raw data hash in `data/README.md` | yes | fill before final run |
| JSON result per seed | yes | write under `experiments/results/<config>/<seed>/` |
| Markdown summary per seed | yes | write under `experiments/results/<config>/<seed>/` |
| Aggregate result tables | yes | to be generated after Phase 16 |
| Figure scripts | yes | `evaluation/plots.py` exists as target |
| Statistical test output | yes | use `evaluation/statistical.py` |
| Threats to validity | yes | locked in `docs/evaluation-protocol.md` |

## Threats to Validity

| Category | Threat | Planned treatment in paper |
| --- | --- | --- |
| Construct validity | CI failure is used as a deployment-failure proxy | Explicitly state proxy limitation and avoid production-outage overclaim |
| Internal validity | Synthetic hidden-state assumptions may favor bandits | Keep real replay as primary evidence and separate synthetic mechanism results |
| External validity | TravisTorrent is old and OSS-heavy | Restrict claims to CI/CD trajectory decision support; propose GitHub Actions export as future work |
| Statistical validity | Multiple policies and cost matrices inflate false discoveries | Use paired bootstrap and Holm-Bonferroni correction |
| Evaluation validity | Unknown logging propensities affect IPS | Report matched actions, ESS, propensity assumption, and clipping |
| Reproducibility | Raw datasets are local and gitignored | Record dataset version, hash, filters, seeds, and configs |

## Phase 16 Exit Criteria

Phase 16 is done when these documents define exactly what evidence the paper needs, without implementing new algorithms or changing the codebase:

1. `docs/evaluation-protocol.md`
2. `docs/algorithm.md`
3. `docs/results-plan.md`
