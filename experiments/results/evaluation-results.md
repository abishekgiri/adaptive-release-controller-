# Experimentation & Evaluation Results

## Experiment Setup

| Item | Value |
| --- | --- |
| Phase | Phase 6 - Experimentation & Evaluation |
| Dataset | `knowledge_base/deployments.db` |
| Dataset size | 100 |
| Adaptive policy source | learned_policy |
| Adaptive deploy threshold | 0.35 |
| Adaptive block threshold | 0.65 |

## System Comparison

| System | Success Rate | Failure Rate | MTTR | False Positive Rate | False Negative Rate | Decision Accuracy |
| --- | --- | --- | --- | --- | --- | --- |
| Static | 46.88% | 53.12% | 60.00 | 23.00% | 17.00% | 60.00% |
| Risk-only | 56.67% | 43.33% | 30.00 | 21.00% | 13.00% | 66.00% |
| Adaptive | 60.00% | 40.00% | 30.00 | 29.00% | 6.00% | 65.00% |

## Decision Distribution

| System | DEPLOY | CANARY | BLOCK |
| --- | --- | --- | --- |
| Static | 32 | 0 | 68 |
| Risk-only | 0 | 30 | 70 |
| Adaptive | 0 | 15 | 85 |

## Adaptive Behavior Under Increased Sensitivity

| Field | Value |
| --- | --- |
| Deploy threshold before | 0.40 |
| Deploy threshold after | 0.35 |
| Block threshold before | 0.70 |
| Block threshold after | 0.65 |
| Sensitivity threshold | 10.00% |
| Adjustment | increase_risk_sensitivity |
| Risk-only failure rate | 43.33% |
| Adaptive failure rate | 40.00% |
| Risk-only decision accuracy | 66.00% |
| Adaptive decision accuracy | 65.00% |

## Graph Outputs

| Graph | Path |
| --- | --- |
| Success rate comparison | `experiments/results/graphs/success_rate_comparison.png` |
| Failure rate comparison | `experiments/results/graphs/failure_rate_comparison.png` |
| False positive vs false negative comparison | `experiments/results/graphs/error_rate_comparison.png` |
| MTTR comparison | `experiments/results/graphs/mttr_comparison.png` |
| Decision distribution comparison | `experiments/results/graphs/decision_distribution_comparison.png` |
| Adaptive improvement comparison | `experiments/results/graphs/adaptive_improvement_comparison.png` |

## Research Interpretation

The adaptive controller improved over the static baseline on lower failure rate, lower MTTR, lower false negative rate, higher decision accuracy. The adaptive system differs from risk-only because it uses the learned MAPE-K thresholds. Compared with risk-only control, the adaptive policy reduces failure rate from 43.33% to 40.00% by blocking more risky deployments. This provides the Phase 6 comparison needed to evaluate whether feedback-adjusted deployment control improves reliability compared to static CI/CD.
