# Experimentation & Evaluation Results

## Experiment Setup

| Item | Value |
| --- | --- |
| Phase | Phase 6 - Experimentation & Evaluation |
| Dataset | `knowledge_base/deployments.db` |
| Dataset size | 100 |
| Adaptive policy source | learned_policy |
| Adaptive deploy threshold | 0.40 |
| Adaptive block threshold | 0.70 |

## System Comparison

| System | Success Rate | Failure Rate | MTTR | False Positive Rate | False Negative Rate | Decision Accuracy |
| --- | --- | --- | --- | --- | --- | --- |
| Static | 46.88% | 53.12% | 60.00 | 23.00% | 17.00% | 60.00% |
| Risk-only | 56.67% | 43.33% | 30.00 | 21.00% | 13.00% | 66.00% |
| Adaptive | 56.67% | 43.33% | 30.00 | 21.00% | 13.00% | 66.00% |

## Decision Distribution

| System | DEPLOY | CANARY | BLOCK |
| --- | --- | --- | --- |
| Static | 32 | 0 | 68 |
| Risk-only | 0 | 30 | 70 |
| Adaptive | 0 | 30 | 70 |

## Graph Outputs

| Graph | Path |
| --- | --- |
| Success rate comparison | `experiments/results/graphs/success_rate_comparison.png` |
| Failure rate comparison | `experiments/results/graphs/failure_rate_comparison.png` |
| False positive vs false negative comparison | `experiments/results/graphs/error_rate_comparison.png` |
| MTTR comparison | `experiments/results/graphs/mttr_comparison.png` |
| Decision distribution comparison | `experiments/results/graphs/decision_distribution_comparison.png` |

## Research Interpretation

The adaptive controller improved over the static baseline on lower failure rate, lower MTTR, lower false negative rate, higher decision accuracy. The adaptive and risk-only systems match because the learned policy kept the default thresholds unchanged for this dataset. This provides the Phase 6 comparison needed to evaluate whether feedback-adjusted deployment control improves reliability compared to static CI/CD.
