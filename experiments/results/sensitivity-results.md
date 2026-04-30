# Sensitivity & Threshold Analysis

## Research Question

How does changing the feedback sensitivity threshold affect deployment failure rate, false positives, false negatives, and delivery velocity?

## Experiment Setup

| Item | Value |
| --- | --- |
| Dataset | `knowledge_base/deployments.db` |
| Sensitivity values | 0.05, 0.10, 0.15, 0.20, 0.25, 0.30 |
| Default deploy threshold | 0.40 |
| Default block threshold | 0.70 |
| Deployment velocity | Allowed releases divided by total deployment records |

## Sensitivity Sweep Results

| Sensitivity | Deploy Threshold | Block Threshold | Adjustment | Success Rate | Failure Rate | False Positive Rate | False Negative Rate | Deployment Velocity | Decision Accuracy | Tradeoff Score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.05 | 0.35 | 0.65 | increase_risk_sensitivity | 60.00% | 40.00% | 29.00% | 6.00% | 15.00% | 65.00% | 0.5880 |
| 0.10 | 0.35 | 0.65 | increase_risk_sensitivity | 60.00% | 40.00% | 29.00% | 6.00% | 15.00% | 65.00% | 0.5880 |
| 0.15 | 0.40 | 0.70 | unchanged | 56.67% | 43.33% | 21.00% | 13.00% | 30.00% | 66.00% | 0.5927 |
| 0.20 | 0.40 | 0.70 | unchanged | 56.67% | 43.33% | 21.00% | 13.00% | 30.00% | 66.00% | 0.5927 |
| 0.25 | 0.40 | 0.70 | unchanged | 56.67% | 43.33% | 21.00% | 13.00% | 30.00% | 66.00% | 0.5927 |
| 0.30 | 0.40 | 0.70 | unchanged | 56.67% | 43.33% | 21.00% | 13.00% | 30.00% | 66.00% | 0.5927 |

## Best Reliability and Velocity Tradeoff

The best reliability/velocity tradeoff in this run is sensitivity `0.15`. It uses thresholds `0.40` and `0.70`, producing a 43.33% failure rate, 13.00% false negative rate, 30.00% deployment velocity, and 66.00% decision accuracy.

## Graph Outputs

| Graph | Path |
| --- | --- |
| Sensitivity failure rate | `experiments/results/graphs/sensitivity_failure_rate.png` |
| Sensitivity tradeoff | `experiments/results/graphs/sensitivity_tradeoff.png` |

## Research Interpretation

Sensitivity values at or below 0.10 triggered conservative adaptation. Sensitivity values at or above 0.15 kept the default thresholds unchanged. Lower sensitivity made the controller more conservative by reducing the deploy and block thresholds, which lowered failure rate and false negatives but reduced deployment velocity. Higher sensitivity preserved delivery velocity but allowed more failed deployments. The selected tradeoff is `0.15` because it gives the strongest combined reliability and velocity score for this dataset.
