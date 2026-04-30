# MAPE-K Feedback Loop Results

## Experiment Setup

| Item | Value |
| --- | --- |
| Phase | Phase 5 - MAPE-K Feedback Loop |
| Database | `knowledge_base/deployments.db` |
| Policy artifact | `experiments/results/learned-policy.json` |
| Deploy threshold before | 0.40 |
| Block threshold before | 0.70 |
| Deploy threshold after | 0.40 |
| Block threshold after | 0.70 |
| Adjustment | unchanged |

## Feedback Metrics

| Metric | Value |
| --- | --- |
| Total records | 100 |
| Deployed or canaried | 30 |
| Blocked | 70 |
| Successes | 38 |
| Failures | 62 |
| Success rate | 38.00% |
| Failure rate | 62.00% |
| False positives | 21 |
| False negatives | 13 |
| False positive rate | 21.00% |
| False negative rate | 13.00% |

## Learned Policy

| Field | Value |
| --- | --- |
| Previous deploy threshold | 0.40 |
| Previous block threshold | 0.70 |
| New deploy threshold | 0.40 |
| New block threshold | 0.70 |
| Reason | Error rates are within acceptable limits. |

## Decision Sample

| Commit | Risk Score | Decision | Outcome |
| --- | --- | --- | --- |
| a2a59b2d9dbb | 0.7000 | BLOCK | success |
| 812fa8378bc6 | 0.6500 | CANARY | success |
| db61d5f7a1d9 | 0.6500 | CANARY | failure |
| 68c88137c125 | 0.8000 | BLOCK | success |
| 36e77601ee8d | 0.8200 | BLOCK | failure |
| e59160ad66a8 | 0.7000 | BLOCK | failure |
| 255721d93c73 | 1.0000 | BLOCK | success |
| f12b017ed15b | 1.0000 | BLOCK | success |
| 4fc5f941d555 | 0.9920 | BLOCK | failure |
| 5b9cc2d476ca | 0.6500 | CANARY | failure |
| 62019026e2b6 | 1.0000 | BLOCK | failure |
| cb2f1db1237d | 0.7500 | BLOCK | success |
| e53fc31448ec | 0.7500 | BLOCK | failure |
| ae7a8b2733e9 | 1.0000 | BLOCK | failure |
| d807c9808423 | 0.9000 | BLOCK | failure |
| b972605e7fb0 | 0.7300 | BLOCK | success |
| 18902f7f14f7 | 0.7675 | BLOCK | failure |
| 734f72aec147 | 0.7800 | BLOCK | success |
| b1c3abf09e70 | 0.7000 | BLOCK | success |
| 044670a7dbe5 | 1.0000 | BLOCK | failure |

## Adaptation Validation Examples

| Scenario | False Negative Rate | False Positive Rate | Deploy Before | Deploy After | Block Before | Block After | Adjustment |
| --- | --- | --- | --- | --- | --- | --- | --- |
| High false negative rate | 66.67% | 0.00% | 0.40 | 0.35 | 0.70 | 0.65 | increase_risk_sensitivity |
| High false positive rate | 0.00% | 66.67% | 0.40 | 0.45 | 0.70 | 0.75 | reduce_unnecessary_blocking |
| Acceptable rates | 0.00% | 0.00% | 0.40 | 0.40 | 0.70 | 0.70 | unchanged |

## Interpretation

The observed false positive and false negative rates are acceptable, so the feedback loop kept the policy unchanged.
