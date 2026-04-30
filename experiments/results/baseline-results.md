# Static CI/CD Baseline Results

## Experiment Setup

| Item | Value |
| --- | --- |
| Phase | Phase 2 - Baseline System |
| Baseline | Static CI/CD Pipeline |
| Dataset type | Simulated deployment records |
| Dataset size | 100 deployment records |
| Database | `knowledge_base/deployments.db` |
| Dataset command | `python3 -m experiments.generate_dataset --db knowledge_base/deployments.db --count 100 --reset` |
| Baseline command | `python3 -m experiments.baseline --db knowledge_base/deployments.db --limit 200` |

## Baseline Decision Rule

```python
if tests_passed and coverage > 80:
    decision = "deploy"
else:
    decision = "block"
```

The baseline does not use commit size, risky folders, rollback history, previous failures, deployment outcomes, or adaptive risk feedback.

## Aggregate Metrics

| Metric | Value |
| --- | --- |
| Total records | 100 |
| Deployed | 32 |
| Blocked | 68 |
| Successful deployments | 15 |
| Failed deployments | 17 |
| Success rate | 46.88% |
| Failure rate | 53.12% |
| Average MTTR | 47.89 minutes |
| False positives | 23 |
| False negatives | 17 |
| False positive rate | 23.00% |
| False negative rate | 17.00% |

## Decision Sample

The baseline output prints the first 20 evaluated deployment records.

| Commit | Tests | Coverage | Decision | Outcome | MTTR |
| --- | --- | --- | --- | --- | --- |
| `74a840b449c5` | pass | 81.15 | deploy | failure | 57.13 |
| `e8e92efcdafd` | pass | 77.47 | block | success | 0.00 |
| `7158f65f2d73` | fail | 56.98 | block | failure | 0.00 |
| `36fd6623f464` | fail | 62.72 | block | failure | 0.00 |
| `4f20898c4507` | pass | 85.05 | deploy | success | 0.00 |
| `50368e6b5bf6` | fail | 57.02 | block | failure | 0.00 |
| `ae1e82e77d01` | pass | 74.19 | block | failure | 0.00 |
| `32374d26460b` | pass | 82.01 | deploy | success | 0.00 |
| `89c298d3eb2c` | pass | 78.63 | block | success | 0.00 |
| `fa0a98343ce5` | pass | 84.87 | deploy | success | 0.00 |
| `59f94d31f5d4` | fail | 57.23 | block | failure | 0.00 |
| `d36a3ed35c06` | pass | 87.29 | deploy | failure | 34.10 |
| `bded19400c87` | fail | 55.16 | block | failure | 0.00 |
| `78669a1ec374` | pass | 82.84 | deploy | failure | 25.48 |
| `182e9d67bdbf` | fail | 58.94 | block | failure | 0.00 |
| `8fffe32c596f` | pass | 76.43 | block | success | 0.00 |
| `d24a0b81db28` | pass | 85.04 | deploy | success | 0.00 |
| `f7e47abe312a` | pass | 85.61 | deploy | success | 0.00 |
| `1d89984fb82f` | fail | 52.79 | block | failure | 0.00 |
| `eabda96231fe` | pass | 77.51 | block | failure | 0.00 |

## Interpretation

The static CI/CD baseline allowed 32 deployments and blocked 68 records. Of the deployed changes, 17 failed, producing a 53.12% failure rate among deployed records. The baseline also produced 17 false negatives, meaning risky deployments were allowed through the static gate.

This result establishes the first measurable benchmark for the adaptive system. The future MAPE-K controller should aim to reduce failed deployments, false negatives, and MTTR while keeping false positives controlled.

