# Risk Analysis Engine Results

## Experiment Setup

| Item | Value |
| --- | --- |
| Phase | Phase 3 - Risk Analysis Engine |
| Model | Heuristic risk score |
| Dataset type | Simulated deployment records |
| Dataset size | 100 deployment records |
| Database | `knowledge_base/deployments.db` |
| Dataset command | `python3 -m experiments.generate_dataset --db knowledge_base/deployments.db --count 100 --reset` |
| Evaluation command | `python3 -m experiments.risk_eval --db knowledge_base/deployments.db --limit 200` |

## Risk Formula

```python
risk_score = (
    0.3 * file_count_score
    + 0.3 * lines_changed_score
    + 0.2 * past_failure_score
    + 0.2 * ci_time_score
)
```

## Aggregate Metrics

| Metric | Value |
| --- | --- |
| Total records | 100 |
| Successful deployments | 38 |
| Failed deployments | 62 |
| Average success risk | 0.5318 |
| Average failure risk | 0.5492 |
| Risk separation | 0.0174 |
| Failure correlation | 0.0705 |
| Low risk failure rate | 37.50% |
| Medium risk failure rate | 63.10% |
| High risk failure rate | 75.00% |

## Failure Rate By Risk Level

| Level | Records | Failures | Failure Rate |
| --- | --- | --- | --- |
| low | 8 | 3 | 37.50% |
| medium | 84 | 53 | 63.10% |
| high | 8 | 6 | 75.00% |

## Prediction Sample

| Commit | Risk Score | Confidence | Level | Decision | Outcome |
| --- | --- | --- | --- | --- | --- |
| `a2a59b2d9dbb` | 0.2921 | 0.936 | low | deploy | success |
| `812fa8378bc6` | 0.3559 | 0.9147 | low | deploy | success |
| `db61d5f7a1d9` | 0.3336 | 0.9221 | low | deploy | failure |
| `68c88137c125` | 0.4817 | 0.9272 | medium | review | success |
| `36e77601ee8d` | 0.4303 | 0.9101 | medium | review | failure |
| `e59160ad66a8` | 0.4673 | 0.9224 | medium | review | failure |
| `255721d93c73` | 0.2535 | 0.9488 | low | deploy | success |
| `f12b017ed15b` | 0.2877 | 0.9374 | low | deploy | success |
| `4fc5f941d555` | 0.429 | 0.9097 | medium | review | failure |
| `5b9cc2d476ca` | 0.2898 | 0.9367 | low | deploy | failure |
| `62019026e2b6` | 0.6022 | 0.9326 | medium | review | failure |
| `cb2f1db1237d` | 0.5122 | 0.9374 | medium | review | success |
| `e53fc31448ec` | 0.4327 | 0.9109 | medium | review | failure |
| `ae7a8b2733e9` | 0.4439 | 0.9146 | medium | review | failure |
| `d807c9808423` | 0.616 | 0.928 | medium | review | failure |
| `b972605e7fb0` | 0.4118 | 0.9039 | medium | review | success |
| `18902f7f14f7` | 0.4927 | 0.9309 | medium | review | failure |
| `734f72aec147` | 0.4678 | 0.9226 | medium | review | success |
| `b1c3abf09e70` | 0.7128 | 0.9043 | high | block | success |
| `044670a7dbe5` | 0.6908 | 0.9031 | medium | review | failure |

## Interpretation

Failed deployments have a higher average risk score than successful deployments: `0.5492` compared to `0.5318`. The risk separation is modest, but the failure rate increases by risk level: low-risk deployments fail at `37.50%`, medium-risk deployments fail at `63.10%`, and high-risk deployments fail at `75.00%`.

This satisfies the first Phase 3 target. The heuristic risk engine provides a measurable signal that can be compared against the static CI/CD baseline and later improved with adaptive feedback or supervised machine learning.

