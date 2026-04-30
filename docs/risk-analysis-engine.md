# Phase 3: Risk Analysis Engine

## Goal

Build a change risk predictor that estimates how dangerous a deployment is before it is released.

## Purpose

The static CI/CD baseline only checks tests and coverage. The risk analysis engine evaluates deployment risk using broader operational signals:

- commit size
- files changed
- lines changed
- CI duration
- past failures
- risky folders
- historical deployment outcomes

The first version uses a heuristic risk score. This makes the model explainable and gives the research a clear bridge from static CI/CD rules to a self-adaptive MAPE-K controller.

## Starting Model

The first risk model is a weighted heuristic:

```python
risk_score = (
    0.3 * file_count_score
    + 0.3 * lines_changed_score
    + 0.2 * past_failure_score
    + 0.2 * ci_time_score
)
```

Each component is normalized between `0.0` and `1.0`.

| Component | Normalization |
| --- | --- |
| `file_count_score` | `files_changed / 50`, capped at `1.0` |
| `lines_changed_score` | `(lines_added + lines_deleted) / 1500`, capped at `1.0` |
| `past_failure_score` | `past_failures / 10`, capped at `1.0` |
| `ci_time_score` | `ci_duration / 3600`, capped at `1.0` |

Example output:

```json
{
  "risk_score": 0.72,
  "confidence": 0.81
}
```

## Risk Levels

| Risk Score | Level | Meaning |
| --- | --- | --- |
| `0.00-0.39` | Low | Safe to deploy |
| `0.40-0.69` | Medium | Deploy with caution |
| `0.70-1.00` | High | Block or require review |

## Decision Logic

```python
if risk_score >= 0.70:
    decision = "block"
elif risk_score >= 0.40:
    decision = "review"
else:
    decision = "deploy"
```

## Components

```text
risk_engine/model.py
experiments/risk_eval.py
```

## Running The Evaluation

Generate a deterministic dataset if one does not already exist:

```bash
python3 -m experiments.generate_dataset --db knowledge_base/deployments.db --count 100 --reset
```

Run the risk analysis evaluation:

```bash
python3 -m experiments.risk_eval --db knowledge_base/deployments.db --limit 200
```

## Evaluation Method

The risk evaluator reads deployment records from the Phase 1 SQLite knowledge base. For each record, it:

1. derives historical `past_failures` from earlier deployment outcomes
2. calculates a risk score and confidence
3. assigns a risk level and decision
4. compares risk scores against final outcomes
5. reports whether failed deployments have higher average risk than successful deployments

## Metrics

| Metric | Meaning |
| --- | --- |
| Average risk for successful deployments | Mean risk score for records with `outcome = success` |
| Average risk for failed deployments | Mean risk score for records with `outcome = failure` |
| Risk separation | Failed average risk minus successful average risk |
| Failure correlation | Pearson correlation between risk score and failure label |
| Failure rate by risk level | Failure percentage for low, medium, and high risk groups |

## Optional ML Upgrade

After the heuristic version works, the system can be upgraded to:

- Logistic Regression
- Random Forest
- Gradient Boosting

Candidate input features:

- `files_changed`
- `lines_added`
- `lines_deleted`
- `ci_duration`
- `test_passed`
- `coverage`
- `past_failures`
- `risky_folder_touched`

Target label:

```text
outcome = success/failure
```

## Exit Criteria

Phase 3 is complete when:

- risk score is generated for each deployment
- risk score is stored or printed
- high-risk deployments fail more often than low-risk deployments
- risk score reasonably correlates with failure
- output includes `risk_score` and `confidence`

## Research Importance

This phase answers:

> Can deployment risk be estimated before release better than static CI/CD rules?

The target for this phase is simple: risk score should be higher for failed deployments than successful ones.
