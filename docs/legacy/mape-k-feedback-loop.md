# Phase 5: MAPE-K Feedback Loop

## Goal

Make the adaptive release controller self-adaptive by learning from past prediction outcomes and adjusting decision behavior over time.

## Purpose

Earlier phases produce deployment records, risk scores, and deployment decisions. Phase 5 closes the loop by comparing decisions against actual outcomes and producing a learned threshold policy.

The learned policy is saved as a separate artifact. Phase 5 does not change Phase 3 risk scoring and does not hardcode new defaults into the Phase 4 decision engine.

## MAPE-K Loop

| Step | Responsibility |
| --- | --- |
| Monitor | Load historical deployment decisions and outcomes from SQLite |
| Analyze | Compute success rate, failure rate, false positive rate, and false negative rate |
| Plan | Decide whether thresholds should become more or less conservative |
| Execute | Produce a new threshold policy |
| Knowledge | Persist the learned policy as JSON |

## Component

```text
knowledge_base/learning.py
```

## Experiment

```text
experiments/feedback_loop_eval.py
```

Run:

```bash
python3 -m experiments.feedback_loop_eval --db knowledge_base/deployments.db --limit 200
```

To increase sensitivity for demonstration or experimentation:

```bash
python3 -m experiments.feedback_loop_eval --db knowledge_base/deployments.db --limit 200 --sensitivity 0.10
```

The experiment saves:

```text
experiments/results/learned-policy.json
experiments/results/feedback-loop-results.md
```

## Adaptation Rules

The default thresholds are:

| Threshold | Default | Safe Bounds |
| --- | --- | --- |
| deploy threshold | `0.40` | `0.20-0.60` |
| block threshold | `0.70` | `0.50-0.90` |

Rules:

- If `false_negative_rate > 0.20`, lower thresholds by `0.05`.
- The false negative threshold is configurable with `--sensitivity`.
- If `false_positive_rate > 0.30`, raise thresholds by `0.05`.
- If both rates are acceptable, keep thresholds unchanged.
- `deploy_threshold` must always remain lower than `block_threshold`.

When false negatives are high, safety takes priority and thresholds are lowered to increase risk sensitivity.

## Metrics

| Metric | Meaning |
| --- | --- |
| success rate | fraction of historical records with successful outcomes |
| failure rate | fraction of historical records with failed outcomes |
| false positive rate | safe deployments incorrectly blocked |
| false negative rate | failed deployments that were deployed or canaried |

## Learned Policy Artifact

Example:

```json
{
  "adjustment": "increase_risk_sensitivity",
  "deploy_threshold": 0.35,
  "block_threshold": 0.65,
  "previous_deploy_threshold": 0.4,
  "previous_block_threshold": 0.7
}
```

## Exit Criteria

Phase 5 is complete when:

- historical deployment outcomes are loaded from the knowledge base
- feedback metrics are calculated
- thresholds adapt when false positive or false negative rates are high
- learned thresholds stay within safe bounds
- learned policy is saved as JSON
- results are saved as Markdown
