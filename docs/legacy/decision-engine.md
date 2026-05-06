# Phase 4: Decision Engine

## Goal

Move from prediction to action by converting a normalized risk score into a deterministic deployment decision.

## Purpose

Phase 3 predicts deployment risk. Phase 4 turns that prediction into an actionable release decision that can later be executed by the adaptive controller.

The decision engine does not change Phase 3 risk scoring behavior. It consumes the existing `risk_score` output and applies fixed, configurable thresholds.

## Component

```text
decision_engine/engine.py
```

## Decision Logic

| Risk Score | Decision | Meaning |
| --- | --- | --- |
| `< 0.40` | `DEPLOY` | Risk is low enough for normal deployment |
| `0.40-0.69` | `CANARY` | Risk needs a limited rollout |
| `>= 0.70` | `BLOCK` | Risk is high enough to stop deployment |

```python
if risk_score < 0.40:
    decision = "DEPLOY"
elif risk_score < 0.70:
    decision = "CANARY"
else:
    decision = "BLOCK"
```

## Structured Output

Each decision returns:

- `decision`
- `risk_score`
- `reason`
- `thresholds`

Example:

```json
{
  "decision": "CANARY",
  "risk_score": 0.55,
  "reason": "Risk score is between the canary and block thresholds; use a canary rollout.",
  "thresholds": {
    "canary": 0.4,
    "block": 0.7
  }
}
```

## Configurable Thresholds

The default thresholds are:

- canary threshold: `0.40`
- block threshold: `0.70`

They can be overridden when creating the engine:

```python
from decision_engine import DecisionEngine

engine = DecisionEngine(canary_threshold=0.35, block_threshold=0.75)
result = engine.decide(0.55)
```

## Helper Function

For simple usage:

```python
from decision_engine import decide_deployment

result = decide_deployment(0.85)
```

## Validation Rules

The engine validates that:

- `risk_score` is numeric
- `risk_score` is between `0.0` and `1.0`
- canary threshold is lower than block threshold
- thresholds are also between `0.0` and `1.0`

## Experiment

Run:

```bash
python3 -m experiments.decision_eval
```

Expected examples:

| Risk Score | Decision |
| --- | --- |
| `0.25` | `DEPLOY` |
| `0.55` | `CANARY` |
| `0.85` | `BLOCK` |

## Exit Criteria

Phase 4 is complete when the system consistently produces deterministic deployment decisions from normalized risk scores.

