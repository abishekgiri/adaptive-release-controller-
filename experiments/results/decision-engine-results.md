# Decision Engine Results

## Experiment Setup

| Item | Value |
| --- | --- |
| Phase | Phase 4 - Decision Engine |
| Component | `decision_engine/engine.py` |
| Experiment command | `python3 -m experiments.decision_eval` |
| Canary threshold | `0.40` |
| Block threshold | `0.70` |

## Result Table

| Risk Score | Decision | Canary Threshold | Block Threshold | Reason |
| --- | --- | --- | --- | --- |
| 0.25 | DEPLOY | 0.4 | 0.7 | Risk score is below the canary threshold; deployment is approved. |
| 0.40 | CANARY | 0.4 | 0.7 | Risk score is between the canary and block thresholds; use a canary rollout. |
| 0.55 | CANARY | 0.4 | 0.7 | Risk score is between the canary and block thresholds; use a canary rollout. |
| 0.70 | BLOCK | 0.4 | 0.7 | Risk score meets or exceeds the block threshold; deployment should be blocked. |
| 0.85 | BLOCK | 0.4 | 0.7 | Risk score meets or exceeds the block threshold; deployment should be blocked. |

## Interpretation

The decision engine produces deterministic deployment actions from normalized risk scores. Low risk scores deploy normally, medium risk scores use canary rollout, and high risk scores are blocked.

