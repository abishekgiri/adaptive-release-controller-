# Adaptive Release Controller Research Summary

## Title

Self-Adaptive Deployment Controller Using MAPE-K Feedback Loop

## Research Question

Can a self-adaptive feedback loop improve deployment reliability compared to static CI/CD pipelines?

## Experiment Goal

This experiment compares three deployment-control strategies on the same deployment dataset:

- Static CI/CD: deploys only when tests pass and coverage is above the fixed threshold.
- Risk-only: uses the Phase 3 risk score with fixed Phase 4 thresholds.
- Adaptive: uses the MAPE-K feedback loop to learn stricter thresholds from historical outcomes.

The purpose is to determine whether adaptive feedback can reduce failed releases and false negatives compared with a traditional static pipeline and a non-adaptive risk model.

## Adaptive Policy Change

The feedback loop was run with increased sensitivity:

```bash
python3 -m experiments.feedback_loop_eval --db knowledge_base/deployments.db --limit 200 --sensitivity 0.10
```

This triggered the MAPE-K controller to become more conservative:

| Threshold | Before | After | Effect |
| --- | ---: | ---: | --- |
| Deploy threshold | 0.40 | 0.35 | Fewer changes qualify for direct deployment |
| Block threshold | 0.70 | 0.65 | More high-risk changes are blocked |

## Key Result

```text
Static -> Risk-only -> Adaptive
46.88% -> 56.67% -> 60.00% success
53.12% -> 43.33% -> 40.00% failure
17.00% -> 13.00% -> 6.00% false negatives
```

## Evaluation Results

| System | Success Rate | Failure Rate | MTTR | False Positive Rate | False Negative Rate | Decision Accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Static | 46.88% | 53.12% | 60.00 | 23.00% | 17.00% | 60.00% |
| Risk-only | 56.67% | 43.33% | 30.00 | 21.00% | 13.00% | 66.00% |
| Adaptive | 60.00% | 40.00% | 30.00 | 29.00% | 6.00% | 65.00% |

## Decision Distribution

| System | DEPLOY | CANARY | BLOCK |
| --- | ---: | ---: | ---: |
| Static | 32 | 0 | 68 |
| Risk-only | 0 | 30 | 70 |
| Adaptive | 0 | 15 | 85 |

## Research Interpretation

The adaptive MAPE-K system improved deployment reliability by lowering false negatives from 13.00% to 6.00% compared with the risk-only system, and from 17.00% to 6.00% compared with the static CI/CD baseline.

This means fewer risky deployments were incorrectly allowed through the pipeline. The adaptive controller also improved deployment success rate from 46.88% under static CI/CD to 60.00%, while reducing failure rate from 53.12% to 40.00%.

The tradeoff is a higher false positive rate. The adaptive system blocked more changes, increasing false positives from 21.00% under risk-only control to 29.00%. This is expected behavior for a more conservative release controller: it accepts more blocking in exchange for fewer failed deployments.

Overall, the result supports the research claim that a self-adaptive feedback loop can improve deployment reliability compared to static CI/CD gates and fixed-threshold risk-only decisioning.

## Reproducibility

Generate the learned adaptive policy:

```bash
python3 -m experiments.feedback_loop_eval --db knowledge_base/deployments.db --limit 200 --sensitivity 0.10
```

Run the evaluation with the learned policy:

```bash
python3 -m experiments.evaluation --db knowledge_base/deployments.db --limit 200 --use-adaptive-policy
```

Primary result files:

- `experiments/results/learned-policy.json`
- `experiments/results/evaluation-results.md`
- `experiments/results/evaluation-results.json`
- `experiments/results/graphs/adaptive_improvement_comparison.png`
