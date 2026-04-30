# Phase 2: Baseline System - Static CI/CD

## Goal

Implement the static CI/CD baseline pipeline that the self-adaptive system will be compared against.

## Purpose

The baseline answers the research question: how good is a normal static CI/CD deployment gate before adaptive feedback is introduced?

This phase creates the benchmark that later MAPE-K experiments must outperform.

## Baseline Logic

```python
if tests_passed and coverage > 80:
    deploy()
else:
    block()
```

## What This Means

The baseline system does not adapt. It only checks fixed rules:

- did tests pass?
- is code coverage above 80%?
- if yes, deploy
- if no, block

It does not learn from past failures, commit size, risky files, rollback history, deployment outcomes, or dynamic risk score.

## Failure Simulation

Because real production failures may not always be available, Phase 2 includes simulated deployment outcomes.

Example simulation factors:

- large commits increase failure chance
- risky folders increase failure chance
- failed tests block deployment
- low coverage blocks deployment

Risky folders may include:

- `auth/`
- `payments/`
- `database/`
- `deploy/`
- `infrastructure/`
- `config/`

## Phase 1 Compatibility

The Phase 1 SQLite schema stores commit and CI fields but does not yet store coverage or changed file paths. To keep the baseline reproducible, `experiments/baseline.py` uses real coverage and risky-folder fields when they are present. If they are missing, it generates deterministic simulated values from the stored deployment record.

This means the same input record always produces the same baseline decision and simulated outcome.

## Deliverable

```text
experiments/baseline.py
```

## Metrics Collected

| Metric | Meaning |
| --- | --- |
| Success rate | Percentage of deployed changes that succeeded |
| Failure rate | Percentage of deployed changes that failed |
| MTTR | Simulated recovery time after deployed failure |
| False positives | Safe deployments incorrectly blocked |
| False negatives | Risky deployments incorrectly allowed |

## Exit Criteria

Phase 2 is complete when:

- baseline logic is implemented
- baseline can run over the dataset from Phase 1
- each deployment receives a decision: `deploy` or `block`
- simulated outcomes are generated
- metrics are calculated
- the system can produce a baseline result table

## Research Importance

This phase is important because the adaptive system needs a fair comparison. The baseline establishes the performance of a normal static CI/CD gate before feedback, learning, or adaptive rollback decisions are introduced.

