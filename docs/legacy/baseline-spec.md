# Baseline CI/CD Specification

## Baseline Name

Static CI/CD Pipeline

## Purpose

The baseline represents a normal CI/CD pipeline that uses fixed deployment rules without adaptive learning or risk analysis.

## Baseline Rules

A deployment is approved if:

1. Build passes.
2. Unit tests pass.
3. Lint checks pass.
4. No required approval is missing.

A deployment is rejected if:

1. Build fails.
2. Unit tests fail.
3. Lint checks fail.
4. Required approval is missing.

## Baseline Decision Logic

```text
IF build == pass
AND tests == pass
AND lint == pass
AND approval == true
THEN deploy
ELSE reject
```

## What Baseline Does Not Consider

The baseline does not consider:

- commit size
- number of changed files
- risky file areas
- dependency changes
- previous deployment failures
- flaky tests
- recent rollback history
- dynamic risk score

## Reproducibility

The baseline pipeline must always produce the same decision for the same input.

Same input = same output.

## Example

| Build | Tests | Lint | Approval | Baseline Decision |
| --- | --- | --- | --- | --- |
| pass | pass | pass | true | deploy |
| pass | fail | pass | true | reject |
| fail | pass | pass | true | reject |
| pass | pass | pass | false | reject |

## Exit Criteria

Phase 0 is complete when:

- failure is clearly defined
- success is clearly defined
- baseline rules are fixed
- dataset scope is selected
- baseline decision logic is reproducible

