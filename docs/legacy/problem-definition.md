# Problem Definition

## Research Problem

Modern CI/CD pipelines usually make deployment decisions using static rules such as:

- tests passed
- build succeeded
- lint passed
- approval received

However, these rules do not fully capture deployment risk. A deployment can pass CI but still fail in production due to hidden risks such as risky code changes, weak test coverage, flaky behavior, dependency changes, or previous failure history.

## Research Question

Can a self-adaptive feedback loop improve deployment reliability compared to a static CI/CD pipeline?

## Failure Definition

A deployment is considered a failure if one or more of the following happen after deployment:

1. The deployment is rolled back.
2. A critical test or smoke check fails after deployment.
3. The system returns a high error rate after deployment.
4. The deployment causes service unavailability.
5. The deployment is manually marked as failed.

## Success Definition

A deployment is considered successful if:

1. CI/CD completes successfully.
2. No rollback occurs.
3. Post-deployment smoke checks pass.
4. Error rate stays within the acceptable threshold.
5. The deployment remains stable during the observation window.

## Dataset Scope

This project will use a simulated deployment dataset first.

Each deployment record will include:

- commit size
- files changed
- test result
- build result
- previous failure history
- dependency change
- risk score
- deployment decision
- final outcome

Real GitHub Actions data may be added later as an extension.

## Metrics

The system will be evaluated using:

- deployment failure rate
- rollback rate
- false approval rate
- false rejection rate
- average risk score accuracy
- comparison against baseline CI/CD

