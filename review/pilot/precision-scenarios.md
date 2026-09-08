# Precision feasibility scenarios

No policy was fitted or compared. Descriptive outcome variability is **not** the between-project variance of paired policy-cost differences. The latter remains unidentified by this pilot. Do not substitute one for the other.

| Repository | Resolved scoped attempts | Failure rate | Within-project Bernoulli variance | Exploratory SHA-cluster bootstrap 95% interval |
|---|---:|---:|---:|---|
| expressjs/express | 68 | 0.2059 | 0.1635 | [0.11940298507462686, 0.3076923076923077] |
| pallets/flask | 88 | 0.7500 | 0.1875 | [0.6551724137931034, 0.8390804597701149] |
| psf/requests | 43 | 0.1395 | 0.1201 | [0.043478260869565216, 0.2631578947368421] |

Across these **three selected projects**, sample variance of observed failure proportions is 0.112189, SD 0.3349. This is a descriptive statistic with only two variance degrees of freedom, incompatible observation windows and possible shared infrastructure. It is not a stable population variance estimate. No defensible population variance confidence interval follows from this convenience sample.

The displayed intervals resample whole head-SHA clusters within project (2,000 descriptive bootstrap draws, fixed seed 20260907). They retain within-SHA dependence but not all serial, workflow or cross-project dependence. Treat them as exploratory and potentially too narrow, not confirmatory coverage guarantees. No policy seeds were run.

Effective project-level replication is at most three and may be smaller; genuinely independent future evaluation projects recruited: zero. Proven usable predecision observations per project: zero. Retrospective row counts cannot estimate a future 1,000-valid-decision accrual rate because preexecution capture and diff coverage are unknown.

## Unchanged-design sensitivity scenarios
Seven primary contrasts, family alpha .05, planning power .80, proposed normalized-cost effect .025. Scenarios below use the existing frozen normal approximation; they do not amend the evaluation protocol or estimate effects.

| Scenario | Assumed paired project-cost SD | Approximate independent evaluation repositories |
|---|---:|---:|
| Optimistic | 0.025 | 13 |
| Moderate | 0.05 | 50 |
| Conservative illustration | 0.1 | 200 |
| High heterogeneity | 0.2 | 799 |
| Bounded-difference stress case | 1.0 | 19957 |

These are optimistic known-SD normal calculations. Actual planning needs justified variance assumptions, finite-project correction, dependence sensitivity and a recruitment budget. “Conservative illustration” is not an upper bound. Pilot failure-rate heterogeneity does not tell us which cost-difference scenario applies. Smaller effects or Brier margins require their own scenarios.

At an assumed 2% failure rate, 1,000 valid decisions yield only 20 expected failures; at 5%, 50. Correlation, incomplete capture and nonbinary terminations reduce information further. None of the scoped pilot projects meets the frozen 50-failure/50-success subgroup reporting screen, and that screen itself never guaranteed validity.

No recruitment commitments, long-run collector capture fraction, verified predecision accrual rate or feasible independent-project budget has been established. Precision gate: FAIL for scaling now. Narrow exploratory scope or an explicitly reviewed acquisition/precision plan is needed before expansion.
