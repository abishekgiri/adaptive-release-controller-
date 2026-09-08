# Bounded access effort and event-accrual decision

**Recommendation: SEEK ACCESS.** This means use the prepared package for a bounded attempt once the user chooses to send it; no outreach has been sent or scheduled. It does not ratify v2 or authorize collection, models, manuscript work or repository tests. Status remains CONDITIONALLY FEASIBLE; all six previously named blockers remain unresolved.

## Outreach budget and qualification milestones

Propose **six weeks, at most 12 serious targeted invitations**, and one follow-up per invitation after 7–10 days. Start the clock at the first actual invitation, not this document. No automatic messages or reminders are created. A serious attempt is a relevant owner/lead or named research-introduction request with the narrow summary, not a mass email or an unanswered generic repository issue. A follow-up is not another independent attempt; record introductions without double-counting organizations.

Maintain a minimal funnel: invitation ID/date/channel, relevant role, response/no-response, qualified conversation, nominated service/owner, screening outcome and reason, permission stage, monthly accrual estimate and shared infrastructure. A **qualified response** identifies a plausible existing gate operator and willingness to answer the screening questions; it is not yet a qualifying repository. Researchers offering advice without an operator introduction remain useful contacts but do not inflate the participant count.

| Review point | Minimum progress | Decision if absent |
|---|---|---|
| Week 3 | At least two qualified conversations or one owner actively supplying agreed screening aggregates | Use remaining invitations on direct owners/warm introductions; no expanded scraping or algorithm work |
| Week 6 | At least three qualified responses and **two repositories passing the hard screen**, with written passive-observation permission or a documented approval completing within two weeks | Stop preparation of the empirical paper. One repo can justify only a separately agreed bounded data-access case, not automatic continuation of this paper |
| At most two additional weeks | Allowed only for an already identified owner's documented administrative approval; no new recruitment extension | If access is still unresolved, stop. This is a single fixed administrative grace period, not a rolling deadline |
| Before any statistical-study commitment | Credible route to **six distinct services/repositories across at least three organizations**, dependency map and feasible adverse-event budget | Do not ratify or promise a cross-project paper. If this scope is unavailable, stop the current empirical-paper effort; no automatic research-question change |

Two screened repositories make a **data-only access/capture validation** worth completing; they do not make a general research paper adequate. Six/three is a coverage planning target, not proof of independence, power or publishability. A thousand reruns from one artifact/organization cannot meet it. A single-organization case-study alternative would need a separate explicit decision; do not pivot again here.

If access and data-only validation pass, a later precision review must set the needed number of independent projects and evaluation events. No number of qualifying repositories alone permits model experiments.

## Per-repository monthly accrual template

Use the same fixed six complete calendar months as screening. All present entries are UNKNOWN because there are zero cooperating repositories. Do not fill them with prior CI-pilot counts.

```text
candidate/service/org cluster:
screen window and source/retention coverage:
month:
  distinct real release/content-environment events:
  distinct protected approval requests:
  explicitly approved requests:
  approved started executions:
  terminal successes:
  execution failures (excluding timeouts):
  execution timeouts:
  explicit denials:
  pending/no recorded action at cutoff:
  admission timeouts:
  cancellations [pre-start / post-start]:
  superseded/skipped/approved-no-start/unknown outcomes:
  correlated retry groups/shared outage groups:
prospective valid-snapshot + label intersection: UNKNOWN until measured
capture outages / missing months / configuration changes:
```

Counts refer to explicit states and transitions, so an approved request later cancelled appears in both its approval transition and its cancellation outcome; do not force overlapping transitions to sum as exclusive bins. Publish mutually exclusive end-state counts separately. Define cancellation rate as requests with explicit cancellation divided by all requests in the window; additionally report post-start cancellation per started execution. Denial and no-action are different. Stable successes cannot compensate for absent failures/timeouts.

For each repository estimate mean and month-to-month range of eligible executions and nonoverlapping failures/timeouts. Report adverse events by month, not only a pooled rate that hides a single outage. Separate first attempts, retries and unique affected artifacts. Screening on at least one adverse case enriches this cohort; its rates do not estimate general GitHub failure prevalence.

## Calendar budget before promising a statistical study

Let `lambda_j` be observed usable adverse events/month in repository j: approved, started, naturally failed/timed out **and** validly captured with linked outcomes. Before passive capture this is unknown. A provisional scenario is historical adverse executions/month multiplied by a clearly labeled hypothesized capture fraction; it is not measured usable accrual. Later replace it with the observed joint count, since capture loss may correlate with failures.

For planning `months(K) = K / sum(lambda_j)` when the denominator is positive; otherwise estimate is undefined/unbounded, not zero months. Calculate separate scenarios for 20, 50 and 100 **evaluation** adverse events; historical counts without authentic feature snapshots cannot simply populate training/evaluation. Reserve development/validation and untouched evaluation intervals separately. A low-rate/low-capture scenario must fit the budget, not only an optimistic mean.

| Hypothetical pooled usable adverse events/month | Months to 20 | Months to 50 | Months to 100 |
|---:|---:|---:|---:|
| 1 | 20 | 50 | 100 |
| 5 | 4 | 10 | 20 |
| 10 | 2 | 5 | 10 |
| 20 | 1 | 2.5 | 5 |

These are arithmetic illustrations, **not forecasts or power guarantees**. As an operational spending screen, require a credible conservative route to about **100 usable evaluation adverse events within six months of evaluation capture**, with additional development data budgeted separately. If even this event-count screen cannot be met, stop the intended comparative empirical paper rather than collect indefinitely. Passing it still needs precision analysis for calibration, paired comparisons and clustering before ratification. An expected count is not assurance it will be reached.

For perspective, six repositories each producing 20 approved executions/month at 1% failures and 80% valid capture would yield only 0.96 usable failures/month, roughly **104 months** for 100. Thus the individual activity threshold is merely an entry screen. The most valuable candidate is an owner-verified high-throughput real release pipeline with sufficiently frequent consequential execution errors, complete provenance and passive capture rights—not a famous repository with many successful CI jobs.

During any later authorized data-only capture, perform a predeclared four-week accrual/coverage check. If observed joint capture or adverse-event accrual invalidates the six-month budget, pause and reassess using measurement evidence; do not generate fixture failures or change the target. When labels mature slowly, show pending cases and uncertainty rather than infer zero failures from immature data.

## Hard stop conditions

Stop the empirical-paper effort if: no two hard-screen passes by the bounded deadline; no credible independent cohort/access path; no verified consequential protected stage; approval ordering or outcome linkage cannot be measured; passive observation requires changing approvals; denied/no-action/missing cases cannot be distinguished; adverse-event accrual is unaffordable; or publication/governance terms preclude honest research. Refusal of intervention alone is not a stop condition for passive prediction. Mechanism-test success cannot override any data-access/accrual stop.

Current biggest risk: **even a willing operator may supply too few naturally failed, validly captured protected executions for a useful study.** Closely related risks are selected labels and snapshots missed before fast approvals. None is fixed by additional algorithm development. The value of the bounded access effort is to resolve these uncertainties cheaply enough to stop if necessary.

See the [sendable package and contact leads](recruitment-package.md), [screen](repository-screening-checklist.md), [owner questions](owner-verification-questionnaire.md), [passive plan](prospective-observation-plan.md) and [engineering-only plan](mechanism-validation-plan.md).

**Final recommendation: SEEK ACCESS.**
