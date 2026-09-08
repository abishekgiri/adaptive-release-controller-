# Repository screening: reproducible admission procedure

**Preparation only; no repository screened or admitted in this phase.** Current confirmed count: zero. This procedure operationalizes the latest access-validation request; it neither ratifies v2 nor changes the research question.

The latest instruction requires **each qualifying repository to have some historical execution failures/timeouts**. This replaces the earlier feasibility proposal to permit all-success repositories in the primary cohort. Preserve the previous report; explicitly record this new recruitment selection. Findings apply to an outcome-enriched volunteer cohort, not all GitHub repositories or an unbiased population failure rate. Keep authorized aggregate screening counts for excluded all-success candidates; never manufacture failures.

## Qualification levels

All hard requirements below must be verified before ACCEPTABLE. Unknown is **PENDING, not PASS**; at the recruitment deadline unresolved candidates do not count as qualifying. A candidate can be useful for semantics discussion without being admitted.

| Criterion | IDEAL | ACCEPTABLE: minimum for access-validation admission | REJECT for this cohort |
|---|---|---|---|
| Platform and operation | Active real service, GitHub Actions, protected deployment stage | Real package publication/delivery/deployment through Actions, with owner-verified consequence; separate operation strata | Toy, tutorial, build/test-only, or no identifiable downstream operation |
| Approval | Existing gate with well-recorded normal reviewer and little bypass | Existing required approval, verified configuration and at least one authentic historical pending-to-approved trace; overrides identifiable | Environment name/YAML alone, only a wait timer/branch restriction, gate added solely for study, or no evidence it holds work |
| Protected job | One stable job per run/environment scope with immutable artifact identity | Stable logical job key and version map; exact request/attempt/job/content join | Ambiguous job bundle or moving reference with unresolved actual content |
| Upstream ordering | All designated CI dependencies terminal before the release request is held | Owner-validated dependency ordering and record evidence; mandatory upstream results received before snapshot | “Upstream CI” is still running at snapshot or ordered only by an assumed timestamp |
| Historical activity | Frequent releases, six complete monthly windows | At least 20 approved started eligible executions/month on average in a fixed six-month screen, plus complete monthly request counts | Below activity floor, unless a new resource/precision review explicitly changes this planning criterion before enrollment; no silent exception |
| Natural adverse events | Several documented execution failures/timeouts across months, meaningful causes | At least one verified failure or **execution** timeout in that window, with no gate rejection or synthetic case counted | Zero verified adverse executions; only cancelled/rejected or manufactured failures |
| Outcome observability | Consistent operation-specific labels, known missingness and failure subtypes | Job success/failure/timeout semantics validated; all missing/cancelled/no-start cases enumerated | Outcome unavailable, systematically masked or gate failure confused with workload failure |
| Timestamp provenance | Owner event journal, real receipts and clock uncertainty | Interpretable provider fields plus feasible passive receipt/approval-order proof; no synthetic reconstruction | Only current metadata used as historic observation, or no way to prove pre-approval capture |
| Reruns | Stable attempt identities, artifact lineage, no ambiguous approval reuse | Retries/reruns recognizable and linked; each admission revalidated | Run-only key conflates attempts or success overwrites prior failure |
| Ownership and passive access | Gate owner plus release operator, existing event archive, owner-run read-only collector | Owner verifies semantics and explicitly permits prospective passive observation under agreed governance | No willing owner, no authority, or observation requires changing decisions |
| Denominator | Every request reconciles with approval/no-action/start/outcome state | Complete request frame with explicit unknowns and review history validation | Denied/no-action cases silently disappear or missing review means presumed approval/denial |

The 20/month floor is a **resource screen**, not a claim of statistical adequacy. One adverse event establishes endpoint availability, not useful precision. IDEAL still requires cohort-level event accrual and independent service/owner coverage. See [recruitment stop rule](recruitment-stop-rule.md).

## Reproducible procedure

1. **Register scope before inspection.** Assign candidate ID; record referral source, screening version/date, named owner role, permission scope, intended operation and GitHub product/version. Fix a six-month UTC interval ending at the latest complete month. No access is implied by nomination.
2. **Receive authorized evidence.** Prefer an owner-generated sanitized package: workflow revisions, environment/rule settings and dates, gate/review journal, all request counts/IDs in scope, Actions attempt/job metadata, artifact joins and terminal outcomes. Preserve source identifiers, hashes, extraction query/version, pagination/retention limits, actual receipts and missing files. YAML is a configuration clue, not operational proof.
3. **Validate the gate with the owner.** Compare configuration to an actual pending-to-approved request, the protected job's start, rule/bypass history and owner's explanation of who authorizes what. Confirm the API's run/environment scope controls the declared job. No live action is performed for screening. Review history and current pending APIs can help, but neither proves historical completeness alone. [GitHub review/pending API](https://docs.github.com/en/rest/actions/workflow-runs).
4. **Validate the endpoint.** Owner walks through one success and each available adverse/administrative category in the fixed window. Independently inspect a deterministic sample: earliest and latest approved success; all adverse cases if at most ten, otherwise ten selected by a declared hash ordering of stable request IDs; and up to five per denied/no-action/cancelled category. This sample validates semantics, not prevalence; all-window counts supply prevalence within the screened frame. Missing categories stay “not observed,” not fabricated. Resolve disagreements before admission.
5. **Reconcile identities/order.** Join request → gate scope → logical/physical job → exact attempt → released content. Check designated upstream completion before admission, approval before protected start, terminal label after execution, and source receipt chronology. Separate genuine timestamps from brackets/proxies; uncertain ties do not pass a strict ordering claim. Record retries and whether approval could cover several jobs.
6. **Reconcile the denominator.** Produce a state-transition table for every request, including denied, unresolved/no-action as of extraction, no-start, cancelled, skipped, superseded and deleted/missing. Approval is not inferred from a job name; absence of a record is not no action. Record configuration epochs and incomplete historical windows.
7. **Assess passive route.** Owner confirms allowed event/metadata reads, capture location, retention, timing evidence and reconciliation access. Permission and architecture qualify the access pilot; actual prospective feature capture remains UNVERIFIED until a separately authorized data-only trace passes [capture checks](prospective-observation-plan.md). Never label paper-ready data solely from a checklist.
8. **Calculate event accrual.** Use counts of actual approved, started, validly resolved executions and natural failures/timeouts, not whole-workflow totals. Separate correlated retries and shared outages. Fill the monthly template in the stop-rule document.
9. **Record and countersign.** Assign IDEAL / ACCEPTABLE / REJECT or PENDING; every hard row has a PASS/FAIL/UNKNOWN, evidence pointer, reviewer, owner confirmation and unresolved issue. Store the decision and reasons before models, preserve rejected candidates' permitted aggregate log, and version any later reassessment.

## Screening record template

```text
candidate_id / owner_role / permission_reference:
screen_version / fixed_window_start_utc / fixed_window_end_utc:
service_cluster / organization_cluster / shared_release_infrastructure:
operation / logical_job_key / environment_scope / workflow_revision:
source_manifest_hash / extraction_version / retention_gaps:
criterion -> PASS|FAIL|UNKNOWN / evidence_id / owner_confirmation / reason:
request_approval_execution_monthly_counts:
adverse_types / retries / correlated_incidents / unresolved_outcomes:
passive_access_reference / capture_validation_status:
overall_level / reviewer / owner_semantic_signoff / decision_date:
```

No sign-off field is prefilled. Repository access qualification is permission for a bounded data-only validation phase **only after authorization**, not study ratification, statistical adequacy or demonstrated enforcement.
