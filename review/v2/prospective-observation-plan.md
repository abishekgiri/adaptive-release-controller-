# Passive prospective observation and leakage contract

**Plan only; collection not started.** Applies to owner-approved, screened real repositories under the unchanged Path B question. No scores, advice or automatic decisions are shown; no gate, wait, reviewer, workflow, callback or release action is altered. Permissions and governance must precede observation. Initial capture validates data, not models. Live enforcement is not required for shadow observation and remains UNVERIFIED.

## Frame, identity and sources

Enumerate **every** eligible protected release request in the authorized window, including missed/fast approvals, denials, no-action, bypass, cancellations and missing outcomes. Keep capture-invalid requests in the coverage ledger. Accept metadata only from an owner-approved read-only collector or owner-forwarded existing event journal, plus bounded REST reconciliation. A new custom protection rule or an auto-approve callback is an intervention and forbidden in this phase.

One request is bound to repository/service/organization, immutable content, environment, run, attempt and protected logical job/version; later physical job IDs attach through a verified join. Do not invent a pending job ID if unavailable. Multiple requests for one artifact and jobs affected by one approval remain linked dependent units. A changed request or artifact requires a new snapshot. Apply the [screening procedure](repository-screening-checklist.md) first.

| Field group | Required observations and treatment |
|---|---|
| Identity | Pseudonymous repository/service/organization, workflow revision, logical/physical job, environment, admission request, run/attempt and artifact/content fingerprint; provenance of each join |
| Request/admission time | Provider request time **only if semantics are verified**, local first notification receipt, durable receipt and pending-state observations. Unavailable provider request time stays null; first receipt is not renamed request creation |
| Upstream CI | Exact content-bound check/job IDs, terminal conclusions, completion timestamps and actual receipt times for the designated upstream set; required dependencies must have finished before admission and been received before cutoff |
| Change context | Locally derived files/churn/category aggregates at verified immutable base/head; source receipt/version/truncation. No full source, author identity or mutable current PR metadata is required |
| Historical features | Counts/rates/duration summaries from prior job outcomes actually received before cutoff, with observation window, denominator, missingness and left-truncation; never future-resolved labels |
| Snapshot | Collector clock/uncertainty, immutable cutoff, required-source readiness, hash, extraction version, eligible/late/missing reasons; feature-only object |
| Approval | Actual owner/reviewer/server event time when verifiable, actual local receipt, source/clock bounds, scope, rule version and bypass status; isolated outcome/control ledger |
| Denial/no action | Explicit rejection time/source where available; pending/no recorded action as of a stated observation end; timeout/cancel/supersede/unknown as distinct transitions. Missing review history is not proof of no action |
| Protected execution | Actual job/attempt start and completion metadata, source receipts, execution witness if available, terminal conclusion and administrative category |
| Outcome | `1` for validated started execution failure/timeout, `0` success, null for denied/no-start/cancelled/superseded/skipped/missing/unknown; separately record reason and original raw conclusion |
| Coverage | Planned capture interval, all outages/delivery gaps, reconciliation passes, deletion/retention gaps, number of source requests, unresolved events and clock corrections |

## Immutable snapshot boundary

1. A capture coordinator observes the existing request while held and resolves immutable identity. Its control-state ledger is separate from the feature builder.
2. Assemble the prespecified available context promptly. Define `t_cut` as the actual snapshot seal time. Required inputs must have been received by `t_cut`; feature values describe knowledge at this instant, not at an earlier invented request time. Record request-to-snapshot delay and all cases missed during assembly.
3. Seal once per request under a deterministic rule, without conditioning on future approval or outcome. The snapshot schema excludes current approval/rejection, future wait duration, current release result and any post-cutoff field. No later backfill into this immutable object; corrections produce separately labeled versions excluded from primary pre-approval evidence.
4. Establish that the gate was still held at `t_cut` using the owner's genuine admission journal and clock bounds, or a later identity-consistent pending-state observation whose server observation must occur after `t_cut`. A webhook received while it reports an earlier pending state is insufficient. Validate there was no release/rehold/changed request. Where order cannot be proven, classify `PREAPPROVAL_ORDER_UNKNOWN`; do not accept merely because job start was later.
5. Approval/rejection events arriving during assembly trigger quarantine of the unfinished snapshot. Delayed events are reconciled afterward against actual decision times/bounds. Even if a collector had not yet heard of approval, a snapshot after actual approval is invalid. Equality within clock resolution is uncertain, not strict precedence.
6. Join future outcomes only into a separate label table. Outcome revisions never mutate features. Reviewers follow normal practice; no attempt is made to prolong the gate for complete collection.

For a verified time interval, require the **latest possible snapshot-seal time to precede the earliest possible approval time**, or a justified later-still-pending witness as above. Report the witness type. If only the time a human decision was received is known, it is an upper bound on earlier decision occurrence, not proof the snapshot preceded that decision.

## Explicit leakage checks for later collector validation

These are acceptance specifications, not tests executed in this phase.

| Check | Required invariant / response on failure |
|---|---|
| Feature schema allowlist | No current admission outcome, decision comments, final wait duration, current protected result or fixture outcome switch enters features |
| Availability | Each raw source and prior outcome used has authentic receipt no later than `t_cut`; unknown availability yields missing/invalid, not backdated availability |
| Upstream lineage/order | Designated CI is tied to actual released content and completed before the admission boundary; a unrelated green check does not pass |
| Historical overlap | A prior job that started earlier but finishes/is received after `t_cut` is not counted in failure history; archive overlap cases for deterministic validation |
| Snapshot immutability | Hash before label join equals hash afterward; appending future events cannot alter an existing feature object |
| Timing race | Include fast approvals, delayed/out-of-order notifications and clock ties in invalid/uncertain counts; do not select only favorable latency traces |
| Retry isolation | Original failure remains its label after retry success; new attempts have new request/snapshot binding and group lineage |
| Identity changes | Moved tag, changed artifact, environment or job version cannot silently reuse context or approval |
| Negative/control outcome handling | Gate failure, no-start, cancellation and missing result never become execution success/failure labels |
| Cross-request history | Later artifacts, approvals or incident knowledge cannot enter an earlier row; shared releases/retries are grouped for any later splits |

Store event and receipt times with source, precision and uncertainty. Maintain an append-only delivery journal with idempotent event processing and raw versus reconciled state; order events using verified semantics, not webhook arrival alone. Provider `updated_at` is not a default completion or approval timestamp. See [previous timestamp audit](../github-event-semantics.md).

## Follow-up and data-only acceptance

Owner-approved reconciliation should compare the complete request journal, review records and Actions jobs at least daily, plus after a reported collector outage. Polling/requests remain bounded to the enrolled repositories and named workflows. Follow each request through the normal owner's decision window and each started job through its configured execution timeout plus an agreed reconciliation lag. Fix these source-specific horizons before capture; five-minute toy-test windows are not transferred to real releases. End-of-study unresolved cases retain censoring/administrative reasons, not fabricated outcomes.

Report per project/configuration epoch: requests; valid preapproval snapshots; late/missing snapshots; approved/denied/unresolved; started/no-start; observed success/failure/timeout; cancelled/superseded/missing; and full intersection counts. Complete-case predictive evaluation would target the approved, started, label-observed, capture-valid subset. Use the [selective-label analysis](selective-label-analysis.md); passive recording does not reveal denied counterfactuals.

A future data-only validation passes only with demonstrable identity, timing and endpoint semantics and reconciled coverage acceptable under a subsequently reviewed precision plan. No percent completeness or model-ready status is assumed today. Collector permission alone is not successful prospective capture. Report unknowns and stop if capture requires intervention or time substitution.
