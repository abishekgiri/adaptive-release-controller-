# Decision and feedback contract

Version `ci-design-v1`, 7 September 2026. Status: **specified and hashed; not approved for headline evaluation**. This contract applies to a new corpus. The existing CSVs cannot be promoted into it by filling missing identifiers or availability times with guesses.

## Research target and unit

Predict whether an eligible GitHub Actions **workflow attempt** will conclude with `failure` or `timed_out`, before execution, using a captured pre-execution snapshot. The target is unsuccessful CI completion, not code defect, production incident, release success, or developer quality. The decision unit is the immutable tuple `(repository_id, run_id, run_attempt)`.

The **primary cohort contains attempt 1 only**, for `push` and `pull_request` events in preselected build/test workflows. All workflow attempts are retained in the normalized archive; reruns form a prespecified secondary cohort. More than one workflow for a commit produces more than one workflow-attempt instance, not multiple versions of one commit prediction. Costs are per eligible attempt. Report SHA-cluster diagnostics and never call these units independent commits or releases.

Previous runs of the same SHA may be available history. In the primary study, **training and outcome-derived history use first attempts only**, preserving the same target population; archival rerun labels are not silently mixed into its rates or model fitting. Report two strata defined at decision time: no earlier observed eligible binary label for the same tested change, and an earlier eligible label available. All earlier labels must satisfy identical availability and follow-up rules. The separate retry cohort can use all earlier attempts, with attempt number encoded and its own models/estimates. Do not infer the current attempt's label from a later run or replace a failed attempt with its successful rerun. A future first-change-only study would require a different cohort/version; it must not be silently mixed into this one.

## Required fields for each observation

| Field | Exact semantics | Prediction use |
|---|---|---|
| `repository_id`, slug, family ID | Stable provider numeric identity; slug retained for traceability; related projects grouped | ID for history grouping/splits, not arbitrary numeric input |
| `workflow_id`, workflow path and definition ref | Stable identity plus the exact version used by the trigger, if verified | Identity and verified static configuration permitted |
| `run_id`, `run_attempt` | Unique workflow run and explicit attempt index | Run ID only a key; attempt is a secondary-cohort feature, constant in primary |
| `head_sha`, `tested_sha`, `base_sha`, diff basis | Event head, actual tested revision and declared comparison base kept separate | Derived diff features only; SHA strings are not numerical risk features |
| branch/ref, event type | Values in the captured trigger/pre-execution payload | Branch class/event allowed; no later PR edits |
| `created_at` | Provider attempt/run creation metadata, with source | Timing check only |
| `decision_at` | Collector snapshot-commit time from a pre-execution state, recorded in UTC | Defines information barrier, not a tunable feature |
| `started_at` | Attempt-specific execution start later reconciled from authoritative attempt records | Quality check; current final duration never an input |
| `available_at` per source/feature | When the collector first durably possessed its source dependencies | Governs inclusion, not original commit author date |
| feature values and lineage | Versioned pure transforms of sources available by decision; explicit null/reason | Only the allowlist in the feature specification |
| terminal raw and normalized conclusion | Immutable first terminal observation for this exact attempt, with later conflicts retained separately | Feedback only |
| label-availability timestamp | Collector first durable receipt of the exact terminal conclusion | Feedback scheduling; never replaced by earlier provider timestamps |
| provider `completed_at`, if trustworthy | Completion-event time; nullable, with provenance | Diagnostic only; cannot advance label availability |
| cutoff and eligibility reason | Fixed cohort/follow-up rules, not chosen after observing model performance | Scoring/coverage only |
| scenario/matrix ID | Prespecified six-entry matrix version | Shared decision objective for all methods |

All timestamps are aware UTC. Preserve original timestamp strings separately. Unknown start/completion/identity fields remain unknown. Do not infer authentic observation times from a parsed date, duration arithmetic, a current API snapshot, a job's maximum completion time, or a mutable `updated_at` field.

## Snapshot and availability rules

At first captured `requested`, `queued`, `waiting` or `pending` state, seal a decision snapshot immediately. Do not delay the snapshot until an expensive diff request completes. Metadata/diffs already captured are usable; unavailable sources yield null features. Deterministic research extraction may happen later **only from those captured raw objects**, using the frozen extractor. This is as-of reconstructibility, not a claim of measured real-time model latency.

When actual start is known, require `created_at <= decision_at < started_at`. A snapshot captured after execution began is retained as `late_capture`, outside the primary pre-execution cohort. For cancellation before execution, preserve the instance with null start and nonbinary outcome; do not pretend it ran. Pending/no-start instances remain unresolved. A resolved binary outcome with no verified attempt start fails the primary data-quality gate. Report capture coverage against **all frame-eligible attempts**, including late/missed/never-started attempts, so this restriction cannot disappear from the results.

Current metadata captured in the decision event may have `available_at == decision_at`. Historical outcome dependencies must satisfy **available_at < decision_at**. Process equal-time decisions as a batch before equal-time feedback. A history aggregate records the maximum availability of its dependencies; an empty history is null risk plus count zero, with the collector's initialization/coverage record as provenance. No fabricated epsilon timestamp is permitted. These rules are stricter than the old CSV replay's equal-time convention and require a new corpus/version.

The stored decision view must never contain current conclusion, test pass count, actual job execution/results, final duration, logs, later incident/rollback data, or future aggregates. The outcome table is separate. A feature named `prior_duration` with current-attempt lineage is rejected even if its timestamp was incorrectly backdated. Immutable commit objects do not prove collector availability: both object identity and receipt evidence are required.

## Outcome and follow-up semantics

| Terminal state | Binary target | Scoring/learning |
|---|---:|---|
| `success` | 0 | One resolved label, full hypothetical loss vector |
| `failure`, `timed_out` | 1 | One resolved label; timeout is explicitly part of unsuccessful completion |
| `cancelled` | null | Competing termination; not success, not an observed failure |
| `skipped`, `neutral`, `action_required`, `stale`, `startup_failure`, unrecognized | null | Preserve raw state and counts; do not infer a binary label |
| no terminal observation within 30 days of decision | null at cutoff | Administratively unresolved; retained in denominator for coverage |

The 30-day window is a locked design default, requiring human ratification before collection/evaluation approval. Late labels remain in the raw archive but are excluded from both primary learning/history and primary scoring. The evaluation stream has a fixed calendar end, followed by the same 30-day follow-up. No policy receives a different censoring mask or earlier observation. A cancellation is not ordinary independent right censoring; report categories separately. Primary cost and probability metrics are explicitly **conditional on resolved binary outcomes**. Coverage and worst-case bounded-loss sensitivity are required; no missing outcome is assigned zero operational loss.

## Action, cost, and feedback contract

Actions are three simulated choices `{deploy, canary, block}` and have no effect on CI execution, observation, later contexts or data collection. They are labels for hypothetical decisions, not operations performed by the collector. Default loss rows, in success/failure order, are deploy `(0,10)`, canary `(1,4)`, block `(2,0.5)`. The scenario registry lives in `design/evaluation-spec.json`. Every observation links to each declared scenario through `decision_scenarios`; scenarios do not create new independent observations.

At label availability supply every learner the same packet: attempt key, original frozen features, binary label, all three computed action losses, matrix ID and availability time. Selected-arm LinUCB/Thompson may deliberately use only the selected loss; this is an algorithmic restriction. Even hiding the label does not create partial feedback when the known chosen-action cost reveals it. For unresolved outcomes, no binary training label or action loss vector is supplied.

The framing is delayed full-information supervised prediction and minimum expected-cost decisions. The action-dependent synthetic drift simulator is outside this contract. No production savings or causal effects are identified.

## Identity, duplication and corrections

- Same `(repository_id,run_id,attempt)` and same canonical decision: one instance, however many export/webhook deliveries occur.
- Same key, conflicting head/workflow/decision snapshot: quarantine and retain both source objects; never “keep last.”
- Same key, same terminal conclusion and completion semantics: preserve earliest authenticated receipt time; later redelivery cannot delay learning.
- Same key, conflicting terminal conclusion/completion: quarantine until independent adjudication; record a new data version if released data change.
- Same run, different attempts: distinct observations. Do not let the latest run endpoint erase earlier attempts.
- Same SHA, different workflows/runs/events: distinct workflow-attempt instances, correlated within a change. Never majority-vote statuses into a commit label.
- Same author name does not establish same author. Use immutable provider identity or a documented pseudonymous mapping; keep trigger actor separate from commit author.

Reference implementation: `data/research_contract.py`. Its validators enforce declared timestamps/keys and packet separation, not authenticity of a GitHub payload. The SQL schema enforces structural relationships; adapter-to-source and lineage closure checks remain mandatory integration work before GO. See `automated-invariants.md` for exactly what has and has not been tested.
