# Prospective CI data collection plan

Version `ci-design-v1`, 7 September 2026. **No collection or policy execution has been started by this task.** This plan replaces neither the old CSVs nor the manuscript. It defines a new data product whose feasibility must be demonstrated on development repositories first.

## Scope and information model

Collect pre-execution workflow-attempt snapshots, immutable change/configuration context, and later CI conclusions. Restrict primary events to `push` and `pull_request`, first attempts, and prospectively selected build/test workflows. Collect other attempts/statuses for coverage and the declared retry cohort. The user must ratify this target; it is not production deployment control. See [decision contract](decision-feedback-contract.md), [features](feature-specification.md), and [selection protocol](project-selection-protocol.md).

The original Flask/Requests exports remain development-only. Missing metadata may be recovered for exploratory description, but those exports cannot become untouched evaluation evidence. Their workflow identities and first-observation times cannot be invented from current repository state.

## Collection route and feasibility gate

Preferred route: participating repositories provide a read-only GitHub App/webhook installation plus raw event archival. Capture `workflow_run` events and trigger/commit metadata; reconcile using read-only REST attempt and job endpoints. This requires repository-owner cooperation and an authenticated event receiver; those resources are not established here. Do not send invitations, install an App, trigger a workflow, or change a repository without separate authorization.

Public REST polling is a fallback for development feasibility, not an automatic equivalent. Queued intervals may be shorter than the poll interval and API fetches may complete after execution starts. A collector that sees only completed runs cannot claim pre-execution snapshots. Historical REST responses can support backfill/diagnostics and future warm-up once actually observed, but cannot be relabeled as earlier observations. A trusted existing event archive is an alternative only after timestamp and completeness validation.

Run a **data-only** feasibility pilot on development repositories after access is approved: no learner, policy, action recommendation or ranking. Quantify pre-execution capture coverage, complete diff availability, correct workflow-ref resolution, terminal-event reconciliation, API load and missingness. If pre-execution coverage or meaningful change coverage fails, stop and revise the target or collection route before recruiting evaluation repositories. A decision at collector notification after CI starts would be a different early-monitoring protocol, not a silent fallback.

## Acquisition stages

1. Freeze the recruitment frame, ownership/fork families, workflow eligibility and project partitions before evaluating any learner. Assign provider repository IDs; retain rename history.
2. Append raw webhook bodies and API responses to content-addressed immutable storage. Record payload SHA256, delivery/request ID, receipt UTC, event/provider timestamp if present, endpoint and query, API version, pagination evidence, response status and access class. Preserve retries and transport failures. Never archive credentials or authorization headers in the scientific artifact.
3. On a pre-execution notification, create one sealed decision snapshot. Derive eligible features from sources already captured; missing sources remain explicit nulls. Do not wait for the label or cherry-pick the earliest successful API response.
4. Retrieve run-attempt-specific metadata and jobs for reconciliation, preserving all attempts. GitHub provides separate attempt endpoints; its list-run result can be filtered and paginated. The documented filtered-search limit requires time-window partitioning with boundary deduplication rather than assuming one request gives complete history. [Workflow-run API](https://docs.github.com/en/rest/actions/workflow-runs).
5. Resolve the exact event head/tested/base revisions. For push use tested revision versus its first parent; for PR use captured head versus its captured merge base, separately retaining tested merge SHA. Root or ambiguous/missing bases yield missing diff features. Fetch all changed-file pages or use a complete local Git object diff; record truncation/binary/rename status. The commit endpoint documents file limits and pagination, and the compare endpoint has a different files limit; do not assume a successful response is a complete diff. [Commit API](https://docs.github.com/en/rest/commits/commits).
6. Resolve workflow YAML and reusable dependencies to immutable refs for the actual event. Parse only; never execute workflow code or untrusted expressions. Jobs declared in YAML are not jobs actually executed; dynamic matrices, conditions and reusable workflows can prevent static expansion. Mark unresolved counts null rather than reading completed jobs as context. [Workflow syntax](https://docs.github.com/en/actions/reference/workflows-and-actions/workflow-syntax).
7. On terminal delivery, attach feedback to the exact attempt. Record first durable receipt as label availability. Attempt job records help audit timestamps but do not prove the run-level label was available at the last job's completion. [Workflow-job API](https://docs.github.com/en/rest/actions/workflow-jobs).
8. Deduplicate canonically, quarantine conflicts, validate source lineage and split membership, and emit quality reports. Hash all raw and normalized releases. Reconciliation may add missing records but cannot rewrite a sealed decision's feature values using later data.

Webhook handling must tolerate delayed, duplicate and out-of-order deliveries. The event documentation should be versioned with the collector, and the workflow trigger's `requested` behavior on reruns cannot be assumed identical to first runs. [Webhook payloads](https://docs.github.com/en/webhooks/webhook-events-and-payloads), [workflow events](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows).

## Required collection-quality reports

Report by repository, workflow, event, attempt and calendar week: frame counts, captured pre-execution snapshots, late/missed capture, timestamp uncertainty, payload completeness, missing terminal events, each terminal category, duplicates/conflicts, unresolved refs/diffs, feature null rates, unique values/quantiles and repeated-change rates. Coverage denominators include all eligible observed frame events, not just retained successes. Missing entire event intervals must be documented through independent API reconciliation; otherwise the denominator itself is unverified.

Provisional development pilot criteria: at least 90% of frame-eligible started first attempts captured before execution; at least 90% terminal-state reconciliation after follow-up; complete verified diffs for at least 80% of eligible snapshots; at least two meaningful change-feature families with genuine nonconstant support. These are **data-product feasibility thresholds**, not statistical sufficiency guarantees. Ratify them before the pilot and retain failed repositories in the recruitment/attrition report. Do not tune thresholds using held-out prediction results. Final research claims must describe the retained sampling/capture regime and its biases.

Data-derived feature-distribution quality must be established on development data; held-out quality monitoring should expose only prespecified aggregate diagnostics to a custodian, not feature-label associations or policy results. See the feature specification for the current measured baseline and future profiling requirements.

## Operational reproducibility and resource estimate

Use a read-only collector with pinned API version and schema revision, resumable checkpoints, stable page sizes, bounded retries honoring rate-limit responses, and explicit completeness markers. A first implementation should pin the API version documented and tested at implementation time (current reviewed examples use `2026-03-10`), rather than silently inheriting the legacy client's version. No API calls have been validated by this task against a live collection account.

Estimate storage/API load from the data-only pilot: raw event bytes per attempt, calls per run/attempt/diff, source-cache hit rate, and daily eligible volume. Scale by the locked recruitment frame and calendar horizon; do not invent a collection budget now. Retain license/access terms, lawful redistribution decisions, author pseudonymization method and provenance mapping. Raw data and identity mappings can have different access policies while reproducible deidentified derived releases retain checksums.

The proposed normalized SQLite schema is [dataset-schema.sql](design/dataset-schema.sql); its [schema guide](normalized-dataset-schema.md) defines joins and missing-value conventions. Store a source journal plus normalized tables, not another flattened CSV with overloaded SHA identity. A CSV export may be a derived view with stable IDs and an accompanying schema, never the source of truth.

## GO gates

Gate 1: target/decision/cost/label semantics ratified. Gate 2: acquisition route and immutable source provenance demonstrated in development. Gate 3: feature coverage and identity/duplicate invariants pass. Gate 4: independent project frame and development/validation/evaluation split locked. Gate 5: precision plan, meaningful margins and fixed calendar horizon approved. Gate 6: collector/extractor/model-runner integration audits and fresh-environment reproduction pass. Gate 7: signed-off protocol and source hashes registered before evaluation access.

**Current status: NO-GO for headline experiments.** The technical design is testable, but collection feasibility, meaningful-context availability, sampling precision and human scope decisions remain open. Documentation and local fixture tests are not evidence that these gates have passed.
