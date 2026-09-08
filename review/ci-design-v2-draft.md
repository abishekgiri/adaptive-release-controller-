# ci-design-v2 — proposed downstream release-admission study

**DRAFT ONLY — NOT RATIFIED, NOT IMPLEMENTED, NO MODEL OR LIVE TEST AUTHORIZATION.**
`ci-design-v1` remains frozen. Primary recommendation: PATH B, a different research question. Documented GitHub gating capability motivates this proposal; live observability, context readiness and enforcement are UNVERIFIED in this project.

## Precise proposed question

Can information available while a protected release job is awaiting admission—including already completed upstream CI—predict unsuccessful execution of that release stage among subsequently approved requests, and support a practically implementable admission workflow?

The first empirical estimand would be prospective prediction quality **within the observed approved-request population**. It would not be the value of a replacement admission policy, production incident reduction, or performance over counterfactual denied releases. Those require additional data and identification assumptions. This deliberately narrower target avoids relabeling upstream CI failure as downstream release benefit.

## Proposed unit and boundary

One instance is an immutable release-admission request keyed by `(repository_id, artifact_digest, target_environment, request_id, protected_job_id, run_id, attempt)`. The artifact must be bound to its verified build/commit and the exact gate request. Multiple requests for the same artifact remain correlated. A retry or changed artifact requires a new request; an approval is never transferable silently.

Boundary D: upstream build/test CI has executed, and a preconfigured protection rule is holding the downstream release job. The decision is **pre-release-stage execution**, not pre-CI or pre-all-computation. Every path to the declared release side effect must require admission. No unprotected parallel job may deploy the same artifact; bypasses and manual overrides are recorded and excluded from claims of automatic enforcement.

An environment-protected job has a documented approval barrier; the custom-rule route provides a request/callback mechanism. This creates a candidate decision point without depending on the 2–12-second scheduling gap. It still requires an authorized cooperating repository and an implementation test. [GitHub protection mechanism](https://docs.github.com/en/actions/how-tos/deploy/configure-and-manage-deployments/create-custom-protection-rules).

## Observable information and snapshot

Candidate inputs: immutable artifact/commit diffs; completed upstream CI/check outcomes tied to that artifact; verified prior release-stage results; test duration and relevant job configuration already observed; available environment/change metadata. Author-based features remain optional until identity and observation coverage are defensible.

Freeze the feature snapshot only after its required dependencies have arrived and while the gate is still held. Preserve per-source real receipt, source hashes, target identity and a monotonic local processing trace. Optional missing inputs remain null. Missing mandatory artifact/gate identity prevents an admissible decision; timeout stays held or records an explicit denial according to a reviewed operational rule. It never becomes implicit approval.

Exclude current release-job results, later health telemetry, future review actions and later revised artifacts. The fact that a previous CI result is known is legitimate context under this new target; it is not the outcome being predicted.

## Actions and outcomes

Proposed primary actions: `approve_release_stage` or `deny_release_stage`. Pending and human review are operational states, not automatically a third learning arm. An approval admits a defined job; it does not guarantee a successful deployment. Canary is excluded until a real staged rollout/exposure/rollback implementation exists.

Proposed primary target for an approved request: the protected release job's execution conclusion, with success=0, failure/timed_out=1. Gate rejection, admission timeout, supersession, cancellation and missing terminal observation remain distinct administrative/missing categories. A rejection-induced failed workflow is never labeled an executed release-stage failure. A stage may perform packaging/publishing/deployment operations; the actual protected operation must be fixed per cohort before approval of the study.

Service-health violations, incidents, rollout exposure and business consequences would require separate independently collected telemetry and a separately ratified horizon. They are not inferred from job conclusions. Neither the old CI cost matrix nor a new arbitrary stage-cost matrix establishes production utility.

Denying a request removes its natural stage-execution label. Delays may alter artifact/environment conditions. The full vector of real action losses therefore cannot generally be reconstructed. Historical prediction on approved requests is a selective-label problem. A later bandit study would additionally need observed chosen-action rewards, action support/logging and an identification strategy; no such experiment is authorized by this draft.

## Proposed timing proof, not executed

An owner-approved harmless test would record: receipt of a genuine pending/protection event; context-source receipts; snapshot completion; decision readiness; approval-send time; API response receipt; server-side release evidence where available; earliest protected job/start marker; and an independently observed first protected side effect. Record clock offsets/resolution and missing observations.

For approved cases compute `usable_margin = protected_execution_start - decision_ready_time`, with a conservative uncertainty interval. Also audit whether the gate was already active, remained closed during computation, and could be bypassed. Local API acknowledgment can arrive after server release, so do not equate acknowledgment ordering with server ordering.

For denied cases, protected execution should not occur during the declared observation period: report nonexecution evidence and monitoring completeness. The numerical margin is undefined, not infinite. Test delayed/missing delivery, no callback, duplicate/stale requests, reruns, supersession and manual bypass as distinct cases. Counting only successful approvals or positive margins would bias feasibility reporting. Deliberately holding a gate can create a positive margin by construction; that demonstrates a control path, not useful predictions or beneficial decisions.

The user has explicitly prohibited repository live enforcement testing in the current task. No environment, webhook, job, gate, callback or deployment has been created or exercised.

## Exact scientific changes requiring new review

| v1 assumption | Evidence against carrying it forward | Proposed v2 replacement | Scientific consequence |
|---|---|---|---|
| An already-created first CI attempt has a proven preexecution interval | 227/227 scoped run timestamps equal creation; no prospective instruction-level boundary demonstrated | A pending protected downstream release request | New population, unit and decision problem |
| Run start field is the execution barrier | Queued API examples can contain it; job and step starts occur later | An installed admission interlock plus independent execution evidence | No timestamp substitution or retroactive repair |
| Features precede all CI execution | Release gating occurs after upstream jobs | Completed upstream CI is explicitly allowed context | Old feature restrictions and leakage tests must be rewritten for the new target, not relaxed silently |
| Target is future CI failure | That conclusion is already known at the new gate | Future execution result of the protected release stage | Old CI labels cannot score the new predictions |
| Deploy/canary/block are three simulated actions | No matching three-way actuator was demonstrated | Approve/deny a named stage; review stays an operational state | New action space; no automatic canary claim |
| Label arrives regardless of action | Denial prevents target-stage execution | Selectively observed target label with explicit administrative states | Full-information replay no longer represents operated admission |
| All losses follow from one label and frozen matrix | Counterfactual denied execution is unobserved; operational utility unmeasured | Define observable cost components separately and acknowledge unidentified quantities | Old cost rankings and regret/benefit claims do not transfer |
| Seven contrasts, ten methods and old precision grid apply | Unit, target, feedback and available population change | New estimands, comparators, margins, splits and precision review | No automatic experiment specification or model preference |
| Existing data can evaluate the controller | Existing archives contain no validated gate decisions or stage outcomes linked to them | New owner-assisted prospective records; old three repos remain development only | No reuse as untouched evaluation evidence |
| A local design freeze approves execution | It only records integrity; current task forbids live testing | Explicit future ratification and separately authorized implementation | Draft remains inert |

## Reuse and stop conditions

Reusable: immutable source archival, receipt journals, attempt/repository identity discipline, null semantics, provenance validators, duplicate/overlap tests, descriptive audit tools and lessons from the corrected replay. Adaptation must be reviewed because the protected unit/feedback have changed.

Not reusable as scientific evidence: old headline comparisons, contextual-bandit superiority, the old label-to-cost matrix as real utility, pre-CI feature restrictions, old project-count justification, or retrospective positive timing gaps as intervention proof.

Do not commission models, a rewritten paper or a larger scrape now. Before investing in this pivot obtain a cooperating application owner, an explicitly authorized harmless protected-stage test, a verifiable target/telemetry source and a plausible independent-project recruitment plan. If these are unavailable, retain the current work as a bounded methodology/audit artifact instead of pursuing a deployment-effectiveness paper. This is a stopping rule for a proposed pivot, not a second primary recommendation.
