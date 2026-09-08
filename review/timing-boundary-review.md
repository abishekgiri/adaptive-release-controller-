# Timing-boundary rescue and pivot review

**Primary recommendation: PATH B — RESEARCH QUESTION MUST PIVOT.**

A documented protected downstream-job boundary supports a different release-admission problem. It does not establish the original pre-CI controller. **Live enforcement is UNVERIFIED**, and neither an empirical GO nor a ratification of `ci-design-v2` follows from this recommendation.

The user explicitly prohibited live repository enforcement testing and confirmed no suitable disposable gated repository is available. No repository was created, changed, dispatched, canceled or used for a live enforcement test. No additional repository data, policy/model experiments, headline changes or manuscript rewrite were performed. The v1 freeze remains unchanged.

## What the evidence establishes

1. **The old field mapping is not a verified execution boundary.** All 227 scoped first attempts have `run_started_at == created_at`. More decisively, GitHub's own run-list example includes this equality while the run is queued. The provider field cannot be promoted to a first-user-instruction timestamp from its name. [GitHub run API](https://docs.github.com/en/rest/actions/workflow-runs#list-workflow-runs-for-a-repository).
2. **Later job starts establish internal ordering, not an externally usable window.** Existing API records allow reconstruction of job and reported step timing for 49 of 60 sampled changes. They do not contain contemporaneous webhook receipt, complete prospective context, a real decision, or an enforcement trace.
3. **Notifications do not hold the scheduler.** GitHub documents delayed and out-of-order webhook delivery. A queued/requested event can arrive after the relevant work has begun. There is no documented event-to-controller deadline that makes the observed scheduling gap a reliable control mechanism. [Webhook troubleshooting](https://docs.github.com/en/webhooks/testing-and-troubleshooting-webhooks/troubleshooting-webhooks).
4. **A protected downstream job is a different, legitimate boundary.** A configured environment rule can hold that job until its conditions pass. Earlier CI may already have executed. This is the basis for a proposed release-stage admission study, not a repaired pre-CI experiment. [GitHub deployment controls](https://docs.github.com/en/actions/how-tos/deploy/configure-and-manage-deployments/control-deployments).

The [event-semantics inventory](github-event-semantics.md) distinguishes field-level documentation from unresolved instruction/scheduler semantics. It covers run, job, check-suite, check-run, webhook delivery, state transitions, protection and runner-hook signals.

## Reconstruction of the 2–12-second gap

Reconstruction uses the original content-addressed archive only. Full per-case timelines and source hashes are in [reconstructed-timelines.csv](timing/evidence/reconstructed-timelines.csv); [summary JSON](timing/evidence/timing-summary.json) provides reproducible distributions. All 923 sampled job objects contain `created_at`, even though the published job API examples reviewed do not expose a precise scheduler definition for that field.

| Development repository | Sampled changes with job timing / all sampled | Run timestamp → earliest job start, median | Range |
|---|---:|---:|---:|
| pallets/flask | 20 / 20 | 2 s | 2–3 s |
| psf/requests | 19 / 20 | 3 s | 2–3 s |
| expressjs/express | 10 / 20 | 3 s | 2–12 s |

The interval is not proved to be pure queueing: record creation, scheduling, allocation or preparation may contribute, and the archive does not instrument those phases. These are 49 correlated, retrospectively selected workflow timelines across three development projects. Missing job timing in 11 samples is retained, not discarded from coverage reporting. The CSV retains workflow creation, earliest recorded job creation, first recorded job start, first reported non-setup step, actual first API receipt, and the latest receipt in the retrieved source bundle.

The first reported non-setup step is explicitly a **proxy**: it is not a verified first user-controlled executable instruction. Selected third-party actions, setup, containers, earlier parallel jobs and runner hooks must be accounted for. We did not substitute this proxy or job start for the frozen boundary.

Actual local API request-to-durable-receipt time, over the archived 240 requests:

| Quantity | Seconds |
|---|---:|
| Minimum | 0.364 |
| Median | 0.518 |
| p95 | 2.671 |
| Maximum | 3.165 |

This is a measured transport/process/persistence interval. It excludes the wait before a poll, webhook delivery, required multi-source feature acquisition, decision computation, server-side admission processing and enforcement. Subtracting its median from a median job gap would not estimate intervention feasibility. Timing tails, per-case joins, clock uncertainty and missing cases matter.

All 49 computable `job_start - first_API_receipt` values are negative because the sample was fetched retrospectively. These are optimistic receipt-based bounds for **that archived collection**, not usable prospective margins or evidence that a future prospective controller is impossible. The actual usable-margin distribution is **n=0, all statistics null**: there is no recorded decision-ready time.

## The four boundary choices

| Boundary | A scientifically honest description | Practical decision | Current status |
|---|---|---|---|
| A | Before an explicitly controlled workflow is invoked | Whether to spend on CI or require review first | Possible architecture; different invocation-request unit and selective labels. Not demonstrated here. |
| B | After workflow existence, before any job/workload executes | Admit or hold all CI work | Needs an all-path interlock, not queued-state observation. No demonstrated preexecution controller. |
| C | Before a particular job after any earlier computation | Execute an expensive test/build stage | A job-conditional task; earlier CI may be valid context. Not the original all-CI claim. |
| D | After upstream CI, before the protected release stage | Approve or deny a specific release job | Documented protection mechanism exists; live operation/context readiness remain UNVERIFIED. Proposed pivot. |

The [enforcement review](enforcement-mechanisms.md) covers environments, custom protection, reviews, wait timers, required checks, branch rules, job conditions, dispatch, reusable workflows, cancellation, downstream workflows, external controllers and self-hosted hooks. A timestamp alone is insufficient for every one of these mechanisms.

## Why Path A is not established

Path A requires an authoritative, reproducible, observable **and enforceable** preexecution boundary for the original decision. The evidence supplies neither the complete observation-to-enforcement trace nor an implementation that controls the original protected work. Lack of prospective enforcement alone prevents establishing Path A. The equality of two provider fields is additional evidence against the current mapping, not proof that all possible preexecution prediction tasks are impossible.

Replacing run start with first job start would change an unvalidated proxy. Calling a post-CI release decision “preexecution” without naming the protected stage would misstate the information set. Canceling an already queued/running CI attempt changes its observation process; it does not preserve v1's assumption that the natural CI outcome arrives regardless of action.

No timestamp relaxation, synthetic receipt, shifted creation time, selected positive gap or known-label reconstruction is proposed as a rescue.

## Why Path B is justified at the design level

There is affirmative authoritative evidence for a job-admission interlock, so the review need not claim that no legitimate control point exists anywhere. The specific proposed pivot is:

> Predict unsuccessful execution of a protected release stage using information sealed while its admission request is held, including completed upstream CI; initially study prediction only within the subsequently approved-request population.

The actuator is approve/deny of a named job bound to an immutable artifact and environment. Canary is not included without a real rollout mechanism. A deployment job failure is not automatically a production incident, and useful prediction is not automatically useful admission control. These restrictions make the question precise enough to review without manufacturing utility evidence.

This is a conditional direction for a new study, not a claim that its required data are currently obtainable. The [v2 draft](ci-design-v2-draft.md) contains the exact unit, target, information set, action/feedback definitions, timing proof obligations, stop conditions and v1→evidence→v2→scientific-consequence diff. It is not implemented or ratified.

## Feedback consequences

Under v1's untouched CI replay, a matured label supports all declared hypothetical action losses: delayed full-information prediction remains the correct formulation for that replay. At a real downstream gate, denying the stage removes its natural execution label. Upstream CI is already known and cannot stand in for the denied stage's counterfactual result. A gate-induced failed workflow is not evidence that the release workload would fail.

The appropriate initial methodology for the proposed new data is selective-label, delayed supervised risk prediction with explicit decision-theoretic assumptions. Bandits become a possible later formulation only if actual actions, chosen-action rewards, observation rules and adequate action support are established. Operational deployment value is not identified from an observational CI archive. The [feedback reassessment](feedback-contract-reassessment.md) explains these distinctions for all boundaries.

## Micro-pilot status and spending decision

The semantic candidate survives as a documented architecture, but the user barred the live test. Therefore it was **not run**, and no event-receipt/feature-ready/decision-ready/enforcement/protected-start latency trace exists. The [prospective evidence status](timing/prospective-evidence-status.md) records this directly. An unavailable margin is not zero; an unexecuted denied job would not have an infinite measured margin.

Spend further effort only on a bounded application/data-access decision: a cooperating owner, the exact protected operation, a legitimate future harmless test and a target-outcome source. The draft specifies what that would need; it does not authorize any of it. Do not fund more retrospective scraping, policy seeds, tuning or manuscript revision now. If this access cannot be obtained, preserve the completed work as a reproducible methodology/audit case study or software artifact, with a publication claim contingent on separate novelty evaluation.

## Required closing answers

- **Is a real prospective decision point demonstrated?** GitHub documents a protected downstream-job admission point; an end-to-end controller has not been demonstrated in this project.
- **Is it observable?** A configured protection request can be observed prospectively in principle. Actual receipt and reliable coverage here are UNVERIFIED.
- **Is it enforceable?** The platform documents a holding mechanism. Live configuration, bypass resistance and correct controller enforcement here are **UNVERIFIED**.
- **Is the available context ready before it?** Not demonstrated. A held job could permit acquisition before approval, but no complete real snapshot/latency trace exists.
- **What feedback structure results?** Full-information for unchanged shadow CI replay; selectively observed, action-dependent target feedback for operated release admission.
- **Are bandits appropriate?** Not established. They are optional future candidates under a real partial-feedback reward contract, not the default framing.
- **Does the original research question survive?** Not as the specified pre-CI release-control experiment. Generic CI prediction remains a separate possible question.
- **PATH A, B or C?** **PATH B — RESEARCH QUESTION MUST PIVOT.** This is a design recommendation, not empirical GO.
- **Should we spend more time?** Only to establish a cooperating application and valid observation/control contract. Otherwise stop this empirical development route and retain the audit/artifact; do not attempt a numerical rescue.
