# Disposable repository: mechanism-only test plan

**NOT AUTHORIZED; NOT RUN. Live enforcement remains UNVERIFIED.** The user has prohibited creating, modifying or using any repository for live enforcement testing. The latest request authorizes this written assessment only. No repository, environment, App, webhook, job, callback or deployment is created by this plan. `ci-design-v2` remains unratified.

A controlled disposable repository could provide useful **engineering feasibility evidence**, once separately authorized. It cannot establish that real projects have predictable release failures or that automatic admission improves outcomes. It is optional for initial observational prediction research and does not resolve cooperation, outcome volume or selective-label identification.

## Exact scope of a future authorization

An authorization would name the repository and owner, permitted environment and harmless job, allowed setup changes, selected gate mechanism, approved/rejected test cases, API permissions, spend/runtime limit, monitoring window and cleanup responsibility. Use no production credentials, customer data, real publishing destination or production side effects. The protected job writes a uniquely identified harmless marker to an isolated sink; a forced nonzero exit is a test input, never a naturally occurring research failure.

Owner setup should establish one pending gate scope per named protected job. Test a required-reviewer path or registered custom protection App path explicitly; success in one does not validate the other. Both permission and reviewer/App identity must be legitimate. GitHub describes the custom request/callback architecture, but it remains a configuration to verify locally, including its current preview status. [Custom-rule documentation](https://docs.github.com/en/actions/how-tos/deploy/configure-and-manage-deployments/create-custom-protection-rules).

The job has an independent first-execution witness and a separate protected-side-effect marker. An earlier unprotected observer may log timing but may not perform the protected side effect. Verify that an approval at run/environment scope cannot accidentally authorize additional jobs. Document any bypass capability without exercising a bypass outside the authorized cases.

## Instrumentation and event record

Each case records immutable repository/request/run/attempt/job/environment/content identity, intended test action, actual gate state, result, all missing fields and collector uptime. Preserve provider event timestamps separately from authenticated local receipt timestamps; record clock resolution/offset uncertainty and monotonic local elapsed times. Do not substitute provider timestamps for receipt.

Proposed timing sequence:

1. Existing gate becomes pending; archive the best available authoritative state evidence.
2. Receive authentic event; record HTTP ingress and durable persistence times separately, delivery ID and verification result.
3. Receive each required context source; compute a fixed deterministic feature summary; seal the snapshot and record `snapshot_ready` and `decision_ready`. No model is needed for this mechanism test.
4. Revalidate the exact request identity and pending state; send the authorized approval/denial; record send and response receipt separately from any server-side state change evidence.
5. Observe first protected execution and the independent side-effect marker, or record the complete bounded monitoring interval without execution.
6. Reconcile duplicate delivery, missing callbacks, terminal states and outcome records against the entire scheduled test-case ledger.

The protection barrier is the mechanism; a webhook is a notification. Delayed/out-of-order deliveries must be included. A receiver acknowledgment deadline is not a delivery-latency guarantee. [Webhook troubleshooting](https://docs.github.com/en/webhooks/testing-and-troubleshooting-webhooks/troubleshooting-webhooks), [webhook best practices](https://docs.github.com/en/webhooks/using-webhooks/best-practices-for-using-webhooks).

## Proposed test cases

| Case | Controlled input | Required evidence / failure condition |
|---|---|---|
| Held with no decision | Leave the configured rule unresolved for the authorized interval | No protected start/marker; still-pending state and uninterrupted monitoring. A timeout is a timeout, not a successful enforcement measurement |
| Valid approval | Approve current exact request after snapshot completion | Authorized scope advances only after all rules pass; job/marker bound to expected artifact. API 2xx alone is insufficient |
| Valid denial | Reject current exact request | Denied state plus no protected execution/marker for the complete prespecified interval; no inference of infinite safety |
| Duplicate delivery/callback | Replay the same allowed test event or callback | Idempotent behavior; no duplicate approval of another request |
| Stale request/rerun | Use an old request against a separately identified new attempt | No silent transfer of approval; wrong/stale identity is rejected or safely recognized by the controller |
| Changed artifact or environment | Present a mismatched context snapshot in the test harness | No callback that authorizes the wrong protected workload |
| Delayed/missing context | Withhold an optional source, then a mandatory identity source | Optional source remains missing; mandatory failure cannot create implicit approval; report readiness failure and delay |
| Collector/callback failure | Authorized timeout, process restart or simulated network failure | No accidental approval; reconcile lost/duplicate events. For a production-like integration, proposed fail-closed behavior requires owner agreement |
| Competing cancellation/supersession | Owner-authorized cancellation or newer request | Old result not applied to the new request; nonexecution administrative status kept distinct from a failed workload |
| Multiple rules/jobs and bypass | Inspect scope, then exercise only specifically authorized benign cases | Approval does not claim to release all rules; identify any broader scope/bypass rather than claiming universal enforcement |

Before running, prerecord case counts, operating conditions, timeout limits and monitoring coverage. Include failures, late events and missing traces; do not repeat until only favorable traces remain. This plan deliberately does not invent sample counts or latency thresholds unsupported by an application SLA. A minimal positive/negative trace validates behavior in those cases; reliability estimates would need a separate sample/operating-envelope plan.

## Quantities and interpretation

- `snapshot_latency`: actual receipt of the qualifying pending notification to complete snapshot; report individual source readiness and misses.
- `decision_processing_latency`: snapshot ready to deterministic decision ready; this is not model performance.
- `approval_round_trip`: local send to response receipt. It includes transport and is not identical to server release latency.
- `usable_margin = first_protected_execution - decision_ready`: report only for verified executions, with clock uncertainty. A lower uncertainty bound above zero is evidence of ordering for that case; job metadata alone may not certify first executable instruction.
- `release_to_execution`: compute only if server release time is genuinely known; otherwise report bounds or unknown, never substitute API response receipt.
- Denied cases: margin is undefined; report nonexecution during a fixed fully monitored horizon, gate outcome and later supersession separately.

API acknowledgment may arrive after the server permits execution. Local send-before-receipt ordering must not be used to invent a server timestamp. Deliberately holding a gate can produce a positive margin by construction; it demonstrates controlled timing, not that passive collection keeps up with real human approvals. Preserve all clock and monitoring limitations.

## What this could and could not prove

| Could demonstrate in the tested configuration | Could not demonstrate |
|---|---|
| An actual pending request reaches a collector; the scope can be resolved | That an uncooperative/public repository offers this access or complete historical records |
| A deterministic snapshot can be sealed while an intentionally held gate remains closed | That required real-world features arrive before normal approvals without changing workflow |
| Authorized approve/deny calls produce verified job/marker behavior | Reliable enforcement across untested plans, runner types, bypasses, outages or organizations |
| Empirical timings and failure traces for the bounded test conditions | A universal platform latency bound or production reliability guarantee |
| Identity binding, duplicate handling and failure-mode behavior | Predictive signal, failure prevalence, cost savings, reviewer benefit, deployment safety or causal effects |

Mechanism traces must live in a distinct engineering-test corpus and never enter research training/evaluation, failure-rate estimates, project counts or claims of real downstream effectiveness. The current [prospective evidence status](../timing/prospective-evidence-status.md) remains unchanged: there are no live traces or measured usable margins. The next authorized work, if any, should address [cooperating access](recruitment-protocol.md), not create this repository automatically.
