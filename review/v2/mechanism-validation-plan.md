# Smallest harmless mechanism validation

**ENGINEERING VALIDATION ONLY — NOT AUTHORIZED, NOT RUN.** This is a concrete reviewable plan, not a workflow installed in any repository. No creation, settings change, dispatch, approval, denial, cancellation or webhook installation is permitted until the user explicitly authorizes the named repository and actions. Live enforcement timing remains UNVERIFIED. Earlier [mechanism analysis](mechanism-test-plan.md) remains applicable.

## Required repository/settings

Propose a disposable **public** repository owned by the user, with no private code, secrets, releases, packages or production links. The actual name remains for future authorization; none is created. Use standard GitHub-hosted Linux runners. Create an environment `mechanism-only` **before** any run, with a required reviewer, branch restriction matching the test default branch, no environment secrets and no custom protection App. Verify available settings and administrator bypass behavior. An automatically created environment without configured protection is not a gate.

Prefer a distinct authorized trigger user and reviewer with self-review prevention enabled. A one-owner test can instead explicitly configure self-review as allowed in this disposable environment; that validates only that configuration and must not weaken any real repository policy. If the selected account/plan cannot provide required reviews, stop the test rather than replace the gate with a sleep. GitHub documents visibility/plan restrictions and required-reviewer settings. [Environment setup](https://docs.github.com/en/actions/how-tos/deploy/configure-and-manage-deployments/manage-environments).

Use exactly one protected job and environment per run, no matrix, reusable workflow, scheduled trigger, concurrency cancellation or automatic approval. Keep the receiver/observer outside the protected job. An existing owner-controlled HTTPS receiver is needed for webhook timing; do not purchase hosting or register a service implicitly. If none is authorized, webhook timing stays NOT TESTED even if REST observation works.

## Minimal proposed workflow — documentation only

After separate authorization, the following would be the sole workflow on the default branch. It checks no source out, installs no dependencies, uploads no artifacts, calls no cloud service and publishes nothing. Its protected operation is only a log marker; this is deliberately not a real release.

```yaml
name: Mechanism only
on:
  workflow_dispatch:
    inputs:
      deliberate_failure:
        description: Engineering fixture only
        type: boolean
        default: false
permissions: {}
jobs:
  upstream:
    runs-on: ubuntu-latest
    timeout-minutes: 2
    steps:
      - name: Harmless upstream marker
        run: printf 'UPSTREAM_COMPLETE\n'
  protected_marker:
    needs: upstream
    runs-on: ubuntu-latest
    timeout-minutes: 2
    environment: mechanism-only
    steps:
      - name: First protected marker
        env:
          FIXTURE_FAIL: ${{ inputs.deliberate_failure }}
        run: |
          printf 'PROTECTED_MARKER run=%s attempt=%s sha=%s\n' "$GITHUB_RUN_ID" "$GITHUB_RUN_ATTEMPT" "$GITHUB_SHA"
          date -u '+marker_utc=%Y-%m-%dT%H:%M:%SZ'
          if [ "$FIXTURE_FAIL" = 'true' ]; then exit 1; fi
```

The boolean is intentionally outcome-determining fixture data, **not a predictive feature**. Exclude it from the demonstration snapshot; that snapshot can contain the upstream result, run/attempt/SHA, workflow hash and a fixed synthetic change-summary object. None is empirical research context. The shell marker occurs after runner setup and shell launch: it is not certified first user-code execution. Provider job/step starts, marker time and local receipt are separate observations.

## Permissions and actor separation

| Operation | Necessary authority; proposed actor |
|---|---|
| Create repository / add workflow | User-authorized repository creation and owner write access. Owner uses UI/git; researchers need no broadly scoped creation token |
| Configure environment/rules | Owner/admin; environment API writes require `Administration: write` if used |
| Read run/job/pending/review records | Public reads where supported, otherwise selected-repository `Actions: read`; deployments/statuses need `Deployments: read` |
| Dispatch four cases, cancel the bounded unanswered case | Owner UI or selected-repository `Actions: write`; dispatch API requires this permission. [Workflow API](https://docs.github.com/en/rest/actions/workflows) |
| Approve/reject pending job | An eligible required reviewer; API route requires `Deployments: write`, with read access and reviewer eligibility. Scope is run/environment, not an arbitrary job ID. [Review API](https://docs.github.com/en/rest/actions/workflow-runs#review-pending-deployments-for-a-workflow-run) |
| Receive passive events | Owner-configured repository webhook for workflow/job events; webhook setup requires owner authority (`Webhooks: write` if using that API). Read-only App subscription is needed if testing App-only review events; do not enable a custom gate merely to receive notifications |
| Store evidence | Authorized local directory and receiver journal; tokens/webhook secret remain operational credentials, never evidence exports |

Required-reviewer API behavior is the target here. Success does not validate a custom protection App callback. Selected API versions, token type, scope and event subscriptions must be recorded; tokens are never printed. The workflow's `permissions: {}` does not provide the external observer/reviewer credentials.

## Four cases and exact actions

Proposed hard cap: four fresh dispatches, no reruns/retries automatically. One later extension to test rerun identity or custom rules needs separate authorization. Capture every attempt, including failed preparation.

| Case | Actions after authorization | PASS condition |
|---|---|---|
| 1 — hold then approve success | Dispatch success; observe pending gate, hold deliberately for 60 seconds after authentic pending receipt, seal deterministic snapshot, then eligible reviewer approves | No protected start while held; snapshot precedes reviewer approval-send; authenticated approval is linked to correct run/environment; upstream success and protected marker/success recorded |
| 2 — deny | Dispatch success; seal snapshot while pending; reviewer rejects; observe through terminal reconciliation and a five-minute post-rejection window | Rejected request recorded; no protected start/marker during complete observation; gate-induced workflow failure not encoded as execution failure |
| 3 — no action | Dispatch success; leave pending for five minutes after verified pending state; then owner cancels the test run for cleanup | Complete pending/no-action observation with no protected marker; subsequent cancellation remains a separate administrative state, not denial |
| 4 — approved execution failure | Dispatch deliberate-failure fixture; snapshot while pending; approve | Marker emitted, then execution failure recorded with exact attempt identity; parser distinguishes this from case 2's rejected/nonexecuted job |

If no pending observation arrives within five minutes of upstream completion, mark the case INCONCLUSIVE for timing, stop further dispatches and let the authorized owner cancel it. Do not approve a case whose identity/gate/snapshot is unresolved. Each denied/no-action nonexecution statement is bounded by recorded monitoring coverage; log retrieval failure is not proof that no marker existed.

## Evidence and PASS / FAIL

Record settings export/hash and reviewer mode; workflow revision; request/run/attempt/job/environment/SHA IDs; local dispatch-send/response times; provider timestamps; webhook delivery IDs, payload hashes and actual ingress/durable receipt; REST request/response clocks; upstream conclusion receipt; snapshot inputs/hash/cutoff; reviewer approval/rejection send/response; available server decision evidence; job/step start/completion; marker content/time and actual retrieval receipt; terminal outcome; uptime/outages and clock uncertainty. Reconcile the four-case ledger against API records. Record raw approval state separately from feature storage.

**PASS (bounded engineering behavior):** all four cases have valid identity joins and complete evidence; approve/deny/no-action behavior matches the table; the snapshot is sealed before the first authorized approval request is sent; no observed protected execution precedes admission; outcomes parse correctly. Report webhook receipt/order findings separately—REST success does not pass a missing webhook test.

**FAIL:** protected execution while the validated gate should be held or after rejection; an approval affects the wrong scope/attempt; stale/ambiguous identity accepted; approval information enters the snapshot; or rejected/unknown cases receive invented execution labels. Stop at a safety/scope failure; do not continue for favorable traces.

**INCONCLUSIVE / NOT TESTED:** missing clocks, uncertain order, receiver outage, unavailable event type, inaccessible logs or no authorized setup. These are not PASS. No global PASS if a required component is inconclusive.

For approved cases report `job_started_at - snapshot_ready` as a **provider-metadata margin**, plus `marker_time - snapshot_ready` with clock bounds. Claim a true first-execution margin only if an independently validated first-execution timestamp becomes available; otherwise it remains UNKNOWN. API acknowledgment time is not server release time. Duplicate/out-of-order events must be journaled without overwriting state. This tiny test checks observed identity consistency, not stability over untested reruns.

## Resources and limits of the result

Public repositories using standard hosted runners are currently free for Actions compute; private usage can consume included minutes and incur owner charges above quota. Larger runners are charged even for public repositories. Use no larger runner, cache, package, artifact upload or purchased receiver. The four cases configure at most eight jobs with two-minute execution timeouts (16 configured execution minutes); this is **not a guaranteed billing ceiling**, and pending waits/runner setup/rounding must be distinguished. Verify actual usage and hosting costs before authorization. [GitHub Actions billing](https://docs.github.com/en/billing/concepts/product-billing/github-actions).

A pass would demonstrate the tested gate, observer, snapshot and reviewer path in this disposable configuration. It would not demonstrate real-project feature readiness during unaltered approvals, natural outcome frequency, predictive performance, generalization, production safety, utility, universal enforcement reliability or causal benefit. Deliberate holds create timing room by design. Store fixtures in an engineering-only corpus, excluded from study rows, project counts and adverse-event accrual. Nothing has been executed.
