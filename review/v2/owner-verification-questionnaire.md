# Owner verification questionnaire

**Not sent.** Start with a 20–30 minute discussion and aggregate counts. Do not request credentials or identifiable records in the initial reply. Answer “unknown” where necessary; an unknown hard requirement stays pending. “Owner” means a person authorized to attest to the release process, not necessarily someone authorized to sign a data agreement.

## Minimum questions before inclusion

| # | Question to owner | Evidence needed before qualification |
|---|---|---|
| 1 | What does the protected job actually publish, deliver or deploy, and what happens operationally when it fails? | Sanitized process description and actual side-effect/destination type; confirm this is more than ordinary CI |
| 2 | Which GitHub Actions workflow, logical job and environment are involved? What immutable content does one approval cover? | Workflow revision, environment ID, artifact/content fingerprint and request/run/attempt/job mapping |
| 3 | Which existing rule requires approval, who normally approves, and can approval be bypassed, reused or applied to multiple jobs? | Settings with dates, reviewer role, self-review/bypass policy, scope and alternative release paths |
| 4 | Can you verify a real historical pending → approved → started trace, rather than only YAML? | Owner-attested operational record and relevant timing sources; no test action requested |
| 5 | Which upstream CI checks must have finished before this stage awaits approval? Are their results tied to the exact released content? | Dependency/process verification and representative completed-result joins; identify optional/still-running checks |
| 6 | Over the last six complete months, how many release events, approval requests, approved executions, failures, execution timeouts and cancellations occurred each month? | Aggregate counts; separate denied, no-action, no-start and unknown cases; identify missing months |
| 7 | What do job success, failure and timeout mean here? Are failures masked or remote deployments asynchronous? | Examples/categories of real success and natural failure/timeout; distinguish setup error, delivery error, gate rejection and cancellation |
| 8 | How are denials, expired/unanswered requests, cancellations and bypasses recorded? Can you enumerate all requests? | Review/controller journal, current/past state coverage, retention limits; do not infer no action from missing metadata |
| 9 | What generates request/approval/start/completion timestamps? Do you retain actual event receipts and know clock uncertainty? | Timestamp dictionary; identify mutable update fields and proxies; explain how pre-approval snapshot order could be proven |
| 10 | How are reruns, retries, superseded artifacts and multiple environments identified? | Stable attempt IDs and lineage; confirm failed attempts remain visible after successful retries |
| 11 | Would you permit a passive collector or owner-operated export that leaves approvals unchanged and hides all scores? | Named permission authority, allowed sources, hosting/retention limits, realistic setup burden and an explicit prospective route |
| 12 | Who must approve data/ethics/security terms, and may honest aggregate null/adverse results be published? | Owner/data-controller/institutional contacts by role; approved minimization and disclosure process before collection |

## Follow-ups only after a plausible fit

- Which workflow/configuration changes occurred during the screening window? Could current settings misrepresent prior protection?
- Are releases associated with a service shared across repositories or with retries of the same artifact? Which teams/organizations share infrastructure or incidents?
- Can change counts, category flags and prior outcome summaries be computed inside your environment without exporting source, identities or comments?
- Does the deployment controller expose terminal state independently of the workflow, and what evidence supports semantic agreement? Health/rollback telemetry is optional and not required to replace the fixed primary target.
- How long do approvals normally take? Could the collector miss fast decisions? Can existing journals confirm this without adding a hold?
- What would an advance warning help someone do, if it were accurate? What false-alert burden would make it useless? This checks relevance, not a promise to change decisions.
- Are emergency releases, no-action requests or deleted logs systematically different? Can their numbers be reported without disclosing sensitive details?
- Is a historical contemporaneous feature archive available? If not, historical records can screen outcomes, but cannot automatically become leakage-free prediction training data.

## Attestation to request after authorized evidence review

```text
Candidate ID:
Owner role / authority scope:
Protected operation and admission scope verified:
Workflow/gate versions and applicable dates:
Evidence IDs for gate, target, timing and retries:
Known unknowns / missing records / exceptions:
Passive observation permitted under agreement reference:
Data-sharing/ethics approval reference (or pending):
Owner confirmation date:
Research screener / outcome of checklist:
```

Confirmation must identify remaining uncertainty; it is not a warranty of platform behavior, statistical independence or legal compliance. Do not seek a blanket assertion that all timestamps are reliable. Resolve each critical meaning against a source and the owner's operating process. See [screening procedure](repository-screening-checklist.md) and [governance notes](data-governance-notes.md).
