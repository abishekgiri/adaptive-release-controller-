# Data governance and institutional review questions

**Preparation only; no organization/user records collected in this phase.** These notes identify matters to check with the relevant institution and data owner; they are not legal conclusions, an exemption determination or a statement that a particular law applies. Institution, sponsor, jurisdictions and participant organizations are currently unspecified.

Before research records or interviews are collected, ask the institutional ethics/IRB office or designated reviewer whether the activity studies identifiable individuals, organizational systems or both; whether it is human-participant research, exempt research or outside its remit; and what consent/notice or waiver process, if any, applies. Approval decisions and operational failures can expose individual behavior even when the primary unit is a job. Merely hashing names does not settle the question. U.S. OHRP guidance discusses identifiable private information and coded secondary data and recommends institutional determination; it does not decide this project's status. [OHRP guidance](https://www.hhs.gov/ohrp/coded-private-information-or-biospecimens-used-research.html).

## Questions to settle before collection

| Topic | Decision needed from responsible institution/owner |
|---|---|
| Research/ethics scope | Does collection or an owner interview require ethics review, consent/notice, exemption determination or another documented process? Who is authorized to decide? |
| Organizational permission | Who may authorize metadata export, passive event forwarding and eventual publication? Does maintainer approval cover employees, contractors and organization policy? |
| Private/proprietary information | Which workflow/change/failure details can leave the organization? Can approved aggregates or in-environment analysis meet the purpose? |
| Employee/participant concerns | Could data be used to rank developers/reviewers or reveal working patterns? Exclude this purpose and individual identity features; consult applicable workplace/participant processes |
| Data roles and location | Who controls data, operates collection and accesses it? What hosting, jurisdiction, cross-border transfer or contractual restrictions must be reviewed? |
| Security and incident handling | Who approves the collector, credentials, storage, authentication, least privilege, access logging and breach response? |
| Retention/withdrawal | How long are raw records, link keys and derived data retained? What can be deleted on withdrawal, and what cannot be recalled after aggregate publication? |
| Publication/reproducibility | May honest negative findings and approved aggregate artifacts be released? Which records require controlled access? No favorable-results-only agreement |
| Scope changes | Who must approve score display, additional telemetry, identities, model study or intervention? Initial permission does not cover these automatically |

## Minimization by field

| Collect if needed | Minimize before transfer | Do not collect for this study |
|---|---|---|
| Stable request/job/attempt/content joins | Owner-held keyed pseudonyms; separate linkage key; preserve retry/service grouping | Access tokens, webhook secrets, cloud/deployment credentials |
| Workflow/gate version and operation semantics | Sanitized topology, role labels, hashes and process attestation; redacted snippets only if necessary | Full proprietary source, full diffs, private comments/discussion text |
| Precise chronology in controlled environment | Per-source receipt/clock bounds; publication uses aggregate distributions or carefully justified consistent time shifts | Public per-person work schedules or approval histories |
| Change/history aggregates and upstream results | Counts, category flags, truncation and provenance; local extraction | Developer emails, names, avatars, IP addresses or individual performance rankings |
| Execution outcomes and reasons | Small owner-coded categories and missingness; restricted examples for semantic audit | Customer records, production database extracts, raw application/incident transcripts |

Exact hashes, timestamps, unusual failures and repository topology can enable reidentification. Call the data **pseudonymized** unless a responsible review supports a stronger claim. Prefer owner-run extraction with keys and original sources retained by the owner. If shifting time for disclosure, use a consistent transformation that preserves within-system order and intervals; document loss of cross-system alignment. Never coarsen the controlled analytical timestamps until they can no longer prove the timing contract.

## Proposed lifecycle to agree, not an imposed policy

1. **Screening:** collect contact role and aggregate feasibility answers only, with an explanation of purpose. Keep identifiable contact records separate from research metadata; no unnecessary personal details.
2. **Agreement:** record purpose, permitted repositories/fields/window, owner and institution decisions, credential/hosting responsibility, publication terms and withdrawal limits. No broad secondary-use consent by default.
3. **Capture:** source-filter before export; store encrypted, access-controlled data with an approved access list and access logs. The collector's needed secret is held operationally, never exported as research data. Stop collection and notify the owner if prohibited content enters an export.
4. **Retention:** propose deleting transferred raw payloads within 30 days after validation, unless an approved need for a longer audit period is documented; preserve only approved minimized provenance needed for reproducibility. Define a bounded derived-data period, for example 12 months after study closure, subject to institutional/sponsor requirements. No default indefinite retention.
5. **Validation/reproducibility:** owner retains original evidence/key as agreed; share extraction versions, sanitized schemas, hashes, aggregate counts and reason-coded exclusions. Source hashes establish file identity, not scientific validity or absence of private content.
6. **Closure:** revoke collection access, stop the receiver/export, remove records according to agreement and document deletion. A gate is never removed because observation ends; research staff do not control it.

These proposed periods must not conflict with mandatory institutional retention. Do not delete source evidence required by an agreed audit before an approved minimized provenance record exists. Public disclosure is a separate decision from internal collection. Do not promise anonymity, exemption, confidentiality terms, approvals or benefits that have not been agreed.
