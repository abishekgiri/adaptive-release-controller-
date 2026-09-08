# Proposed cooperating-repository recruitment protocol

**DRAFT — no contacts or data requests sent.** Recruitment is not authorized by this document. Current feasibility is **CONDITIONALLY FEASIBLE**, with zero confirmed participants and no evidence yet of willingness, adequate outcome counts or passive capture access. Do not infer cooperation from public repository visibility, popularity or existing code in this project.

## Ideal participant and minimum requirements

The ideal participant is an owner of an actively maintained service or release pipeline using GitHub Actions with an **existing** protected release environment, a meaningful approval decision, an immutable artifact path, frequent real releases and a retained, interpretable downstream execution record. A release manager and a platform engineer should be available. A harmless existing environment is desirable for later instrumentation work; participation need not include intervention rights.

Use the following screening rules before choosing repositories. First ask for process descriptions and aggregates over the same proposed recent six-month window, not unrestricted credentials or a dump of developer records. Missing historical coverage is itself an answer.

| Criterion | Include / exclude rule | Evidence to request after recruitment authorization |
|---|---|---|
| Real protected operation | Include actual publishing/delivery/deployment work; exclude jobs that only build/test, tutorials, newly invented gates and disposable examples from the empirical cohort | Sanitized workflow excerpt and owner explanation of the protected side effect |
| Existing approval mechanism | Include a gate operated for a legitimate existing reason; exclude an `environment` label without a real hold/review, or a gate installed solely to create study examples | Rule/configuration export, applicable dates and normal reviewer role |
| Action scope | Require request/run/environment scope to map to a single declared job; if it admits a bundle, reassess the unit/outcome before enrollment | Job dependency and environment map, overrides and alternate release paths |
| Identity and label linkage | Require immutable release content, request/attempt IDs and terminal outcome linkage; exclude moving-ref-only or irreconcilable records | Data dictionary and owner-validated joins |
| Time provenance | Historical studies need contemporaneous source evidence; prospective study needs passive observation while the existing gate is still held | Existing event journal or permission for a read-only collector; no artificial extension of approval time |
| Outcome meaning and completeness | Require known success/failure/timeout semantics plus administrative/missing-state counts; exclude systematic masking of failure with no alternate source | Versioned job contract, counts and small prespecified semantic sample |
| Release frequency | Prefer at least 20 eligible requests/month for the access pilot; lower-frequency candidates remain eligible if collective accrual meets the later precision/time budget | Six-month monthly counts of requested, approved, started and resolved attempts; this is a planning preference, not a power threshold |
| Failure variation | Need enough natural adverse outcomes **across the cohort**; no manufactured failures | Aggregate outcome counts with missingness and operational subtype coverage. Do not exclude an otherwise eligible all-success repository merely for its outcome rate |
| Permission and data rights | Require owner authority, approved scope, privacy terms and permission to publish at least aggregate methodology/results | Named organizational approver and agreed terms |
| Independent replication | Record shared organization, services, maintainers, workflow templates, deployment infrastructure and duplicated artifacts | Dependency map; different repository names do not establish independence |

Exclusions and reasons must be recorded before examining predictive performance. If recruiting unusually failure-prone projects for practical event accrual, label the sample as enriched and restrict prevalence/generalization claims; do not conceal this as representative sampling. Retain zero-failure eligible projects in cohort coverage reports. Historic gate/rule changes define separate periods, not interchangeable records.

The prior Flask/Requests/Express development queries neither demonstrate these requirements nor designate those maintainers as candidates who have consented. They cannot silently become untouched evaluation projects.

## Recruitment route and bounded sequence

No random repository scrape is proposed. These are channels to use **only after the user authorizes outreach**:

| Priority / channel | Practical route | Why it may work; uncertainty |
|---|---|---|
| 1 — existing contribution relationships | User identifies maintainers they already know; request a short owner-mediated feasibility discussion | Lower introduction burden; no such relationships have been established in this session |
| 2 — university collaborators / software-engineering groups | Ask a collaborator or group liaison for an introduction to an operator with protected release infrastructure | May provide ethics/data infrastructure; a lab toy pipeline is not a substitute for a real workload |
| 3 — company platform/release teams | Seek an owner-operated sanitized export and passive shadow capture, without production credentials | Best chance of meaningful release volume/telemetry; approvals, legal review and confidentiality may take longer |
| 4 — open-source infrastructure maintainers | Approach organizations known through an introduction or volunteered process description to operate protected publishing/deployment environments | Public code can ease provenance; release volume and failures may be sparse |
| 5 — organizations already using protected environments | Use platform-engineer or research-community introductions and opt-in calls, with moderator permission | Relevant configuration is more likely; no list of verified willing organizations currently exists |

Proposed funnel: after explicit authorization, spend at most **six weeks** on up to **12 targeted invitations** and one polite follow-up per invitation; no repeated unsolicited contact. Count delivered invitations, responses, technical eligibility, permission barriers and owner commitments. At week three, review whether introductions are reaching actual infrastructure owners. At week six, stop empirical preparation if nobody can commit to Level 1 validation and a credible Level 2 route. These are resource-budget proposals, not claims about expected response rates; no clock or automation has been started.

One cooperating owner is enough for a **data/access feasibility pilot**, not cross-project claims. A provisional replication target is six distinct services across at least three organizations with documented dependencies. This is a recruitment coverage goal, **not statistical sufficiency or a generalization guarantee**. A single-company multi-service study can be valuable as a bounded case study. A later precision assessment must determine whether a confirmatory study is affordable before any model evaluation is commissioned.

## What we would ask for, and expected burden

All times below are estimates for planning, not measured commitments.

1. **Initial screen: 20–30 minutes.** Confirm protected operation, gate owner, request frequency, approximate outcomes, historical coverage, permitted sharing and instrumentation constraints. No identifiable research records required for this screen.
2. **Semantic/access review: about 1–2 hours of owner/engineer time.** Review sanitized configuration, immutable joins, cancellation/timeout meaning, aggregate counts and the minimum access matrix. Legal/ethics/security approval is additional and may dominate elapsed time.
3. **Data-only validation: about 2–4 engineer hours if an existing exporter is adaptable.** Supply a bounded fixed-window sample and reconcile distinct failure/administrative categories plus prespecified successes. Bespoke integration may exceed this budget and trigger reassessment.
4. **Optional future passive capture: approximately half to two engineer days for setup, then 15–30 minutes/week initially.** Owner operates or authorizes a least-privilege collector; capture lasts according to a later accrual budget, not an assumed short pilot. No scores shown and no gate delays introduced.
5. **Optional mechanism testing is separate.** A participant may decline it and still support prediction research. No request for automatic production admission accompanies this invitation.

Ask for: pseudonymous service/request/attempt IDs, content fingerprints, environment class, exact context receipt/cutoff evidence, bounded change aggregates, upstream completed-result summaries, prior stage history, approval/denial/no-start states, execution outcomes, missingness and owner-coded failure categories. Optional operational cost ranges or a statement of useful warning burden help decide whether prediction is worth studying. Raw source can remain with the owner while a versioned extractor emits approved aggregates.

## Privacy, security and ethics considerations

These are proposed collection constraints responsive to the study, not a request to perform a new security audit. Do not collect secrets, tokens, cloud credentials, customer payloads, production database contents, full incident transcripts, raw application logs, developer emails, private message text or individual performance rankings. Author identity is not a required feature. Avoid full source/diffs when derived counts/categories suffice; use restricted, redacted snippets only when needed to validate outcome semantics.

Repository URLs, commit hashes, file paths, exact timestamps and rare failure combinations can reidentify organizations or people even after names are removed. Prefer owner-held linkage keys, keyed pseudonyms, local extraction, encrypted transfer/storage and controlled access. Preserve precise timing in the protected analytical environment; coarsen only released summaries without breaking the analysis. Do not promise anonymity merely because identifiers are hashed. Agree retention, deletion/withdrawal limits, incident handling, secondary-use limits and publication disclosure rules before transfer. Owner permission to share organization data does not automatically resolve employee/participant obligations.

Before collecting organization/user-level research records or conducting research interviews, obtain a determination from the relevant institutional ethics/IRB process and comply with applicable organizational/local requirements. The U.S. HHS guidance distinguishes secondary coded data where investigators cannot readily identify individuals from research involving identifiable private information; it recommends designated institutional determination rather than investigator self-certification. It does not establish that this particular project is exempt or that U.S. rules govern every collaborator. [OHRP coded-data guidance](https://www.hhs.gov/ohrp/coded-private-information-or-biospecimens-used-research.html), [OHRP decision charts](https://www.hhs.gov/ohrp/regulations-and-policy/decision-charts-2018/index.html).

Participant review may correct factual errors and remove confidential material, but terms must allow honest null/adverse results. If only favorable results may be published, exclude the arrangement as a basis for the intended research paper. Use aggregate/pseudonymized reproducibility artifacts or an approved controlled-access route; never make raw public release a participation requirement.

## Proposed recruitment message — not sent

> Hello [name], I am assessing whether a research study of protected release jobs is practical. The question is whether information already available while a release awaits approval can predict the execution outcome of subsequently approved jobs. This is initially an observational study; we would not change approval decisions or request production credentials.
>
> Does your team already use GitHub Actions with a meaningful protected release or deployment environment? If so, would the responsible maintainer/platform engineer be open to a 20–30 minute feasibility conversation about release volume, recorded outcomes and a possible sanitized, owner-produced data export? Participation and any later instrumentation would require separate agreement. We do not need customer data, secrets, personal rankings or unrestricted source access, and a live enforcement test is optional and separately authorized.
>
> We are seeking both positive and negative feasibility evidence and cannot promise a useful prediction model. There is no obligation to share records during the initial conversation. Thank you for considering it.

## Present assessment

Obtaining a collaborator is **plausible through owner-mediated access, but unverified**. The available evidence supports neither “easy to recruit” nor “unobtainable.” The first useful milestone is an owner commitment and a valid outcome sample—not more algorithm implementation. See [access levels](access-requirements.md) and [stop criteria](stop-criteria.md).
