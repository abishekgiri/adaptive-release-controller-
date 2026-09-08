# Protected release-job study: recruitment package

Prepared 2026-09-07. Ready-to-adapt text; **nothing sent**. Replace bracketed fields truthfully; omit affiliation if none. This package seeks feasibility discussions, not consent to collect data. The research question is unchanged and `ci-design-v2` remains unratified. Send the short message first; attach the one-page summary only where useful.

## 1. Very short outreach

Hello [name], I’m assessing a study of whether information available before approval of a protected release job can predict whether the subsequently approved job succeeds or fails. Does your team use GitHub Actions with an existing release approval gate? Would its owner be open to a 20-minute feasibility conversation? Initially we need process details and aggregate counts—not secrets, source contents or changed deployment decisions. [Name / contact]

## 2. Professional email

**Subject:** Feasibility collaboration: outcomes of approved release jobs

Hello [name],

I’m preparing a narrowly scoped observational study: “Studying whether information available before approval of a protected release job can predict whether the subsequently approved job succeeds or fails.”

We are looking for teams already using GitHub Actions with a meaningful release approval gate, historical approved executions and some execution failures or timeouts. The first step is a 20–30 minute discussion of your process and approximate event counts. Any later data sharing or passive observation would require a separate agreement; we would not change approvals or show recommendations during observation.

The minimum data would be pseudonymous request/job identifiers, timing, approval states, upstream CI summaries, bounded change/history aggregates and protected-job outcomes. Source contents, secrets and customer data are unnecessary. An owner-operated exporter is an option.

In return, we would provide a concise report on timing coverage, outcome completeness and study feasibility, plus any later agreed aggregate research findings. We cannot promise a useful predictor or operational improvement.

Would you or the person responsible for release infrastructure be willing to discuss suitability? No records or credentials are needed for the initial conversation.

Thank you,
[Name]
[Affiliation, if applicable]
[Contact]

## 3. Open-source-maintainer version

**Subject:** Small feasibility request about your release approval workflow

Hello [name],

I’m exploring whether information available before approval of a protected release job can predict whether that subsequently approved job succeeds or fails. I’m seeking projects with an existing GitHub Actions approval gate around actual package publication or deployment, including some historical execution failures/timeouts.

I’m not asking you to add a gate or change release decisions. A first conversation would establish what the protected job does, roughly how often it runs, and whether its approvals and outcomes are recorded. If suitable, we could agree on a small metadata export and later passive observation. Public source contents need not be copied; change counts and category flags can be derived locally.

The initial ask is 20–30 minutes, with no commitment to further work. We would return an outcome/timing data-quality summary and agreed aggregate findings. We will not rank maintainers or present the project’s failures without agreed disclosure terms.

Would the maintainer responsible for releases be interested? Thank you for considering it.
[Name / contact]

## 4. University/research-collaborator version

**Subject:** Access-feasibility collaboration on protected release-job outcomes

Hello [name],

I’m reviewing access feasibility for a study of whether pre-approval CI/change/history information predicts success or failure of subsequently approved protected release jobs. The current question is supervised prediction within the observed approved population; denied releases have no assumed counterfactual labels.

Do you know an operator with an existing GitHub Actions release gate, retained approval/outcome metadata, some adverse execution outcomes and willingness to permit passive observation? I would value an introduction or a short discussion of access and measurement pitfalls. A newly created laboratory repository would be useful only for separate engineering checks, not empirical evidence for this question.

No model experiment, intervention or data collection is underway. Any collaboration would first resolve owner permission, outcome semantics, timing, event accrual and the relevant institutional ethics/data review. The initial request is a 20–30 minute feasibility discussion, not access to private records.

We would share the measurement protocol, feasibility findings and any later agreed research outputs. Authorship or institutional affiliation is not presumed.
[Name / contact]

## 5. Company/engineering-team version

**Subject:** Read-only feasibility study of protected release-job outcomes

Hello [name/team],

I’m seeking a release-platform owner for a limited observational study: whether information available before approval of a protected release job can predict whether the subsequently approved job succeeds or fails.

An appropriate pipeline already uses GitHub Actions and an approval gate for actual delivery/deployment work. We initially need only a process walkthrough and monthly aggregate counts. If there is a fit, your team could run an approved exporter/collector and share minimized, pseudonymous records of requests, approvals, timing, upstream summaries, change/history aggregates and job conclusions.

We do not need source contents, production credentials, customer data, developer identities or private discussion text. Existing approval decisions would continue unchanged, with no scores shown to reviewers. Any prospective instrumentation would require your owner/security/data approvals; intervention is outside this request.

We estimate 20–30 minutes for the first discussion, then 1–2 hours for semantic review if interested. Integration and legal-review effort would be estimated separately. Your team would receive a data-quality/feasibility report and any agreed aggregate findings, with no promised accuracy or deployment improvement.

Could you direct this to the release-platform owner and release manager?
[Name / contact]

## 6. Security/privacy FAQ

| Question | Answer |
|---|---|
| What exact records would you need? | Stable pseudonymous repository/service/request/run/attempt/job/environment/content IDs; gate/rule version; actual source-receipt and snapshot times; upstream CI summaries; bounded change/history aggregates; approval/denial/pending states; protected-job starts, terminal outcomes and missingness. See the capture specification. |
| Do you need source contents? | No. Owners can compute files-changed/churn/category counts and content fingerprints locally. A sanitized workflow excerpt or process walkthrough helps validate semantics; full proprietary source is unnecessary. |
| Do you need secrets or production access? | No secrets, tokens, production shells, cloud credentials or customer data should be transferred to researchers. If an owner operates the collector, operational credentials stay under its control. |
| What access level? | Level 1 owner-approved exports/read access for validation; Level 2 passive observation for a future prospective study. No researcher administration or approval rights are needed for observation. |
| Will you alter releases or approvals? | No. No added hold, automatic callback, score display, approval, denial, cancellation or workflow change during initial observation. Installing passive subscriptions still requires owner approval. |
| Can identifiers be anonymized? | Use owner-held keyed pseudonyms and minimized fields. Exact timing, hashes and rare events can reidentify projects, so we promise controlled pseudonymization, not guaranteed anonymity. |
| What personal data is needed? | Individual developer/reviewer identity is not required. Reviewer role and versioned approval policy may suffice; omit names, emails, comments and performance rankings. |
| Can data stay in our environment? | Yes: owner-run extraction and controlled analysis/export are preferred alternatives. Permission to publish honest aggregate findings must be agreed in advance. |
| What burden should we expect? | Initial call 20–30 minutes; semantic review 1–2 hours; a simple existing export approximately 2–4 engineer hours. Passive integration may take half to two engineer days plus brief weekly checks. These are planning estimates; do not promise bespoke integration fits them. |
| What do we receive? | A concise report on outcome definitions, missingness, timing eligibility and feasibility; an opportunity to correct factual/confidentiality issues; later agreed aggregate research results if a study proceeds. No paid service, useful model or operational benefit is promised. |
| Can we withdraw or restrict sharing? | Scope, retention, withdrawal/deletion limits, access and publication terms are agreed before collection. Confidentiality review may not suppress honest null/adverse findings. |
| Is ethics review needed? | The relevant institution must determine applicable ethics/IRB and data-review requirements before collection. We do not assert exemption or provide a legal conclusion. |

## 7. One-page study summary

**Study:** Studying whether information available before approval of a protected release job can predict whether the subsequently approved job succeeds or fails.

**Purpose.** Establish whether advance information has measurable predictive value for a real protected release operation. A useful result could inform later research on reviewer decision support. Job failure is not automatically a production incident; deployment benefit is not being claimed.

**Suitable participant.** A team with GitHub Actions, an existing meaningful approval gate, a clearly identified protected publishing/deployment job, historical approved executions including some failures or execution timeouts, interpretable identifiers/timing, an owner who can verify semantics and permission for passive observation.

**What we ask first.** A 20–30 minute process/feasibility discussion and approximate monthly request/outcome counts. No credentials or raw records during the initial conversation. Subsequent sharing requires separate terms and applicable institutional review.

**Minimum data.** Pseudonymous request/run/attempt/job/environment/content identifiers; gate/rule version and timing; upstream completed CI summaries; locally derived change/history aggregates; approval/denial/pending state; job start/completion and outcome, including missing/administrative states. Every predictive feature must be captured before approval. Denied jobs are not assigned imagined success/failure labels.

**Not required.** Source-code contents, secrets, production/customer data, developer names, private comments or unrestricted repository access. An owner-operated exporter/collector can keep credentials and raw sources inside the organization.

**Observation only.** Existing decisions remain unchanged; no added gate delay or recommendations are shown. Initial historical validation uses Level 1 exports/read access; later prospective capture uses Level 2 passive access. A disposable mechanism test is separate and requires explicit authorization.

**Burden and return.** Beyond the first call, estimate 1–2 hours for semantics and 2–4 engineer hours for a simple export; integration/legal review may take longer. Participants receive a timing/outcome data-quality report and agreed aggregate findings, with no promised accuracy or savings.

**Governance.** Agree minimization, pseudonymization, controlled access, retention, publication and withdrawal terms before collection. The study is still conditionally feasible and may stop if access, meaningful outcomes or sufficient adverse events cannot be obtained.

**Contact:** [Name / contact / truthful affiliation if applicable]

## Sender-only contact shortlist — do not include in invitations

First contact a release manager and the engineer who administers the approval environment at an organization where you already have a legitimate introduction, or the release maintainer of a project you contribute to. No such relationship has been identified in this session; do not imply that it exists.

Two specific research-introduction leads, verified from public professional pages on 2026-09-07:

- **Michael Hilton, Carnegie Mellon University:** `mhilton@cmu.edu`. His departmental page describes continuous-integration research; ask for access/measurement advice or an operator introduction, not assume possession of a dataset. [Professional contact](https://www.cs.cmu.edu/~mhilton/), [CMU research description](https://s3d.cmu.edu/people/core-faculty/hilton-michael.html).
- **Andy Zaidman, TU Delft:** `a.e.zaidman@tudelft.nl`. His research/contact page covers empirical software testing/evolution and CI-related work. Ask whether a relevant industry or open-source operator might be interested. [Professional contact/research](https://azaidman.github.io/).

These are relevant public professional contacts, **not confirmed collaborators or verified owners of qualifying repositories**. No invitation, calendar booking or follow-up has been sent. Engineering-team recipients should be the actual gate owner and release manager; a general developer-relations address is an introduction route, not consent.
