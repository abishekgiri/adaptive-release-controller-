# First five access-acquisition leads

Assessed 2026-09-07 using bounded public-page research. No repository clone, API run-history collection, scraping pipeline, experiments or design changes. **Five leads, zero contacted, zero qualifying repositories.** Priority ranks the value of an initial inquiry; it does not certify qualification or willingness. The [existing screening criteria](repository-screening-checklist.md) remain unchanged.

Start with **ACQ-001, Astral's release-infrastructure owner**. Hold the other drafts and manage each contact separately. A warm introduction from the user can change contact order without changing qualification. No known personal relationship is assumed. Only explicit approval of a concrete recipient/channel/message permits sending; ledger entry or draft creation is not authorization.

The initial screen still requires an existing real gate, suitable single protected-job scope, historical approved executions averaging at least 20/month over the fixed six-month window, some natural execution failure/timeout, interpretable timing/identities, owner semantic verification and permission for passive observation. No lead here has passed these. Especially for package releases, weekly cadence may be insufficient. Do not count ungated nightlies, multiple artifacts, retries or neighboring projects to manufacture the threshold.

## ACQ-001

**Astral release-infrastructure owner / Ruff maintainers — HIGH.**

**Why relevant:** Ruff's own release instructions describe GitHub Actions builds and another team member's deployment approval before PyPI upload. That is stronger than an unexplained environment name. [Maintainer instructions](https://github.com/astral-sh/ruff/blob/main/CONTRIBUTING.md).

**What remains uncertain:** an indexed workflow version describes a dedicated approval job and a protection App connecting it to release jobs. We cannot assume one human approval equals one independently gated publish job, or that it occurs after every required upstream result. The current owner must verify the exact boundary/scope. [Workflow evidence](https://github.com/astral-sh/ruff/actions/runs/25183731115/workflow?pr=24581). “High frequency” in prose is not a measured 20/month count. Adverse cases and observation permission are unknown.

**Contact:** ask for the current release-infrastructure owner through [Ruff Discussions → General](https://github.com/astral-sh/ruff/discussions), or a user-supplied warm introduction. No direct owner email verified. This is one routing inquiry, not a bug report or a message to every maintainer.

**Draft — not sent:**

> Hello Ruff maintainers, your release guide describes building artifacts and waiting for another team member's approval before publishing. I'm studying whether information available before approval of a protected release job can predict whether the subsequently approved job succeeds or fails. Could the release-infrastructure owner confirm the current gate scope, roughly how many approved executions it handles monthly, and whether there have been any execution failures/timeouts? If it looks suitable, would you consider a brief discussion of passive observational metadata collection? No source contents, secrets, production access or changed release decisions are requested.

## ACQ-002

**Dask release managers — MEDIUM.**

**Why relevant:** the project's release procedure explicitly describes manual `pypi` approval after build/check/smoke-test completion for Dask and Distributed. This is a close documentary match to the desired sequence. [Release procedure](https://github.com/dask/dask/blob/main/docs/release-procedure.md).

**Why not HIGH:** six-month execution volume and natural adverse cases are unverified and may be too low. Dask and Distributed are coordinated releases, not evidence of two independent organizations. Documented retry handling is not evidence that qualifying failures actually occurred. A maintainer must confirm the gate, scope and records rather than relying on the guide.

**Contact:** [Dask Discourse](https://dask.discourse.group/) asking for the current release manager. The project directs general discussions there and discourages cross-posting. [Community guidance](https://docs.dask.org/en/stable/support.html). No personal recipient or private data access is assumed.

**Draft — not sent:**

> Hello Dask release maintainers, your release procedure describes approving PyPI publication after the build and smoke tests pass. I'm studying whether information available before approval of a protected release job can predict whether the subsequently approved job succeeds or fails. Could the current release manager share approximate monthly approved-execution counts and whether any failures/timeouts occurred in the past six months? If the workflow qualifies, would you be open to discussing passive metadata collection? We would not change approvals or request source contents, secrets or production access.

## ACQ-003

**Lucain Pouget (@Wauplin), as a route to the Hugging Face Hub release owner — MEDIUM.**

**Why relevant:** Pouget coauthored a June 2026 account of the library's release process and publicly identifies as a Hub-library maintainer. This is an actual maintainer lead, not an inferred role based on a commit. [Release-process post](https://huggingface.co/blog/huggingface-hub-release-ci), [maintainer self-identification](https://discuss.huggingface.co/t/outside-contributions-silently-ignored-multiple-times-now/175134).

**Qualification concerns:** the post describes weekly releases, which alone would fall short of the activity screen. The current workflow documents required reviewers for `pypi`, but multiple publish jobs share it and build work occurs inside a protected publish job; pre-gate upstream CI availability and single-job admission scope are not established. The blog's human review of release notes must not be confused with a pre-execution environment gate. [Workflow](https://github.com/huggingface/huggingface_hub/blob/main/.github/workflows/release.yml). Useful initial contact, not a presumptive qualifying repository. If actual counts/scope fail, reject this repository without changing criteria; an introduction to another existing operator is a separate lead.

**Contact:** [professional forum profile](https://discuss.huggingface.co/u/Wauplin); a forum message if that route is enabled and approved, otherwise one appropriate Hub-community routing request. No email verified; do not invent one or attach recruitment to an unrelated bug thread.

**Draft — not sent:**

> Hi Lucain, I read your post about the Hub library's release workflow and saw the documented PyPI environment gate. I'm studying whether information available before approval of a protected release job predicts whether that subsequently approved job succeeds or fails. Could you clarify whether the current gate covers an identifiable job, its approximate approved-execution volume, and whether failures/timeouts are recorded? If suitable, would you or its owner consider a short discussion of passive metadata collection? No approval changes, source contents, secrets or production access are requested.

## ACQ-004

**Michael Hilton, Carnegie Mellon University — MEDIUM, introduction lead.**

**Why relevant:** his departmental profile describes research on continuous integration, while his current professional page supplies contact details. [Research relevance](https://s3d.cmu.edu/people/core-faculty/hilton-michael.html), [current contact](https://www.cs.cmu.edu/~mhilton/).

**Limit:** no evidence reviewed establishes that he operates a qualifying protected-release repository or controls access to one. The value is a possible operator introduction. A researcher reply is not a qualifying repository or participant commitment.

**Contact:** professional email `mhilton@cmu.edu`.

**Draft — not sent:**

> Hello Michael, your work on continuous integration prompted this request. I'm seeking access for a narrow observational study: whether information available before approval of a protected release job predicts whether the subsequently approved job succeeds or fails. Do you know an operator using GitHub Actions with an existing approval gate who might discuss workflow qualification and passive metadata collection? The first ask is a brief feasibility conversation, not data transfer or a design review. We would not change decisions or request source contents, secrets or production access.

## ACQ-005

**Andy Zaidman, TU Delft — MEDIUM, introduction lead.**

**Why relevant:** his professional page describes empirical testing/evolution work and CI-related research and lists his institutional contact. [Research and contact](https://azaidman.github.io/).

**Limit:** relevant expertise does not establish a willing operator, a qualifying GitHub Actions gate, or permission to share records. Seek one relevant introduction, not an open-ended research consultation or assumed collaboration.

**Contact:** professional email `a.e.zaidman@tudelft.nl`.

**Draft — not sent:**

> Hello Andy, your empirical software-testing and CI work prompted me to ask about an operator introduction. I'm studying whether information available before approval of a protected release job predicts whether the subsequently approved job succeeds or fails. Do you know a team with an existing GitHub Actions approval gate, regular protected executions and recorded failures/timeouts that might discuss passive observational metadata collection? We initially need only a short qualification conversation. No source contents, secrets, production access or changes to approval decisions are requested.

## Handling the first reply

Record the actual reply and date before changing screening status. First establish owner identity, actual gate/operation, approximate six-month approved execution/adverse counts and willingness to discuss passive observation. If these fail the existing screen, close the candidate honestly. Only then request the detailed existing owner questionnaire under appropriate permission. No institution or organization is counted toward the multi-organization study until an actual qualifying repository and access commitment exist.

The [recruitment ledger](recruitment-ledger.json) stores all requested fields; [readable status](recruitment-ledger.md) summarizes the current queue. Unsent messages have null contact/follow-up dates. No countdown, reminder, invitation or message is started by this assessment.
