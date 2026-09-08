# Adaptive release controller: research and access acquisition

**Current phase: access acquisition. Research status: conditionally feasible.**

The current study asks:

> Can information available before approval of a protected release job predict whether the subsequently approved job succeeds or fails?

We are seeking **two genuinely qualifying repositories** before assessing a credible path to a larger multi-organization study. The initial work concerns passive observation of existing approval workflows. It does not change release decisions or require source-code contents, secrets, or production-system access; owners can provide minimized metadata and locally derived change summaries.

The project has not established predictive value, deployment benefit, or live enforcement. The proposed outcome is the execution result of a named protected release job, not a production incident. Denied, cancelled, never-started, and missing executions are kept separate from observed success/failure labels.

## Start here

| Purpose | Document |
|---|---|
| Review the first five leads and personalized message drafts | [Candidate shortlist](review/v2/first-five-candidates.md) |
| Track contact, responses, follow-ups, and qualification | [Recruitment ledger](review/v2/recruitment-ledger.md) · [Detailed records](review/v2/recruitment-ledger.json) |
| Introduce the study to a potential participant | [Recruitment package and FAQ](review/v2/recruitment-package.md) |
| Check whether a repository qualifies | [Screening checklist](review/v2/repository-screening-checklist.md) · [Owner questions](review/v2/owner-verification-questionnaire.md) |
| Understand the required data and access | [Outcome definition](review/v2/outcome-definition.md) · [Access levels](review/v2/access-requirements.md) |
| Review passive capture and data handling | [Observation plan](review/v2/prospective-observation-plan.md) · [Governance notes](review/v2/data-governance-notes.md) |
| Decide whether recruitment and event accrual justify continuing | [Bounded recruitment stop rule](review/v2/recruitment-stop-rule.md) |

Candidates are handled one at a time. A public workflow or interested contact is not a qualifying repository: the owner must verify the gate, job semantics, timing, activity, adverse outcomes, and permission for passive observation. The ledger is the source for current outreach status; messages and follow-ups require explicit approval before sending.

## Current limits

- The chosen direction is **Path B**. [`ci-design-v1`](review/design/README.md) remains frozen and **NO-GO**; [`ci-design-v2`](review/ci-design-v2-draft.md) remains an **unratified draft**.
- The next step is access validation. New model experiments, algorithm development, manuscript revisions, and repository scraping are not part of the current phase.
- Real protected-stage semantics, prospective feature capture, outcome completeness, and adverse-event frequency still require owner-verified evidence. See the [feasibility assessment](review/v2/stop-criteria.md).
- Live enforcement remains **UNVERIFIED**. The [disposable-repository test plan](review/v2/mechanism-validation-plan.md) is unexecuted and requires separate authorization. Its fixtures could establish engineering behavior only, not real-world predictive performance or utility.
- Denial suppresses the natural execution label. The initial prediction population and the limits of subsequent decision-support claims are set out in the [selective-label analysis](review/v2/selective-label-analysis.md). Contextual bandits are not the default formulation.

## Historical audit and corrected results

The repository began with simulated deploy, canary, and block decisions under asymmetric costs. The September 2026 submission audit found temporal feature leakage, a defective Page-Hinkley implementation, inconsistent delayed/censored evaluation, and missing simple cost-aware baselines. The corrected results withdraw the original general operating-envelope claim. See [the technical review](review/submission-review.md), [the audit-revised manuscript](paper/adaptive-deployment-control.md), and [the numerical source of truth](paper/source-of-truth.md).

The following retained results concern that earlier simulation/CI-proxy study. They do not evaluate the current protected release-job question.

| Setting | Static rules | LinUCB | Simple control |
|---|---:|---:|---:|
| Synthetic fixture | 1884 | 1863 | Bayesian rate 1713.5 |
| High failure cost | 2578 | 1972 | Always block 1865 |
| Low safe-block cost | 1591 | 1072 | Always block 1005 |
| GitHub Actions proxy | 640 | 779 | Rolling cost rule 624.5 |

Real-data Thompson Sampling costs 665.7 ± 39.8 across seeds 0-29; all policies are evaluated on the same 578 resolved labels among 600 workflow rows. These are simulated CI-proxy costs, not measured deployment savings. Repeated algorithm seeds do not estimate uncertainty across repositories.

The subsequent [data-only pilot](review/pilot/README.md) did not establish the required prospective observation route. The [timing-boundary review](review/timing-boundary-review.md) explains the move to the current access-acquisition work. Original artifacts remain archived under `review/original/`.

## Install and run regression checks

Run from the repository root with Python 3.13:

```bash
python3.13 -m venv .venv
.venv/bin/python -m pip install -r requirements-audit.txt
.venv/bin/python -m pytest
```

These are software regression checks, not new research experiments or evidence of deployment effectiveness.

## Historical reproduction tooling

The commands below are retained to document the earlier audit workflow. **They are not the next step in the current phase and should not be run without renewed authorization.** They regenerate experiment and manuscript artifacts. The [design preflight](review/design/README.md) is not wired into every legacy runner, so the NO-GO status is not a universal execution interlock.

```bash
.venv/bin/python -m experiments.reproduce_submission
.venv/bin/python paper/build_corrected.py
bash paper/build_documents.sh
```

The last command also needs Pandoc and a TeX installation providing `pdflatex` on PATH. The document sources compile as a readable audit revision; venue-specific submission formatting is not claimed complete. The existing Word export is generated from the same Markdown source using Pandoc.

The reproduction runs 30 seeds for main replay, cost robustness, artificial delays, five cost matrices, and two duplicate/commit sensitivity cohorts; the deterministic ablation uses seed 0. Separate synthetic drift evaluation uses 30 environment seeds, with independent stationary seeds 1000-1029 for the detector calibration screen. Every seed's summary, project action trace, and figure input is retained under `experiments/results/headline/corrected/`. `manifest.json` records code/data hashes and the tested environment. Original results and manuscript formats are archived under `review/original/` and are superseded.

Frozen data are in `data/fixtures/`; no ignored `data/raw/` file is required. Read [fixture provenance](data/fixtures/README.md): the synthetic generator and complete original GitHub collection manifest remain unavailable. The real export lacks workflow/run identifiers and contains repeated SHAs. Numerical reproduction does not solve those validity limitations.

## Historical implementation and limitations

The corrected historical replay uses `data/loaders.py`, `policies/linucb.py`, `policies/thompson.py`, `policies/cost_rules.py`, `delayed/buffer.py`, `evaluation/online_replay.py`, and the named experiment runners. Historical CI features and labels obey the chosen observation clock. Costs exclude the same unknown-label rows for all actions. Page-Hinkley follows a documented upward reference recurrence; its threshold still needs scenario-specific calibration.

Several legacy modules remain placeholders, including the offline classifier, epsilon-greedy policy, replay-environment abstraction, standalone delay sampler and imputation strategies. SNIPS/DR and production propensity logging are not implemented. The CI data cannot support causal off-policy evaluation of unlogged deployment actions. Those paths are not used as evidence for the corrected paper.

The historical corrected contribution is a bounded simulation study and a baseline-selection finding. The current empirical study remains conditional on obtaining suitable access and observations; existing code and repeated algorithm seeds do not resolve those requirements.
