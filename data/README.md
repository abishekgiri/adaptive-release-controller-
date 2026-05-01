# data/

This directory holds dataset metadata, schemas, and loader code.
Raw data files live in `data/raw/` which is gitignored. Never commit binary blobs, CSV dumps, or database files here.

---

## Dataset Evaluation

Three candidates were evaluated against the contextual bandit formulation in `docs/problem-formulation.md`.  
The key requirements are: commit-level context features, a linkable build/deployment outcome, temporal ordering for trajectory construction, and multi-project coverage for drift analysis.

### Evaluation Matrix

| Criterion | TravisTorrent | Apachejit | Custom GH Actions |
|-----------|:---:|:---:|:---:|
| CI/CD trajectory construction | ✓ | ~ | ✓ |
| Commit → build outcome linkage | ✓ (direct) | ~ (SZZ proxy) | ✓ (direct) |
| Delayed reward modeling | ~ (build duration only) | ✓ (bug report lag) | ✓ (configurable) |
| Project-level drift analysis | ✓ (~1000 projects, 4+ years) | ✓ (14 Apache projects) | ✓ (configurable) |
| Scale | ✓ (3.7M builds) | ~ (~300k commits) | ~ (rate-limited) |
| Standard benchmark (reproducible) | ✓ (MSR 2017, CC0) | ✓ (MSR 2021) | ✗ (must archive) |
| Context feature richness | ✓ (CI signal + churn) | ~ (churn only, no CI signal) | ✓ (full API payload) |
| Suitability for first experiment | **Primary** | **Fallback** | Deferred |

**Legend:** ✓ = meets requirement, ~ = partial, ✗ = does not meet requirement

---

### TravisTorrent (Primary)

**Reference:** Beller et al., "TravisTorrent: Synthesizing Travis CI and GitHub for Full-Stack Research", MSR 2017.  
**Source:** https://travistorrent.testroots.org  
**License:** CC0 (public domain)  
**Scale:** ~3.7M build records across ~1,000 GitHub projects, spanning approximately 2012–2016.

**Why it meets our requirements:**
- Builds are ordered by timestamp per project — natural trajectory construction over `(commit, build_outcome)` pairs.
- `tr_status ∈ {passed, failed, errored, canceled}` is a direct build-level outcome label, usable as the reward signal after mapping to the cost matrix.
- `tr_duration` gives build elapsed time, which we use to model reward delay: the reward is not observed until the build finishes, so `k_t ≈ ⌈tr_duration / Δt⌉` decision steps.
- Multi-project, multi-year coverage supports concept drift analysis out of the box. Projects with large commit volume (Rails, Mozilla, Homebrew) will produce long enough trajectories for meaningful bandit learning curves.
- Published at MSR, widely cited, CC0 license. Reviewers at MSR, SEAMS, ICSE-SEIP will recognise it.

**Limitations and validity risks:**
- **Build failure ≠ deployment failure.** TravisTorrent records CI build outcomes, not production deployment outcomes. We treat build failure as a proxy for deployment failure. This must be stated as a threat to validity: a build that passes may still fail in production; a build that fails may have been a flaky test. Mitigation: we restrict to non-flaky projects (build failure rate < 40%, consistent test suite sizes) and note this limitation in `docs/evaluation-protocol.md`.
- **No canary semantics.** The dataset has no notion of partial rollout. We map the three-action space `{deploy, canary, block}` to the build outcome as follows: a build that passed is treated as a `deploy` candidate; the policy then decides `deploy` vs. `canary` vs. `block`. The counterfactual cost for `canary` and `block` on builds that were actually deployed must be estimated via IPS — this is the off-policy evaluation problem directly.
- **Logging policy is implicit.** Every build in TravisTorrent was presumably run-to-completion and the result acted on by the team. The logging policy propensity is unknown. We assume a uniform propensity over `{deploy}` for passing builds and use IPS with clipped weights (clip at 20×) as per `docs/problem-formulation.md`.
- **Data vintage (2012–2016).** Modern CI/CD pipelines (GitHub Actions, multi-stage deployments, canary infrastructure) are not represented. Generalizability claims are limited to the dataset's era and project mix.
- **Project heterogeneity.** Projects range from active OSS with high commit volume to nearly-inactive repositories. We filter to projects with ≥ 500 builds and ≥ 12 months of history before loading.

---

### Apachejit (Fallback)

**Reference:** Fan et al., "Revisiting the VCCFinder Approach for Identifying Vulnerability-Contributing Commits", MSR 2021 (dataset reused in subsequent JIT work).  
**Source:** https://github.com/anapalu/ApacheJIT  
**License:** MIT  
**Scale:** ~14 Apache Software Foundation projects, ~300k commit records.

**Why it is the fallback:**
- SZZ-based labels link commits to bug-fix commits, giving a natural delayed outcome: the "reward" (did this commit introduce a bug?) is observed only when the subsequent fix commit lands, which models delayed feedback more authentically than build duration.
- Apache projects have long histories and clear release cycles, which makes drift analysis credible.
- Used in the JIT prediction literature (Kamei 2013, McIntosh & Kamei 2018), so comparisons to those baselines are direct.

**Why it is not primary:**
- No CI signal. Apachejit contains commit metadata (churn, complexity, author features) but not build pass/fail, test counts, or build duration. Our context space `x` is poorer: we lose the most informative features (`ci_tests_passed`, `ci_run_duration_seconds`) that `features/extractor.py` already extracts.
- SZZ-based labels have known false positive rates (~20–30%). The reward signal is noisier than a direct build outcome.
- 14 projects is a thin basis for generalizability claims at a venue like MSR or ICSE.

**Switch condition:** If TravisTorrent's build-outcome proxy is rejected by reviewers as too indirect for a deployment-outcome claim, we switch to Apachejit and reframe the outcome as "commit introduces a post-deployment defect" using the SZZ label.

---

### Custom GitHub Actions Export (Deferred)

Using `ingestion/github_client.py` against the GitHub REST API.

**Why deferred:**
- Outcome labeling requires defining "deployment failure" from first principles (revert commits? subsequent workflow failures? linked issue reports?). Each definition is a research decision that must be justified and is not standard.
- API rate limits (5,000 req/hr authenticated) constrain scale. A 100-project, 2-year export takes hours and must be re-run for reproducibility.
- Not a standard benchmark. Reproducibility requires archiving the raw export (large blob) or re-running the export script, both of which create logistics problems for artifact evaluation.
- Best suited for a **future phase** once the bandit formulation is validated on a standard dataset.

---

## Decision

**Primary dataset: TravisTorrent**  
**Fallback dataset: Apachejit**

The choice is locked for the first experiment. It may be revisited after results are in, but any switch resets the evaluation pipeline and requires re-running all baselines.

---

## Required Data Fields

These are the fields the bandit pipeline reads. They map directly to the context space `x` and reward `r` defined in `docs/problem-formulation.md`.

### Context fields (pre-action, go into `x`)

| Field | Source column (TravisTorrent) | Description |
|-------|-------------------------------|-------------|
| `commit_sha` | `git_trigger_commit` | Unique commit identifier |
| `project_slug` | `gh_project_name` | Owner/repo for trajectory grouping |
| `committed_at` | `git_committed_at` | Timestamp for temporal ordering |
| `files_changed` | `gh_files_changed` | Number of files modified |
| `lines_added` | `gh_lines_added` | Insertions |
| `lines_deleted` | `gh_lines_deleted` | Deletions |
| `src_churn` | `gh_src_churn` | Source-only line churn |
| `tests_run` | `gh_tests_run` | Number of tests executed |
| `tests_added` | `gh_tests_added` | Tests added in this commit |
| `build_duration_s` | `tr_duration` | Build elapsed time (seconds) |
| `is_pr` | `gh_is_pr` | Whether commit is a PR merge |
| `author_experience` | derived: prior commits by same author | Author familiarity with project |
| `recent_failure_rate` | derived: rolling 7-day failure rate | Recent CI health signal |
| `has_dependency_change` | derived from `gh_files_changed` paths | Matches `DEPENDENCY_FILES` in `features/extractor.py` |
| `has_risky_path_change` | derived from changed file paths | Matches `RISKY_PATH_PREFIXES` in `features/extractor.py` |

### Reward fields (post-action, go into `r` after delay `k`)

| Field | Source column (TravisTorrent) | Description |
|-------|-------------------------------|-------------|
| `build_outcome` | `tr_status` | `passed` / `failed` / `errored` / `canceled` |
| `build_started_at` | `tr_started_at` | When build began (action time proxy) |
| `build_finished_at` | `tr_finished_at` | When outcome was observable (reward arrival) |

### Reward mapping to cost matrix

| `build_outcome` | Assumed action | Cost (from `CostConfig` default) |
|-----------------|----------------|----------------------------------|
| `failed` / `errored` after `deploy` | deploy → failure | 10 |
| `failed` / `errored` after `canary` | canary → failure | 4 |
| `passed` blocked | block → would succeed | 2 |
| `passed` deployed | deploy → success | 0 |
| `canceled` | treated as censored reward | — |

---

## Reproducibility

When the dataset is downloaded, record the following here (fill in before running any experiment):

```
Dataset:       TravisTorrent
Version/tag:   (fill in: e.g. travistorrent_8_2_2017.csv)
Download URL:  https://travistorrent.testroots.org/dumps/
SHA-256:       (fill in after download: sha256sum travistorrent_*.csv)
Download date: (fill in)
Rows (raw):    (fill in)
Rows (after filtering ≥500 builds, ≥12 months): (fill in)
Projects (raw): (fill in)
Projects (after filter): (fill in)
```

**Fallback (Apachejit):**
```
Dataset:       ApacheJIT
Version/tag:   (fill in: git commit SHA of https://github.com/anapalu/ApacheJIT)
SHA-256:       (fill in)
Download date: (fill in)
```

---

## Storage Rules

- `data/raw/` — gitignored. All downloaded CSVs, parquet files, and intermediate formats live here. Never commit.
- `data/schemas.py` — typed dataclasses for the unified trajectory format. Committed.
- `data/loaders.py` — TravisTorrent and Apachejit loaders. Committed. Reads from `data/raw/`.
- This `data/README.md` — committed. Updated each time dataset version or filter criteria change.
