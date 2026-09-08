# Pilot data-quality report

Collection: 2026-09-07T05:53:41.692041+00:00 through 2026-09-07T18:00:27.800178+00:00. 3 polling rounds. Public development repositories only.

The latest-300-per-project frame is bounded and outcome-unfiltered. It is not a complete historical cohort, a fixed-calendar sample, or a representative prevalence estimate. Earlier attempts of observed reruns are expanded, which explains more attempts than runs.

| Development repository | Runs | Attempts | First / rerun | Head SHAs | Scoped first attempts | Success / failure / other | Scoped binary failure rate |
|---|---:|---:|---:|---:|---:|---|---:|
| expressjs/express | 303 | 328 | 303 / 25 | 90 | 94 | 54 / 14 / 26 | 20.588% |
| pallets/flask | 300 | 300 | 300 / 0 | 99 | 89 | 22 / 66 / 1 | 75.000% |
| psf/requests | 310 | 320 | 310 / 10 | 31 | 44 | 37 / 6 / 1 | 13.953% |

Total: **913 workflow runs, 948 valid attempts, three repositories**. There are 227 retrospectively scope-eligible first attempts, but **zero primary-eligible decision snapshots**. Scope eligibility here means workflow/event/attempt rules only, not temporal eligibility.

Exact reobservations across poll/export records: 587. These are repeated observations, not new independent runs or evidence of duplicate execution. Conflicting canonical identities/outcomes: 0. Acquisition anomalies quarantined: 1.

The schema contains 211 verified push-attempt identities. 737 attempts are retained in raw/diagnostic archives but omitted from normalized attempts because the tested revision or workflow identity is not proved. No guessed tested SHA or historical decision timestamp is inserted.

In 227/227 scoped first attempts, provider run_started_at equals created_at exactly. Sampled earliest job starts occur 2–12 seconds later. Blindly mapping run_started_at to the frozen execution-start boundary makes created_at <= decision_at < started_at an empty interval in these observations. The provider run lifecycle timestamp is not yet validated as the intended execution boundary. The later job time is not automatically a valid substitute; obtain authoritative semantics before any new capture claim.

Run completion time remains null. Label availability is the actual first successful terminal-response receipt. The maximum completed job time is a diagnostic and is not substituted for run-label availability. Current updated_at is not a completion timestamp.

Unique changes cannot be counted exactly across PR/merge semantics. Distinct head SHAs and provider PR numbers are reported separately in data-quality.json; neither is asserted to equal independent changes. Repeated SHAs cluster workflows and reruns.

## Per-project qualification, coverage and clustering

### expressjs/express
Provider ID 237159; JavaScript; provider repository-size proxy 9,861 KiB (not source LOC); nonfork=True, active=True. Historical activity older than six months is confirmed by an archived API query (returned count 715; filtered counts are not asserted exhaustive).
Recent sampled coverage 2026-06-15T20:29:31Z–2026-09-07T16:01:25Z (83.8 days). Listed workflows 8; observed workflows 6. Selected CI workflow IDs/names: 10933076 (ci), 106129301 (iojs-ci).
Observations per workflow: `{"106129301": 8, "109225080": 86, "10933076": 97, "138197155": 27, "150125695": 6, "84668683": 104}`.
All-attempt terminal categories: `{"action_required": 57, "cancelled": 5, "failure": 46, "success": 220}`. Scoped categories: `{"action_required": 23, "cancelled": 3, "failure": 14, "success": 54}`.
Scoped head-SHA clusters 88; largest contains 2 first workflow attempts. PR numbers present 16; PR-attempt rows without a PR number 212.
Change extraction 20/20 complete inventories; verified push diff basis 3/20; all sources were retrieved after their original CI runs.
New runs observed during this session 3; valid preexecution snapshots 0. These are sparse polling rounds, not an estimate of a continuously operated collector’s capture probability.

### pallets/flask
Provider ID 596892; Python; provider repository-size proxy 12,368 KiB (not source LOC); nonfork=True, active=True. Historical activity older than six months is confirmed by an archived API query (returned count 573; filtered counts are not asserted exhaustive).
Recent sampled coverage 2026-05-24T23:40:23Z–2026-09-07T01:36:18Z (105.1 days). Listed workflows 9; observed workflows 5. Selected CI workflow IDs/names: 1367898 (Tests).
Observations per workflow: `{"115253103": 100, "1367898": 89, "243361684": 3, "269761080": 2, "3605901": 106}`.
All-attempt terminal categories: `{"cancelled": 1, "failure": 152, "success": 147}`. Scoped categories: `{"cancelled": 1, "failure": 66, "success": 22}`.
Scoped head-SHA clusters 87; largest contains 2 first workflow attempts. PR numbers present 2; PR-attempt rows without a PR number 161.
Change extraction 20/20 complete inventories; verified push diff basis 2/20; all sources were retrieved after their original CI runs.
New runs observed during this session 0; valid preexecution snapshots 0. These are sparse polling rounds, not an estimate of a continuously operated collector’s capture probability.

### psf/requests
Provider ID 1362490; Python; provider repository-size proxy 13,627 KiB (not source LOC); nonfork=True, active=True. Historical activity older than six months is confirmed by an archived API query (returned count 802; filtered counts are not asserted exhaustive).
Recent sampled coverage 2026-06-24T22:41:21Z–2026-09-07T16:54:44Z (74.8 days). Listed workflows 12; observed workflows 9. Selected CI workflow IDs/names: 3526169 (Tests).
Observations per workflow: `{"12530612": 75, "133599238": 22, "247665300": 46, "254573157": 31, "27729629": 46, "2828917": 42, "309211475": 1, "3526169": 50, "72331353": 7}`.
All-attempt terminal categories: `{"action_required": 5, "failure": 70, "skipped": 4, "success": 241}`. Scoped categories: `{"action_required": 1, "failure": 6, "success": 37}`.
Scoped head-SHA clusters 30; largest contains 2 first workflow attempts. PR numbers present 2; PR-attempt rows without a PR number 79.
Change extraction 20/20 complete inventories; verified push diff basis 10/20; all sources were retrieved after their original CI runs.
New runs observed during this session 10; valid preexecution snapshots 0. These are sparse polling rounds, not an estimate of a continuously operated collector’s capture probability.

The largest gap between polling rounds was 12.03 hours. Across the session, 13 additional runs were observed, 3 in the selected first-attempt/event/workflow scope; none was captured before its reported start. This demonstrates failure of this intermittent session to create eligible snapshots, not a general impossibility result for polling.

The three projects qualify for retrospective development diagnostics. None yet qualifies as a verified prospective data source under the frozen contract. Python/JavaScript and different test/job structures provide some heterogeneity, but two Python web libraries and one JavaScript web framework form a narrow convenience sample. Three distinct owner/project IDs are an upper bound on independent pilot clusters, not proof of independence or population coverage. No owner participation has been obtained.

The 60 sampled change inventories have source checksums and deterministic extraction. Collection success and variation are not predictive validation. Samples use SHA-hash order and are not selected by CI label. See feature-feasibility.csv for every feature and repository.
