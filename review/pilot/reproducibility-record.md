# Pilot reproducibility record

Pilot `ci-design-v1-pilot-1`; frozen parent `ci-design-v1`, digest
`71c8a3d61f377b028cc6d37116cf5900eb180b01e3d44aabbb051775d36146cb`.
The parent documents/schema/tests are unchanged. Pilot artifacts are separate.

`collection-manifest.json` records actual UTC collection boundaries, all queried
endpoints, the three development assignments, API version, source checksums,
source-journal and acquisition-state checksums, collector source hashes,
transformed SQLite/logical-dump hashes, Python/PyYAML/GitHub CLI versions and
platform. The collector's base checkout is recorded, but the collector is
uncommitted: the base commit alone does NOT identify it. Use the recorded source
file hashes. No token or authorization request header is stored. Raw bodies are
public API data and may include public author identities; this pilot has not
established redistribution arrangements for a future corpus.

The collector uses `gh api --method GET` with an explicit development allowlist.
Routes are repository metadata, workflow lists, run lists, exact run attempts,
commit details at a SHA, workflow contents at a SHA, and attempt-specific jobs.
Requests and successful raw-source durability are time-stamped locally. Receipt
is never replaced by commit date, provider updated_at or the last job's end.
Duplicate requests remain in the receipt journal; content-addressed objects are
stored once. Rate limits are bounded with retry/backoff, not bypassed.

Sources reviewed: GitHub documents the filtered run-list limit and attempt
endpoints in the [workflow-run API](https://docs.github.com/en/rest/actions/workflow-runs),
file-pagination limits in the [commit API](https://docs.github.com/en/rest/commits/commits),
and event-specific workflow behavior in [workflow events](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows).
The live requests used API version `2026-03-10`. Endpoint success is not proof of
historical feature availability or of an exhaustive run/diff population.

## Reproduce the exact archived transformation, without network or policies

From the repository root, using Python 3.13 on macOS arm64:

```sh
python3.13 -m venv /private/tmp/ci-pilot-reproduction
/private/tmp/ci-pilot-reproduction/bin/python -m pip install --no-index --find-links review/pilot/wheels --require-hashes -r review/pilot/requirements.lock
/private/tmp/ci-pilot-reproduction/bin/python -m review.pilot.pipeline --output /private/tmp/ci-pilot-reproduced
/private/tmp/ci-pilot-reproduction/bin/python -m review.pilot.report
```

The last command regenerates descriptive reports against the archived pilot in
the repository; it does not collect new data or fit models. The pinned local wheel
is platform-specific. Other platforms need a separately verified PyYAML 6.0.3
wheel and an explicit additional hash; do not disable hash verification. Python,
SQLite and GitHub CLI versions are recorded in the evidence/manifest.

`evidence/fresh-environment-verification.json` records the actual separate-venv
run and matching artifacts. This is fresh dependency isolation on the same host,
not independent-operator or independent-OS reproduction. Archived source bytes
allow deterministic transformation. A later API refetch cannot reproduce the
original actual receipt time, mutable run states, pagination order or retention.

Focused data-only test command:

```sh
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3.13 -m pytest -q tests/test_research_contract.py tests/test_design_gate.py tests/test_pilot_integration.py -p no:cacheprovider
```

The test runner used pytest 9.0.2 as recorded in evidence. Tests require the
archived acquisition files; they neither fetch data nor train any policy.
The 13 integration obligations and remaining work are listed in
`integration-test-status.md` rather than falsely reported as all complete.

## Repeat acquisition as a new diagnostic session

```sh
python3.13 -m review.pilot.collector collect --output /private/tmp/ci-pilot-new-acquisition
python3.13 -m review.pilot.collector poll --output /private/tmp/ci-pilot-new-acquisition
python3.13 -m review.pilot.pipeline --input /private/tmp/ci-pilot-new-acquisition --output /private/tmp/ci-pilot-new-normalized
```

This requires authorized GitHub read access and produces new observations, not
an exact reproduction. The collector refuses to overwrite an existing initial
acquisition. Polling resumes from the saved acquisition state; raw responses
survive failure. This is a bounded read-only pilot collector, not a certified
continuous receiver/service. There is no background scheduler, webhook receiver,
owner installation, operational uptime test, or calibrated polling interval.
New prospective snapshots require a fresh data-quality review; the report script
fails closed rather than recycling this pilot's zero-snapshot verdict.

This session's first two rounds occurred around 05:53–05:59 UTC; the third was
around 18:00 UTC on 7 September 2026. The approximately 12-hour unmonitored gap
is recorded explicitly. No claim of continuous monitoring or validated capture
probability is made. The final verdict is NO-GO for scaling the currently
demonstrated route, with owner-assisted prospective acquisition still unresolved.
