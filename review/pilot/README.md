# Data-only development feasibility pilot

**Final decision: NO-GO for scaling the currently demonstrated collection route.**
No policy models were trained and the frozen parent protocol/manuscript were not modified.

Start with [data feasibility](data-feasibility-report.md), then inspect:

- [Precollection selection rules](repository-selection.md)
- [Every-feature feasibility CSV](feature-feasibility.csv)
- [Data quality and identity/timing coverage](data-quality-report.md)
- [Precision scenarios and uncertainty](precision-scenarios.md)
- [Reproducibility record](reproducibility-record.md)
- [Machine-readable collection manifest](collection-manifest.json)
- [Normalized schema validation](schema-validation.json)
- [Integration-test scope and outstanding work](integration-test-status.md)
- [Test execution results](evidence/integration-tests.txt)
- [Fresh-environment evidence](evidence/fresh-environment-verification.json)

The normalized SQLite database is `pilot.sqlite`; `dataset.sql` is its logical
reproduction. Raw response bodies live in `raw/`, with every successful/error
receipt and endpoint in `source-journal.jsonl`. `attempt-diagnostics.json` retains
observations that cannot honestly satisfy the normalized identity contract.
`feature-diagnostics.json` contains explicitly retrospective extraction only.
`predictor-export.json` is empty because no genuine predecision instance was
certified. Raw response hashes do not authenticate historical availability.

The supported collector/transformer is a bounded read-only pilot, not a deployed
continuous receiver. Positive prospective PR capture, actual execution-boundary
semantics and learner/runner integration are not certified. Do not use the pilot
row count, retrospective variation, or passing fixture tests as a substitute.
