# Development pilot selection, recorded before repository collection

Pilot `ci-design-v1-pilot-1`; parent protocol remains frozen and NO-GO for models.

Selection rule: use the two existing development projects, `pallets/flask` and
`psf/requests`, plus one explicitly assigned development project,
`expressjs/express`, to probe a second language/toolchain. No validation or
evaluation repository is queried. Express is consumed as development data by
this assignment and cannot later be presented as untouched evaluation data.

Eligibility checks, specified before collection: public, not archived or a mirror;
accessible Actions build/test workflows; evidence of CI activity at least six
months ago; recent push/PR activity; retrievable immutable commit sources.
Record success/failure counts as feasibility evidence, not a selection target.
Do not replace a selected repository if it has no failures or weak features.
Retain failures of eligibility/access in the attrition ledger. These are a
purposive convenience sample, not a random or population-representative sample.

Workflow inclusion is by CI purpose before inspecting outcome associations:
names/paths containing CI, test, build, coverage, or integration, excluding
release, publish, deployment, documentation-only, labeling and security-only
workflows. Keep every observed workflow in diagnostics; record selection
explicitly and manually verify definitions. Primary events remain push and
pull_request and primary training/scoring remain first-attempt only.

Bounded pilot allocation: up to 300 recent run summaries per project at a fixed
request-time upper boundary; expand every observed rerun to attempt-specific
records; inspect at most 20 distinct changes per project in a deterministic
SHA-hash order, independent of outcomes. These bounded diagnostic samples are
not complete historical cohorts. Verify one older historical run per project
for the six-month activity screen. Record live polling observations separately,
with actual receipt times; never manufacture historical decision timestamps.

The polling window is a finite feasibility session, not a six-month study or
a test of all proposed pilot coverage thresholds. It may expose no new eligible
runs. No event is triggered, no repository changed, no owner contacted, and no
webhook installed. API-access feasibility does not establish owner cooperation
or the number of independently recruitable evaluation projects.

Repository-specific qualification evidence and all exclusion/coverage counts
will be written to `data-quality-report.md` and `collection-manifest.json`.
