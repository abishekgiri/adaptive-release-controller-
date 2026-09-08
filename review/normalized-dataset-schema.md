# Normalized dataset schema

`ci-design-v1`. The executable reference DDL is [design/dataset-schema.sql](design/dataset-schema.sql). It creates an empty SQLite database; it does not migrate old CSVs or collect data. SQLite STRICT tables, foreign keys and CHECK constraints cover structure. Python validators cover aware-UTC timing, feature contracts and cross-record semantics. Source authenticity and complete lineage require the future collector/integration audit.

| Table | Primary key | Purpose / major relations |
|---|---|---|
| repositories | repository_id | Stable slug, family cluster and exactly one split |
| source_objects | SHA256 | Content-addressed raw payload, receipt/event times, access location, API version and completeness |
| changes | repository_id, head_sha, tested_sha | Head versus tested revision, base/diff basis, distinct author and actor keys |
| workflows | repository_id, workflow_id | Stable workflow path/identity |
| workflow_versions | repository_id, workflow_id, definition_sha | Immutable definition and resolution proof |
| attempts | repository_id, run_id, attempt | One actual workflow attempt, linked to workflow and change; branch/event/timestamps |
| decisions | attempt key | One sealed feature snapshot, protocol version, eligibility and exclusion reason |
| cost_matrices | matrix_id | Six explicit nonnegative cost entries; Python also rejects nonfinite values |
| decision_scenarios | attempt key, matrix_id | Scenario membership without duplicating the underlying observation |
| feature_definitions | feature_name | Group, type, definition version, optionality |
| feature_values | attempt key, feature_name | Typed JSON value or SQL null plus reason and source availability |
| feature_lineage | attempt key, feature_name, source SHA256 | Every source dependency, including history source manifests |
| observed_jobs | attempt key, job_id | Post-execution diagnostic records; excluded from current predictors |
| feedback | attempt key | Raw/normalized terminal state, binary label or null, availability/completion and source |
| extraction_runs | extraction_id | Protocol/code/input/output manifest hashes and creation time |

Use canonical UTC strings `YYYY-MM-DDTHH:MM:SS.ffffffZ` in SQLite exports and aware UTC objects in Python. SQL itself does not authenticate timestamps. Text comparisons are not the clock validator. Persist unmodified raw timestamp strings in source objects. `completed_at` is nullable when not authenticated; `available_at` is the collector's real receipt time. UTC conversion must not silently accept naive local time.

Missing feature values use SQL NULL, never string `'null'`, zero, false, NaN or an empty string standing for absence. `missing_reason` is required for null and absent for a real value. The Python allowlist defines admissible names, types, sources and ranges. Extension histograms are JSON objects with integer counts; static runner types are arrays of strings. Feature source lineage is a many-to-many relation; aggregate histories may point to a hashed manifest enumerating their constituent source objects.

Referential checks alone do not ensure that a feature's source belongs to the correct repository/attempt or that its lineage closes. The future adapter must verify each dependency's source identity/time and availability, traversing aggregate manifests. Recomputing the feature from those sources must produce the exact stored value. A collector cannot mark an incomplete diff `complete=1` merely to satisfy a CHECK constraint.

For model input export, join `decisions` to allowed `feature_values` and validated source lineage only. Never join `feedback` or current `observed_jobs` into the predictor view. A separate delayed-feedback interface returns label/loss packets after availability. The same raw corpus supports multiple cost scenarios through an explicit link; analysis must not inflate sample size by counting scenario replicas as new independent data.

Raw updates and duplicate deliveries remain append-only objects. Normalized decision conflicts are quarantined; terminal redelivery preserves earliest authenticated availability, while conflicting conclusions/completion semantics require adjudication/versioning. Releasing a new normalized version never overwrites an old data release. Keep raw payloads and any identity mapping under their declared access policy; derived reproducible pseudonymous releases need their own manifests.

Future tables for model predictions and tuning results are intentionally absent: no learning method is being executed in this phase. A future prediction row must key to the exact decision, model/encoder/protocol/matrix/seed versions and training cutoff; that implementation is an outstanding integration invariant.
