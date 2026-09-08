-- Proposed normalized SQLite reference schema. No existing data are migrated.
-- UTC times are canonical ISO-8601 with Z, enforced by the Python ingestion gate.
PRAGMA foreign_keys = ON;
CREATE TABLE repositories (
  repository_id INTEGER PRIMARY KEY CHECK(repository_id > 0),
  slug TEXT NOT NULL UNIQUE,
  family_id TEXT NOT NULL,
  split TEXT NOT NULL CHECK(split IN ('development','validation','evaluation')),
  CHECK(lower(slug) NOT IN ('pallets/flask','psf/requests') OR split='development')
) STRICT;
CREATE TABLE source_objects (
  sha256 TEXT PRIMARY KEY CHECK(length(sha256)=64),
  repository_id INTEGER NOT NULL REFERENCES repositories,
  source_kind TEXT NOT NULL,
  received_at TEXT NOT NULL,
  event_time TEXT,
  api_version TEXT,
  delivery_id TEXT,
  storage_uri TEXT NOT NULL,
  complete INTEGER NOT NULL CHECK(complete IN (0,1))
) STRICT;
CREATE TABLE changes (
  repository_id INTEGER NOT NULL REFERENCES repositories,
  head_sha TEXT NOT NULL,
  base_sha TEXT,
  tested_sha TEXT NOT NULL,
  diff_basis TEXT NOT NULL CHECK(diff_basis IN ('first_parent','pr_merge_base','unresolved')),
  author_key TEXT,
  actor_key TEXT,
  source_sha256 TEXT NOT NULL REFERENCES source_objects,
  PRIMARY KEY(repository_id,head_sha,tested_sha)
) STRICT;
CREATE TABLE workflows (
  repository_id INTEGER NOT NULL REFERENCES repositories,
  workflow_id INTEGER NOT NULL,
  path TEXT NOT NULL,
  PRIMARY KEY(repository_id,workflow_id)
) STRICT;
CREATE TABLE workflow_versions (
  repository_id INTEGER NOT NULL,
  workflow_id INTEGER NOT NULL,
  definition_sha TEXT NOT NULL,
  source_sha256 TEXT NOT NULL REFERENCES source_objects,
  resolution_status TEXT NOT NULL CHECK(resolution_status IN ('verified','unresolved')),
  PRIMARY KEY(repository_id,workflow_id,definition_sha),
  FOREIGN KEY(repository_id,workflow_id) REFERENCES workflows
) STRICT;
CREATE TABLE attempts (
  repository_id INTEGER NOT NULL,
  run_id INTEGER NOT NULL CHECK(run_id > 0),
  attempt INTEGER NOT NULL CHECK(attempt > 0),
  workflow_id INTEGER NOT NULL,
  head_sha TEXT NOT NULL,
  tested_sha TEXT NOT NULL,
  branch TEXT NOT NULL,
  event_type TEXT NOT NULL,
  created_at TEXT NOT NULL,
  started_at TEXT,
  source_sha256 TEXT NOT NULL REFERENCES source_objects,
  PRIMARY KEY(repository_id,run_id,attempt),
  FOREIGN KEY(repository_id,workflow_id) REFERENCES workflows,
  FOREIGN KEY(repository_id,head_sha,tested_sha) REFERENCES changes
) STRICT;
CREATE TABLE cost_matrices (
  matrix_id TEXT PRIMARY KEY,
  deploy_success REAL NOT NULL CHECK(deploy_success >= 0),
  deploy_failure REAL NOT NULL CHECK(deploy_failure >= 0),
  canary_success REAL NOT NULL CHECK(canary_success >= 0),
  canary_failure REAL NOT NULL CHECK(canary_failure >= 0),
  block_success REAL NOT NULL CHECK(block_success >= 0),
  block_failure REAL NOT NULL CHECK(block_failure >= 0)
) STRICT;
CREATE TABLE decisions (
  repository_id INTEGER NOT NULL,
  run_id INTEGER NOT NULL,
  attempt INTEGER NOT NULL,
  decision_at TEXT NOT NULL,
  protocol_version TEXT NOT NULL,
  primary_eligible INTEGER NOT NULL CHECK(primary_eligible IN (0,1)),
  exclusion_reason TEXT,
  snapshot_sha256 TEXT NOT NULL REFERENCES source_objects,
  PRIMARY KEY(repository_id,run_id,attempt),
  FOREIGN KEY(repository_id,run_id,attempt) REFERENCES attempts,
  CHECK(primary_eligible=0 OR attempt=1),
  CHECK((primary_eligible=1 AND exclusion_reason IS NULL) OR
        (primary_eligible=0 AND exclusion_reason IS NOT NULL))
) STRICT;
CREATE TABLE feature_definitions (
  feature_name TEXT PRIMARY KEY,
  feature_group TEXT NOT NULL,
  definition_version TEXT NOT NULL,
  value_type TEXT NOT NULL,
  optional INTEGER NOT NULL CHECK(optional IN (0,1))
) STRICT;
CREATE TABLE decision_scenarios (
  repository_id INTEGER NOT NULL,
  run_id INTEGER NOT NULL,
  attempt INTEGER NOT NULL,
  matrix_id TEXT NOT NULL REFERENCES cost_matrices,
  PRIMARY KEY(repository_id,run_id,attempt,matrix_id),
  FOREIGN KEY(repository_id,run_id,attempt) REFERENCES decisions
) STRICT;
CREATE TABLE observed_jobs (
  repository_id INTEGER NOT NULL,
  run_id INTEGER NOT NULL,
  attempt INTEGER NOT NULL,
  job_id INTEGER NOT NULL,
  started_at TEXT,
  completed_at TEXT,
  conclusion TEXT,
  source_sha256 TEXT NOT NULL REFERENCES source_objects,
  PRIMARY KEY(repository_id,run_id,attempt,job_id),
  FOREIGN KEY(repository_id,run_id,attempt) REFERENCES attempts
) STRICT;
CREATE TABLE feature_values (
  repository_id INTEGER NOT NULL,
  run_id INTEGER NOT NULL,
  attempt INTEGER NOT NULL,
  feature_name TEXT NOT NULL REFERENCES feature_definitions,
  value_json TEXT CHECK(value_json IS NULL OR json_valid(value_json)),
  missing_reason TEXT,
  available_at TEXT NOT NULL,
  PRIMARY KEY(repository_id,run_id,attempt,feature_name),
  FOREIGN KEY(repository_id,run_id,attempt) REFERENCES decisions,
  CHECK((value_json IS NULL AND missing_reason IS NOT NULL) OR
        (value_json IS NOT NULL AND value_json != 'null' AND missing_reason IS NULL))
) STRICT;
CREATE TABLE feature_lineage (
  repository_id INTEGER NOT NULL,
  run_id INTEGER NOT NULL,
  attempt INTEGER NOT NULL,
  feature_name TEXT NOT NULL,
  source_sha256 TEXT NOT NULL REFERENCES source_objects,
  PRIMARY KEY(repository_id,run_id,attempt,feature_name,source_sha256),
  FOREIGN KEY(repository_id,run_id,attempt,feature_name) REFERENCES feature_values
) STRICT;
CREATE TABLE feedback (
  repository_id INTEGER NOT NULL,
  run_id INTEGER NOT NULL,
  attempt INTEGER NOT NULL,
  raw_conclusion TEXT NOT NULL,
  conclusion TEXT NOT NULL CHECK(conclusion IN ('success','failure','timed_out','cancelled',
    'skipped','neutral','action_required','stale','startup_failure','unknown')),
  binary_label INTEGER CHECK(binary_label IN (0,1)),
  available_at TEXT NOT NULL,
  completed_at TEXT,
  source_sha256 TEXT NOT NULL REFERENCES source_objects,
  PRIMARY KEY(repository_id,run_id,attempt),
  FOREIGN KEY(repository_id,run_id,attempt) REFERENCES attempts,
  CHECK((conclusion='success' AND binary_label IS 0) OR
        (conclusion IN ('failure','timed_out') AND binary_label IS 1) OR
        (conclusion NOT IN ('success','failure','timed_out') AND binary_label IS NULL))
) STRICT;
CREATE TABLE extraction_runs (
  extraction_id TEXT PRIMARY KEY,
  protocol_sha256 TEXT NOT NULL,
  code_sha256 TEXT NOT NULL,
  input_manifest_sha256 TEXT NOT NULL,
  output_manifest_sha256 TEXT NOT NULL,
  created_at TEXT NOT NULL
) STRICT;
