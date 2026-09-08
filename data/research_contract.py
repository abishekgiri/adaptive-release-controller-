"""V1 research-contract validation; independent of the legacy experiment runners.

No network access, fitting, or experiment execution. Collection adapters must
provide authentic provenance; passing validation cannot authenticate a payload.
"""
from dataclasses import dataclass
from datetime import datetime, timedelta
import math
from typing import Mapping


FEATURE_GROUPS = {
    "files_changed": "change", "additions": "change", "deletions": "change",
    "total_churn": "change", "extension_counts": "change",
    "top_directory_count": "change", "dependency_file_changed": "change",
    "test_file_changed": "change", "configuration_changed": "change",
    "risky_path_changed": "change",
    "project_failure_rate_7d": "history", "project_resolved_count_7d": "history",
    "workflow_failure_rate_7d": "history", "workflow_resolved_count_7d": "history",
    "same_change_prior_failures": "history", "same_change_prior_resolved": "history",
    "prior_workflow_duration_s": "history", "author_failure_rate_90d": "history",
    "author_resolved_count_90d": "history", "repository_commits_30d": "history",
    "history_age_s": "history", "history_left_truncated": "history",
    "workflow_identity": "workflow", "event_type": "workflow",
    "branch_class": "workflow", "attempt_number": "workflow",
    "declared_job_count": "workflow", "declared_runner_types": "workflow",
    "repository_language": "repository", "repository_age_days": "repository",
}
MISSING_REASONS = {"not_captured", "no_history", "incomplete_diff", "unknown_identity",
                   "unresolved_workflow_ref", "dynamic_definition", "not_applicable"}
SOURCE_KINDS = {"metadata", "diff", "workflow_definition", "repository_snapshot", "history"}
CONCLUSIONS = {"success", "failure", "timed_out", "cancelled", "skipped", "neutral",
               "action_required", "stale", "startup_failure", "unknown"}
DEVELOPMENT_ONLY = {"pallets/flask", "psf/requests"}
ACTIONS = ("deploy", "canary", "block")
DEFAULT_COSTS = ((0.0, 10.0), (1.0, 4.0), (2.0, 0.5))


def utc(value: datetime) -> None:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise ValueError("timestamps must be timezone-aware UTC")


@dataclass(frozen=True, order=True)
class AttemptKey:
    repository_id: int
    run_id: int
    attempt: int

    def __post_init__(self):
        if any(type(x) is not int or x <= 0 for x in (self.repository_id, self.run_id, self.attempt)):
            raise ValueError("repository, run and attempt IDs must be positive integers")


@dataclass(frozen=True)
class Decision:
    key: AttemptKey
    workflow_id: int
    head_sha: str
    branch: str
    event_type: str
    created_at: datetime
    decision_at: datetime
    started_at: datetime | None
    source_sha256: str
    protocol_version: str = "ci-design-v1"
    captured_status: str = "queued"


def digest(value: str) -> None:
    if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("source must have a lowercase SHA256 digest")


def validate_decision(d: Decision) -> None:
    for t in (d.created_at, d.decision_at):
        utc(t)
    if d.started_at is not None:
        utc(d.started_at)
    if (d.created_at > d.decision_at or
            (d.started_at is not None and d.decision_at >= d.started_at)):
        raise ValueError("primary decision must precede attempt execution")
    if d.captured_status not in {"requested", "queued", "waiting", "pending"}:
        raise ValueError("decision must be captured from a pre-execution state")
    if type(d.workflow_id) is not int or d.workflow_id <= 0:
        raise ValueError("workflow ID is required")
    if not d.head_sha or not d.branch or d.event_type not in {"push", "pull_request"}:
        raise ValueError("verified change, branch and eligible event are required")
    if d.protocol_version != "ci-design-v1":
        raise ValueError("unsupported protocol version")
    digest(d.source_sha256)


@dataclass(frozen=True)
class FeatureValue:
    name: str
    value: object
    available_at: datetime
    source_kind: str
    source_sha256: str
    source_attempt: AttemptKey | None = None
    missing_reason: str | None = None


def finite_json(value):
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float) and math.isfinite(value):
        return
    if isinstance(value, (list, tuple)):
        for v in value:
            finite_json(v)
        return
    if isinstance(value, dict) and all(isinstance(k, str) for k in value):
        for v in value.values():
            finite_json(v)
        return
    raise ValueError("feature value must be finite JSON data")


def validate_feature_type(f: FeatureValue, d: Decision):
    if f.value is None:
        return
    name, value = f.name, f.value
    count_names = {"files_changed", "additions", "deletions", "total_churn", "top_directory_count",
                   "project_resolved_count_7d", "workflow_resolved_count_7d",
                   "same_change_prior_failures", "same_change_prior_resolved",
                   "author_resolved_count_90d", "repository_commits_30d", "attempt_number",
                   "declared_job_count", "repository_age_days"}
    bool_names = {"dependency_file_changed", "test_file_changed", "configuration_changed",
                  "risky_path_changed", "history_left_truncated"}
    if name in count_names:
        valid = type(value) is int and value >= 0
    elif name in bool_names:
        valid = type(value) is bool
    elif "failure_rate" in name:
        valid = type(value) in (int,float) and 0 <= value <= 1
    elif name in {"prior_workflow_duration_s", "history_age_s"}:
        valid = type(value) in (int,float) and value >= 0
    elif name == "extension_counts":
        valid = isinstance(value,dict) and all(isinstance(k,str) and type(v) is int and v >= 0 for k,v in value.items())
    elif name == "declared_runner_types":
        valid = isinstance(value,(list,tuple)) and all(isinstance(v,str) and v for v in value)
    else:
        valid = isinstance(value,str) and bool(value)
    if not valid:
        raise ValueError("feature type/range violates its definition")
    expected = {"event_type":d.event_type,"attempt_number":d.key.attempt,
                "workflow_identity":f"{d.key.repository_id}:{d.workflow_id}"}
    if name in expected and value != expected[name]:
        raise ValueError("feature metadata disagrees with decision identity")
    if name == "branch_class" and value not in {"default","nondefault","pr"}:
        raise ValueError("invalid branch class")


def validate_features(d: Decision, features: tuple[FeatureValue, ...]) -> None:
    validate_decision(d)
    seen = set()
    for f in features:
        if f.name not in FEATURE_GROUPS or f.name in seen:
            raise ValueError("forbidden or duplicate feature name")
        seen.add(f.name)
        utc(f.available_at)
        digest(f.source_sha256)
        if f.source_kind not in SOURCE_KINDS:
            raise ValueError("postdecision/unverified source kind")
        expected_kind = {"change":"diff","repository":"repository_snapshot"}.get(FEATURE_GROUPS[f.name])
        if f.name in {"declared_job_count","declared_runner_types"}:
            expected_kind = "workflow_definition"
        if f.name in {"workflow_identity","event_type","branch_class","attempt_number"}:
            expected_kind = "metadata"
        # A history masquerade has its own diagnostic below.
        if expected_kind and f.source_kind not in {expected_kind,"history"}:
            raise ValueError("feature source kind violates its definition")
        if f.available_at > d.decision_at:
            raise ValueError("feature source unavailable at decision")
        # Equal-time outcome arrivals are batched AFTER all equal-time decisions.
        if f.source_kind == "history" and f.available_at >= d.decision_at:
            raise ValueError("history must be available strictly before decision")
        if f.source_kind == "history" and FEATURE_GROUPS[f.name] != "history":
            raise ValueError("outcome history cannot masquerade as change context")
        if FEATURE_GROUPS[f.name] == "history" and f.source_kind != "history":
            raise ValueError("history features require history provenance")
        if f.source_attempt is not None:
            if f.source_kind != "history" or f.source_attempt == d.key:
                raise ValueError("current-attempt outcomes are forbidden features")
            if f.source_attempt.repository_id != d.key.repository_id:
                raise ValueError("project-local history cannot cross repositories")
        if f.value is None:
            if f.missing_reason not in MISSING_REASONS:
                raise ValueError("null feature requires explicit missing reason")
        elif f.missing_reason is not None:
            raise ValueError("missing values must not be encoded as zero")
        finite_json(f.value)
        validate_feature_type(f,d)
    values = {f.name:f.value for f in features}
    for rate,count in (("project_failure_rate_7d","project_resolved_count_7d"),
                       ("workflow_failure_rate_7d","workflow_resolved_count_7d"),
                       ("author_failure_rate_90d","author_resolved_count_90d")):
        if count in values and rate in values and values[count] == 0 and values[rate] is not None:
            raise ValueError("empty history must have null risk, not a zero probability")


@dataclass(frozen=True)
class Feedback:
    key: AttemptKey
    conclusion: str
    available_at: datetime
    source_sha256: str
    completed_at: datetime | None = None

    @property
    def binary_label(self):
        return {"success": 0, "failure": 1, "timed_out": 1}.get(self.conclusion)


def validate_feedback(d: Decision, f: Feedback) -> None:
    validate_decision(d)
    utc(f.available_at)
    digest(f.source_sha256)
    if f.key != d.key or f.conclusion not in CONCLUSIONS:
        raise ValueError("feedback must match the exact attempt and a declared conclusion")
    if f.available_at <= (d.started_at or d.decision_at):
        raise ValueError("terminal feedback cannot precede/equal attempt start")
    if f.binary_label is not None and d.started_at is None:
        raise ValueError("binary label requires verified start timestamp")
    if f.completed_at is not None:
        utc(f.completed_at)
        if not (d.started_at or d.decision_at) <= f.completed_at <= f.available_at:
            raise ValueError("completion and availability timestamps are inconsistent")


def loss_vector(f: Feedback, costs=DEFAULT_COSTS):
    if len(costs) != 3 or any(len(row) != 2 for row in costs):
        raise ValueError("cost matrix must contain three actions and two outcomes")
    if any(type(v) not in (int, float) or not math.isfinite(v) or v < 0
           for row in costs for v in row):
        raise ValueError("costs must be finite and nonnegative")
    if f.conclusion not in CONCLUSIONS:
        raise ValueError("unknown raw conclusions must first be mapped to unknown")
    if f.binary_label is None:
        return None
    return dict(zip(ACTIONS, (row[f.binary_label] for row in costs)))


def canonicalize_decisions(decisions):
    unique = {}
    for d in decisions:
        validate_decision(d)
        if d.key in unique and unique[d.key] != d:
            raise ValueError("conflicting duplicate decision; quarantine instead of overwrite")
        unique[d.key] = d
    return tuple(sorted(unique.values(), key=lambda d: (d.decision_at, d.key)))


def canonicalize_feedback(feedback):
    unique = {}
    for f in feedback:
        if f.key in unique:
            old = unique[f.key]
            if old.conclusion != f.conclusion or old.completed_at != f.completed_at:
                raise ValueError("conflicting terminal feedback; requires adjudicated version")
            unique[f.key] = min((old,f),key=lambda x:(x.available_at,x.source_sha256))
        else:
            unique[f.key] = f
    return tuple(unique.values())


def visible_history(d: Decision, decisions, feedback, lookback: timedelta | None = None,
                    *, include_reruns: bool = False):
    validate_decision(d)
    if lookback is not None and lookback <= timedelta(0):
        raise ValueError("lookback must be positive")
    known = {p.key: p for p in canonicalize_decisions(decisions)}
    result = []
    for f in canonicalize_feedback(feedback):
        if f.key not in known:
            raise ValueError("orphan feedback")
        validate_feedback(known[f.key], f)
        if (f.key.repository_id == d.key.repository_id and f.key != d.key
                and (include_reruns or f.key.attempt == 1)
                and f.binary_label is not None and f.available_at < d.decision_at
                and f.available_at <= known[f.key].decision_at + timedelta(days=30)
                and (lookback is None or f.available_at >= d.decision_at-lookback)):
            result.append(f)
    return tuple(sorted(result, key=lambda f: (f.available_at, f.key)))


def evaluation_losses(d: Decision, f: Feedback, cutoff: datetime):
    """Common 30-day resolution limit; never invent losses for unobserved rows."""
    utc(cutoff)
    validate_feedback(d,f)
    if f.available_at > min(cutoff,d.decision_at+timedelta(days=30)):
        return None
    return loss_vector(f)


def validate_splits(rows: list[Mapping]):
    ids, slugs, families = set(), set(), {}
    for row in rows:
        repo, slug, family, split = (row[k] for k in ("repository_id", "slug", "family_id", "split"))
        if type(repo) is not int or repo <= 0 or not slug or not family:
            raise ValueError("stable repository and family identity required")
        if split not in {"development", "validation", "evaluation"}:
            raise ValueError("invalid split")
        if repo in ids or slug.lower() in slugs:
            raise ValueError("repository duplicated across split manifest")
        if slug.lower() in DEVELOPMENT_ONLY and split != "development":
            raise ValueError("existing projects are development only")
        if family in families and families[family] != split:
            raise ValueError("related repository family crosses splits")
        ids.add(repo)
        slugs.add(slug.lower())
        families[family] = split
