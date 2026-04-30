"""MAPE-K feedback loop for learning deployment decision policy."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from decision_engine import DecisionEngine
from knowledge_base.db import DEFAULT_DB_PATH, connect, initialize_database


DEFAULT_DEPLOY_THRESHOLD = 0.40
DEFAULT_BLOCK_THRESHOLD = 0.70
DEFAULT_ADJUSTMENT_STEP = 0.05

MIN_DEPLOY_THRESHOLD = 0.20
MAX_DEPLOY_THRESHOLD = 0.60
MIN_BLOCK_THRESHOLD = 0.50
MAX_BLOCK_THRESHOLD = 0.90

FALSE_NEGATIVE_LIMIT = 0.20
FALSE_POSITIVE_LIMIT = 0.30

DEFAULT_POLICY_PATH = Path("experiments/results/learned-policy.json")


@dataclass(frozen=True)
class DeploymentOutcomeRecord:
    """Historical deployment decision joined with its outcome."""

    deployment_id: int
    commit_sha: str
    decision: str
    risk_score: float
    outcome: str


@dataclass(frozen=True)
class FeedbackMetrics:
    """Observed prediction and decision quality metrics."""

    total_records: int
    deployed_or_canaried: int
    blocked: int
    successes: int
    failures: int
    false_positives: int
    false_negatives: int
    success_rate: float
    failure_rate: float
    false_positive_rate: float
    false_negative_rate: float


@dataclass(frozen=True)
class LearnedPolicy:
    """Decision threshold policy learned from feedback."""

    deploy_threshold: float
    block_threshold: float
    previous_deploy_threshold: float
    previous_block_threshold: float
    sensitivity_threshold: float
    adjustment: str
    reason: str
    metrics: FeedbackMetrics

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable policy representation."""

        payload = asdict(self)
        payload["metrics"] = asdict(self.metrics)
        return payload


class FeedbackLoop:
    """MAPE-K feedback loop for adaptive deployment thresholds."""

    def __init__(
        self,
        deploy_threshold: float = DEFAULT_DEPLOY_THRESHOLD,
        block_threshold: float = DEFAULT_BLOCK_THRESHOLD,
        adjustment_step: float = DEFAULT_ADJUSTMENT_STEP,
        sensitivity_threshold: float = FALSE_NEGATIVE_LIMIT,
    ) -> None:
        validate_thresholds(deploy_threshold, block_threshold)
        validate_rate(sensitivity_threshold, "sensitivity_threshold")
        self.deploy_threshold = deploy_threshold
        self.block_threshold = block_threshold
        self.adjustment_step = adjustment_step
        self.sensitivity_threshold = sensitivity_threshold

    def run(self, records: list[DeploymentOutcomeRecord]) -> LearnedPolicy:
        """Analyze historical outcomes and return an adapted policy."""

        metrics = calculate_feedback_metrics(records)
        new_deploy_threshold = self.deploy_threshold
        new_block_threshold = self.block_threshold
        adjustment = "unchanged"
        reason = "Error rates are within acceptable limits."

        if metrics.false_negative_rate > self.sensitivity_threshold:
            new_deploy_threshold = self.deploy_threshold - self.adjustment_step
            new_block_threshold = self.block_threshold - self.adjustment_step
            adjustment = "increase_risk_sensitivity"
            reason = (
                "False negative rate is above the sensitivity threshold; lower "
                "thresholds to make deployment decisions more conservative."
            )
        elif metrics.false_positive_rate > FALSE_POSITIVE_LIMIT:
            new_deploy_threshold = self.deploy_threshold + self.adjustment_step
            new_block_threshold = self.block_threshold + self.adjustment_step
            adjustment = "reduce_unnecessary_blocking"
            reason = (
                "False positive rate is above the limit; raise thresholds to "
                "reduce unnecessary blocking."
            )

        bounded_deploy, bounded_block = apply_threshold_bounds(
            deploy_threshold=new_deploy_threshold,
            block_threshold=new_block_threshold,
        )

        return LearnedPolicy(
            deploy_threshold=bounded_deploy,
            block_threshold=bounded_block,
            previous_deploy_threshold=self.deploy_threshold,
            previous_block_threshold=self.block_threshold,
            sensitivity_threshold=self.sensitivity_threshold,
            adjustment=adjustment,
            reason=reason,
            metrics=metrics,
        )


LearningPolicy = FeedbackLoop


def load_deployment_history(
    db_path: str | Path = DEFAULT_DB_PATH,
    limit: int = 200,
) -> list[DeploymentOutcomeRecord]:
    """Read deployment decisions and outcomes from the knowledge base."""

    initialize_database(db_path)
    with connect(db_path) as connection:
        rows = list(
            connection.execute(
                """
                SELECT id, commit_sha, decision, risk_score, outcome
                FROM deployments
                ORDER BY id ASC
                LIMIT ?
                """,
                (limit,),
            )
        )

    return [
        DeploymentOutcomeRecord(
            deployment_id=int(row["id"]),
            commit_sha=str(row["commit_sha"]),
            decision=normalize_decision(str(row["decision"])),
            risk_score=float(row["risk_score"]),
            outcome=str(row["outcome"]).lower(),
        )
        for row in rows
    ]


def calculate_feedback_metrics(
    records: list[DeploymentOutcomeRecord],
) -> FeedbackMetrics:
    """Compute MAPE-K feedback metrics from historical decisions."""

    total_records = len(records)
    successes = [record for record in records if record.outcome == "success"]
    failures = [record for record in records if record.outcome == "failure"]
    blocked = [record for record in records if record.decision == "BLOCK"]
    deployed_or_canaried = [
        record for record in records if record.decision in {"DEPLOY", "CANARY"}
    ]
    false_positives = [
        record for record in blocked if record.outcome == "success"
    ]
    false_negatives = [
        record for record in deployed_or_canaried if record.outcome == "failure"
    ]

    return FeedbackMetrics(
        total_records=total_records,
        deployed_or_canaried=len(deployed_or_canaried),
        blocked=len(blocked),
        successes=len(successes),
        failures=len(failures),
        false_positives=len(false_positives),
        false_negatives=len(false_negatives),
        success_rate=safe_divide(len(successes), total_records),
        failure_rate=safe_divide(len(failures), total_records),
        false_positive_rate=safe_divide(len(false_positives), total_records),
        false_negative_rate=safe_divide(len(false_negatives), total_records),
    )


def derive_decisions_from_risk_scores(
    records: list[DeploymentOutcomeRecord],
    deploy_threshold: float = DEFAULT_DEPLOY_THRESHOLD,
    block_threshold: float = DEFAULT_BLOCK_THRESHOLD,
) -> list[DeploymentOutcomeRecord]:
    """Recompute decisions from stored risk scores using a threshold policy."""

    engine = DecisionEngine(
        canary_threshold=deploy_threshold,
        block_threshold=block_threshold,
    )
    return [
        DeploymentOutcomeRecord(
            deployment_id=record.deployment_id,
            commit_sha=record.commit_sha,
            decision=engine.decide(record.risk_score).decision,
            risk_score=record.risk_score,
            outcome=record.outcome,
        )
        for record in records
    ]


def save_policy(
    policy: LearnedPolicy,
    path: str | Path = DEFAULT_POLICY_PATH,
) -> None:
    """Persist a learned threshold policy to JSON."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(policy.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def normalize_decision(decision: str) -> str:
    """Normalize historical decision strings."""

    normalized = decision.strip().upper()
    if normalized == "REVIEW":
        return "CANARY"
    if normalized in {"DEPLOY", "CANARY", "BLOCK"}:
        return normalized
    return normalized


def apply_threshold_bounds(
    deploy_threshold: float,
    block_threshold: float,
) -> tuple[float, float]:
    """Clamp thresholds to safe policy bounds."""

    bounded_deploy = clamp(
        deploy_threshold,
        MIN_DEPLOY_THRESHOLD,
        MAX_DEPLOY_THRESHOLD,
    )
    bounded_block = clamp(
        block_threshold,
        MIN_BLOCK_THRESHOLD,
        MAX_BLOCK_THRESHOLD,
    )

    if bounded_deploy >= bounded_block:
        bounded_deploy = max(MIN_DEPLOY_THRESHOLD, bounded_block - 0.01)

    return round(bounded_deploy, 4), round(bounded_block, 4)


def validate_thresholds(deploy_threshold: float, block_threshold: float) -> None:
    """Validate policy thresholds."""

    if not 0.0 <= deploy_threshold <= 1.0:
        raise ValueError("deploy_threshold must be between 0.0 and 1.0")
    if not 0.0 <= block_threshold <= 1.0:
        raise ValueError("block_threshold must be between 0.0 and 1.0")
    if deploy_threshold >= block_threshold:
        raise ValueError("deploy_threshold must be lower than block_threshold")


def validate_rate(value: float, name: str) -> None:
    """Validate a normalized rate."""

    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0.0 and 1.0")


def clamp(value: float, minimum: float, maximum: float) -> float:
    """Clamp a value between two bounds."""

    return max(minimum, min(maximum, value))


def safe_divide(numerator: float, denominator: float) -> float:
    """Divide while returning 0 for empty denominators."""

    if denominator == 0:
        return 0.0
    return round(numerator / denominator, 4)
