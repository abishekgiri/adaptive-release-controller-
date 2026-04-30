"""Heuristic deployment risk model."""

from __future__ import annotations

from dataclasses import dataclass


FILE_COUNT_NORMALIZER = 50
LINES_CHANGED_NORMALIZER = 1500
PAST_FAILURE_NORMALIZER = 10
CI_TIME_NORMALIZER = 3600

LOW_RISK_THRESHOLD = 0.40
HIGH_RISK_THRESHOLD = 0.70


@dataclass(frozen=True)
class RiskInput:
    """Input signals used by the heuristic risk model."""

    commit_sha: str
    files_changed: int
    lines_added: int
    lines_deleted: int
    ci_duration: float
    past_failures: int = 0
    risky_folder_touched: bool = False
    historical_records: int = 0


@dataclass(frozen=True)
class RiskPrediction:
    """Risk model output for one deployment candidate."""

    commit_sha: str
    risk_score: float
    confidence: float
    level: str
    decision: str
    file_count_score: float
    lines_changed_score: float
    past_failure_score: float
    ci_time_score: float


def predict_risk(risk_input: RiskInput) -> RiskPrediction:
    """Predict deployment risk using the Phase 3 heuristic model."""

    file_count_score = normalize(risk_input.files_changed, FILE_COUNT_NORMALIZER)
    lines_changed_score = normalize(
        risk_input.lines_added + risk_input.lines_deleted,
        LINES_CHANGED_NORMALIZER,
    )
    past_failure_score = normalize(
        risk_input.past_failures,
        PAST_FAILURE_NORMALIZER,
    )
    ci_time_score = normalize(risk_input.ci_duration, CI_TIME_NORMALIZER)

    risk_score = round(
        (0.3 * file_count_score)
        + (0.3 * lines_changed_score)
        + (0.2 * past_failure_score)
        + (0.2 * ci_time_score),
        4,
    )
    confidence = calculate_confidence(risk_input=risk_input, risk_score=risk_score)
    level = risk_level(risk_score)

    return RiskPrediction(
        commit_sha=risk_input.commit_sha,
        risk_score=risk_score,
        confidence=confidence,
        level=level,
        decision=risk_decision(risk_score),
        file_count_score=file_count_score,
        lines_changed_score=lines_changed_score,
        past_failure_score=past_failure_score,
        ci_time_score=ci_time_score,
    )


def normalize(value: float, normalizer: float) -> float:
    """Normalize a positive value to the range 0.0-1.0."""

    if normalizer <= 0:
        raise ValueError("normalizer must be greater than 0")
    return round(clamp(float(value) / normalizer, 0.0, 1.0), 4)


def risk_level(risk_score: float) -> str:
    """Return the descriptive risk level for a score."""

    if risk_score >= HIGH_RISK_THRESHOLD:
        return "high"
    if risk_score >= LOW_RISK_THRESHOLD:
        return "medium"
    return "low"


def risk_decision(risk_score: float) -> str:
    """Return the deployment decision implied by a risk score."""

    if risk_score >= HIGH_RISK_THRESHOLD:
        return "block"
    if risk_score >= LOW_RISK_THRESHOLD:
        return "review"
    return "deploy"


def calculate_confidence(risk_input: RiskInput, risk_score: float) -> float:
    """Estimate confidence from signal completeness and threshold distance."""

    completeness = signal_completeness(risk_input)
    history_strength = normalize(risk_input.historical_records, 100)
    threshold_distance = distance_from_nearest_threshold(risk_score)

    confidence = (
        0.45
        + (0.30 * completeness)
        + (0.15 * history_strength)
        + (0.10 * threshold_distance)
    )
    return round(clamp(confidence, 0.0, 1.0), 4)


def signal_completeness(risk_input: RiskInput) -> float:
    """Return the fraction of required signals available."""

    signals = (
        risk_input.files_changed >= 0,
        risk_input.lines_added >= 0,
        risk_input.lines_deleted >= 0,
        risk_input.ci_duration >= 0,
        risk_input.past_failures >= 0,
    )
    return sum(1 for signal in signals if signal) / len(signals)


def distance_from_nearest_threshold(risk_score: float) -> float:
    """Return normalized distance from medium/high decision thresholds."""

    distance = min(
        abs(risk_score - LOW_RISK_THRESHOLD),
        abs(risk_score - HIGH_RISK_THRESHOLD),
    )
    return round(clamp(distance / 0.30, 0.0, 1.0), 4)


def clamp(value: float, minimum: float, maximum: float) -> float:
    """Clamp a numeric value between two bounds."""

    return max(minimum, min(maximum, value))

