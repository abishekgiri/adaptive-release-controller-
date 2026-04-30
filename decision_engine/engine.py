"""Decision engine that converts risk scores into deployment actions."""

from __future__ import annotations

from dataclasses import asdict, dataclass


DEFAULT_CANARY_THRESHOLD = 0.40
DEFAULT_BLOCK_THRESHOLD = 0.70

DEPLOY = "DEPLOY"
CANARY = "CANARY"
BLOCK = "BLOCK"


@dataclass(frozen=True)
class DecisionResult:
    """Structured deployment decision output."""

    decision: str
    risk_score: float
    reason: str
    thresholds: dict[str, float]

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""

        return asdict(self)


class DecisionEngine:
    """Map normalized deployment risk scores to deterministic actions."""

    def __init__(
        self,
        canary_threshold: float = DEFAULT_CANARY_THRESHOLD,
        block_threshold: float = DEFAULT_BLOCK_THRESHOLD,
    ) -> None:
        validate_thresholds(
            canary_threshold=canary_threshold,
            block_threshold=block_threshold,
        )
        self.canary_threshold = canary_threshold
        self.block_threshold = block_threshold

    def decide(self, risk_score: float) -> DecisionResult:
        """Return a deployment decision for a normalized risk score."""

        validate_risk_score(risk_score)
        normalized_score = round(float(risk_score), 4)
        thresholds = {
            "canary": self.canary_threshold,
            "block": self.block_threshold,
        }

        if normalized_score < self.canary_threshold:
            return DecisionResult(
                decision=DEPLOY,
                risk_score=normalized_score,
                reason=(
                    "Risk score is below the canary threshold; deployment is "
                    "approved."
                ),
                thresholds=thresholds,
            )
        if normalized_score < self.block_threshold:
            return DecisionResult(
                decision=CANARY,
                risk_score=normalized_score,
                reason=(
                    "Risk score is between the canary and block thresholds; "
                    "use a canary rollout."
                ),
                thresholds=thresholds,
            )
        return DecisionResult(
            decision=BLOCK,
            risk_score=normalized_score,
            reason=(
                "Risk score meets or exceeds the block threshold; deployment "
                "should be blocked."
            ),
            thresholds=thresholds,
        )


def decide_deployment(risk_score: float) -> DecisionResult:
    """Return the default deployment decision for a risk score."""

    return DecisionEngine().decide(risk_score)


def validate_risk_score(risk_score: float) -> None:
    """Validate that risk score is normalized."""

    if isinstance(risk_score, bool):
        raise TypeError("risk_score must be a number, not a boolean")
    try:
        value = float(risk_score)
    except (TypeError, ValueError) as error:
        raise TypeError("risk_score must be numeric") from error
    if value < 0.0 or value > 1.0:
        raise ValueError("risk_score must be between 0.0 and 1.0")


def validate_thresholds(canary_threshold: float, block_threshold: float) -> None:
    """Validate configurable decision thresholds."""

    validate_risk_score(canary_threshold)
    validate_risk_score(block_threshold)
    if canary_threshold >= block_threshold:
        raise ValueError("canary_threshold must be lower than block_threshold")

