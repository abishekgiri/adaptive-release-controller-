"""Evaluate deterministic Phase 4 decision engine behavior."""

from __future__ import annotations

import argparse
from typing import Any

from decision_engine import DecisionEngine, DecisionResult


DEFAULT_SCORES = (0.25, 0.40, 0.55, 0.70, 0.85)


def evaluate_decisions(
    scores: tuple[float, ...] = DEFAULT_SCORES,
    canary_threshold: float = 0.40,
    block_threshold: float = 0.70,
) -> list[DecisionResult]:
    """Evaluate deployment decisions for a sequence of scores."""

    engine = DecisionEngine(
        canary_threshold=canary_threshold,
        block_threshold=block_threshold,
    )
    return [engine.decide(score) for score in scores]


def validate_expected_decisions() -> None:
    """Prove required examples and boundary behavior are deterministic."""

    engine = DecisionEngine()
    expected = {
        0.25: "DEPLOY",
        0.40: "CANARY",
        0.55: "CANARY",
        0.70: "BLOCK",
        0.85: "BLOCK",
    }
    for score, decision in expected.items():
        actual = engine.decide(score).decision
        if actual != decision:
            raise AssertionError(
                f"risk_score={score} expected {decision}, got {actual}"
            )


def results_markdown(results: list[DecisionResult]) -> str:
    """Format decision results as a Markdown table."""

    rows = [
        (
            f"{result.risk_score:.2f}",
            result.decision,
            result.thresholds["canary"],
            result.thresholds["block"],
            result.reason,
        )
        for result in results
    ]
    return markdown_table(
        ("Risk Score", "Decision", "Canary Threshold", "Block Threshold", "Reason"),
        rows,
    )


def markdown_table(headers: tuple[str, ...], rows: list[tuple[Any, ...]]) -> str:
    """Build a Markdown table."""

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(description="Evaluate decision engine outputs.")
    parser.add_argument(
        "--scores",
        nargs="*",
        type=float,
        default=list(DEFAULT_SCORES),
        help="Risk scores to evaluate.",
    )
    parser.add_argument(
        "--canary-threshold",
        type=float,
        default=0.40,
        help="Minimum risk score that triggers CANARY.",
    )
    parser.add_argument(
        "--block-threshold",
        type=float,
        default=0.70,
        help="Minimum risk score that triggers BLOCK.",
    )
    return parser.parse_args()


def main() -> None:
    """Run decision evaluation and print deterministic result table."""

    args = parse_args()
    validate_expected_decisions()
    results = evaluate_decisions(
        scores=tuple(args.scores),
        canary_threshold=args.canary_threshold,
        block_threshold=args.block_threshold,
    )

    print("# Decision Engine Results\n")
    print(results_markdown(results))


if __name__ == "__main__":
    main()

