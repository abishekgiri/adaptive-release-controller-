"""Compare static, risk-only, and adaptive deployment controllers."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from decision_engine import DecisionEngine
from experiments.baseline import baseline_decision, from_mapping
from knowledge_base.db import DEFAULT_DB_PATH, connect, initialize_database


DEFAULT_RESULTS_DIR = Path("experiments/results")
DEFAULT_GRAPHS_DIR = DEFAULT_RESULTS_DIR / "graphs"
DEFAULT_POLICY_PATH = DEFAULT_RESULTS_DIR / "learned-policy.json"
DEFAULT_MARKDOWN_PATH = DEFAULT_RESULTS_DIR / "evaluation-results.md"
DEFAULT_JSON_PATH = DEFAULT_RESULTS_DIR / "evaluation-results.json"

DEFAULT_DEPLOY_THRESHOLD = 0.40
DEFAULT_BLOCK_THRESHOLD = 0.70

DEPLOY = "DEPLOY"
CANARY = "CANARY"
BLOCK = "BLOCK"
SYSTEMS = ("Static", "Risk-only", "Adaptive")


@dataclass(frozen=True)
class DeploymentEvaluationRecord:
    """Normalized deployment record for cross-system evaluation."""

    deployment_id: int
    commit_sha: str
    test_passed: bool
    coverage: float
    risk_score: float
    outcome: str
    recovery_time: float | None = None


@dataclass(frozen=True)
class SystemDecision:
    """One system decision for one deployment record."""

    system: str
    commit_sha: str
    risk_score: float
    decision: str
    outcome: str
    mttr: float
    correct: bool


@dataclass(frozen=True)
class SystemMetrics:
    """Comparable evaluation metrics for a deployment controller."""

    system: str
    total_records: int
    deployed_or_canaried: int
    blocked: int
    deployed_or_canary_successes: int
    deployed_or_canary_failures: int
    success_rate: float
    failure_rate: float
    mttr: float
    false_positives: int
    false_negatives: int
    false_positive_rate: float
    false_negative_rate: float
    decision_accuracy: float
    decision_distribution: dict[str, int]


@dataclass(frozen=True)
class PolicyConfig:
    """Threshold configuration used by the adaptive evaluator."""

    deploy_threshold: float
    block_threshold: float
    source: str
    previous_deploy_threshold: float = DEFAULT_DEPLOY_THRESHOLD
    previous_block_threshold: float = DEFAULT_BLOCK_THRESHOLD
    sensitivity_threshold: float | None = None
    adjustment: str = "default_policy"


def load_records(
    db_path: str | Path = DEFAULT_DB_PATH,
    limit: int = 200,
) -> list[DeploymentEvaluationRecord]:
    """Load deployment records from the knowledge base."""

    initialize_database(db_path)
    with connect(db_path) as connection:
        rows = list(
            connection.execute(
                """
                SELECT *
                FROM deployments
                ORDER BY id ASC
                LIMIT ?
                """,
                (limit,),
            )
        )

    records = []
    for row in rows:
        baseline_input = from_mapping(row)
        records.append(
            DeploymentEvaluationRecord(
                deployment_id=int(row["id"]),
                commit_sha=str(row["commit_sha"]),
                test_passed=baseline_input.tests_passed,
                coverage=baseline_input.coverage,
                risk_score=float(row["risk_score"]),
                outcome=str(row["outcome"]).lower(),
                recovery_time=get_optional_recovery_time(row),
            )
        )
    return records


def load_policy_config(
    policy_path: str | Path = DEFAULT_POLICY_PATH,
    use_adaptive_policy: bool = False,
) -> PolicyConfig:
    """Load learned thresholds with a safe default fallback."""

    if not use_adaptive_policy:
        return PolicyConfig(
            deploy_threshold=DEFAULT_DEPLOY_THRESHOLD,
            block_threshold=DEFAULT_BLOCK_THRESHOLD,
            source="default_policy",
        )

    path = Path(policy_path)
    if not path.exists():
        return PolicyConfig(
            deploy_threshold=DEFAULT_DEPLOY_THRESHOLD,
            block_threshold=DEFAULT_BLOCK_THRESHOLD,
            source="default_fallback",
        )

    payload = json.loads(path.read_text(encoding="utf-8"))
    return PolicyConfig(
        deploy_threshold=float(payload.get("deploy_threshold", DEFAULT_DEPLOY_THRESHOLD)),
        block_threshold=float(payload.get("block_threshold", DEFAULT_BLOCK_THRESHOLD)),
        previous_deploy_threshold=float(
            payload.get("previous_deploy_threshold", DEFAULT_DEPLOY_THRESHOLD)
        ),
        previous_block_threshold=float(
            payload.get("previous_block_threshold", DEFAULT_BLOCK_THRESHOLD)
        ),
        sensitivity_threshold=payload.get("sensitivity_threshold"),
        adjustment=str(payload.get("adjustment", "learned_policy")),
        source="learned_policy",
    )


def evaluate_all_systems(
    records: list[DeploymentEvaluationRecord],
    deploy_threshold: float,
    block_threshold: float,
) -> tuple[dict[str, list[SystemDecision]], dict[str, SystemMetrics]]:
    """Evaluate static, risk-only, and adaptive systems."""

    decisions = {
        "Static": evaluate_static(records),
        "Risk-only": evaluate_threshold_policy(
            system="Risk-only",
            records=records,
            deploy_threshold=DEFAULT_DEPLOY_THRESHOLD,
            block_threshold=DEFAULT_BLOCK_THRESHOLD,
        ),
        "Adaptive": evaluate_threshold_policy(
            system="Adaptive",
            records=records,
            deploy_threshold=deploy_threshold,
            block_threshold=block_threshold,
        ),
    }
    metrics = {
        system: calculate_system_metrics(system, system_decisions)
        for system, system_decisions in decisions.items()
    }
    return decisions, metrics


def evaluate_static(
    records: list[DeploymentEvaluationRecord],
) -> list[SystemDecision]:
    """Evaluate the static CI/CD baseline."""

    decisions = []
    for record in records:
        decision = baseline_decision(record.test_passed, record.coverage).upper()
        decisions.append(
            make_system_decision(
                system="Static",
                record=record,
                decision=decision,
            )
        )
    return decisions


def evaluate_threshold_policy(
    system: str,
    records: list[DeploymentEvaluationRecord],
    deploy_threshold: float,
    block_threshold: float,
) -> list[SystemDecision]:
    """Evaluate a risk-threshold deployment policy."""

    engine = DecisionEngine(
        canary_threshold=deploy_threshold,
        block_threshold=block_threshold,
    )
    return [
        make_system_decision(
            system=system,
            record=record,
            decision=engine.decide(record.risk_score).decision,
        )
        for record in records
    ]


def make_system_decision(
    system: str,
    record: DeploymentEvaluationRecord,
    decision: str,
) -> SystemDecision:
    """Create a normalized decision evaluation row."""

    return SystemDecision(
        system=system,
        commit_sha=record.commit_sha,
        risk_score=record.risk_score,
        decision=decision,
        outcome=record.outcome,
        mttr=deployment_mttr(
            decision=decision,
            outcome=record.outcome,
            recovery_time=record.recovery_time,
        ),
        correct=is_correct_decision(decision=decision, outcome=record.outcome),
    )


def calculate_system_metrics(
    system: str,
    decisions: list[SystemDecision],
) -> SystemMetrics:
    """Calculate evaluation metrics for one system."""

    total_records = len(decisions)
    allowed = [item for item in decisions if item.decision in {DEPLOY, CANARY}]
    blocked = [item for item in decisions if item.decision == BLOCK]
    allowed_successes = [item for item in allowed if item.outcome == "success"]
    allowed_failures = [item for item in allowed if item.outcome == "failure"]
    false_positives = [item for item in blocked if item.outcome == "success"]
    false_negatives = [item for item in allowed if item.outcome == "failure"]
    correct = [item for item in decisions if item.correct]

    return SystemMetrics(
        system=system,
        total_records=total_records,
        deployed_or_canaried=len(allowed),
        blocked=len(blocked),
        deployed_or_canary_successes=len(allowed_successes),
        deployed_or_canary_failures=len(allowed_failures),
        success_rate=safe_divide(len(allowed_successes), len(allowed)),
        failure_rate=safe_divide(len(allowed_failures), len(allowed)),
        mttr=safe_divide(sum(item.mttr for item in allowed_failures), len(allowed_failures)),
        false_positives=len(false_positives),
        false_negatives=len(false_negatives),
        false_positive_rate=safe_divide(len(false_positives), total_records),
        false_negative_rate=safe_divide(len(false_negatives), total_records),
        decision_accuracy=safe_divide(len(correct), total_records),
        decision_distribution={
            DEPLOY: sum(1 for item in decisions if item.decision == DEPLOY),
            CANARY: sum(1 for item in decisions if item.decision == CANARY),
            BLOCK: sum(1 for item in decisions if item.decision == BLOCK),
        },
    )


def deployment_mttr(
    decision: str,
    outcome: str,
    recovery_time: float | None = None,
) -> float:
    """Return MTTR using the Phase 6 fallback simulation."""

    if decision == BLOCK:
        return 0.0
    if outcome != "failure":
        return 0.0
    if recovery_time is not None:
        return recovery_time
    if decision == CANARY:
        return 30.0
    return 60.0


def is_correct_decision(decision: str, outcome: str) -> bool:
    """Return whether a decision matches the observed deployment outcome."""

    if outcome == "success":
        return decision in {DEPLOY, CANARY}
    if outcome == "failure":
        return decision == BLOCK
    return False


def write_outputs(
    metrics: dict[str, SystemMetrics],
    decisions: dict[str, list[SystemDecision]],
    policy_config: PolicyConfig,
    markdown_path: str | Path = DEFAULT_MARKDOWN_PATH,
    json_path: str | Path = DEFAULT_JSON_PATH,
    graphs_dir: str | Path = DEFAULT_GRAPHS_DIR,
) -> None:
    """Write Markdown, JSON, and graph outputs."""

    markdown = evaluation_markdown(
        metrics=metrics,
        policy_config=policy_config,
    )
    Path(markdown_path).write_text(markdown, encoding="utf-8")

    payload = {
        "policy": asdict(policy_config),
        "metrics": {
            system: asdict(system_metrics)
            for system, system_metrics in metrics.items()
        },
        "adaptive_behavior": adaptive_behavior_payload(metrics, policy_config),
        "decisions_sample": {
            system: [asdict(item) for item in system_decisions[:20]]
            for system, system_decisions in decisions.items()
        },
    }
    Path(json_path).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    generate_graphs(metrics=metrics, graphs_dir=graphs_dir)


def evaluation_markdown(
    metrics: dict[str, SystemMetrics],
    policy_config: PolicyConfig,
) -> str:
    """Build the Markdown evaluation report."""

    lines = [
        "# Experimentation & Evaluation Results",
        "",
        "## Experiment Setup",
        "",
        markdown_table(
            ("Item", "Value"),
            [
                ("Phase", "Phase 6 - Experimentation & Evaluation"),
                ("Dataset", "`knowledge_base/deployments.db`"),
                ("Dataset size", next(iter(metrics.values())).total_records if metrics else 0),
                ("Adaptive policy source", policy_config.source),
                ("Adaptive deploy threshold", f"{policy_config.deploy_threshold:.2f}"),
                ("Adaptive block threshold", f"{policy_config.block_threshold:.2f}"),
            ],
        ),
        "",
        "## System Comparison",
        "",
        metrics_markdown(metrics),
        "",
        "## Decision Distribution",
        "",
        decision_distribution_markdown(metrics),
        "",
        "## Adaptive Behavior Under Increased Sensitivity",
        "",
        adaptive_behavior_markdown(metrics, policy_config),
        "",
        "## Graph Outputs",
        "",
        markdown_table(
            ("Graph", "Path"),
            [
                ("Success rate comparison", "`experiments/results/graphs/success_rate_comparison.png`"),
                ("Failure rate comparison", "`experiments/results/graphs/failure_rate_comparison.png`"),
                ("False positive vs false negative comparison", "`experiments/results/graphs/error_rate_comparison.png`"),
                ("MTTR comparison", "`experiments/results/graphs/mttr_comparison.png`"),
                ("Decision distribution comparison", "`experiments/results/graphs/decision_distribution_comparison.png`"),
                ("Adaptive improvement comparison", "`experiments/results/graphs/adaptive_improvement_comparison.png`"),
            ],
        ),
        "",
        "## Research Interpretation",
        "",
        research_interpretation(
            metrics=metrics,
            policy_config=policy_config,
        ),
        "",
    ]
    return "\n".join(lines)


def metrics_markdown(metrics: dict[str, SystemMetrics]) -> str:
    """Format comparable system metrics."""

    rows = []
    for system in SYSTEMS:
        item = metrics[system]
        rows.append(
            (
                system,
                format_percent(item.success_rate),
                format_percent(item.failure_rate),
                f"{item.mttr:.2f}",
                format_percent(item.false_positive_rate),
                format_percent(item.false_negative_rate),
                format_percent(item.decision_accuracy),
            )
        )
    return markdown_table(
        (
            "System",
            "Success Rate",
            "Failure Rate",
            "MTTR",
            "False Positive Rate",
            "False Negative Rate",
            "Decision Accuracy",
        ),
        rows,
    )


def decision_distribution_markdown(metrics: dict[str, SystemMetrics]) -> str:
    """Format decision distribution table."""

    rows = []
    for system in SYSTEMS:
        distribution = metrics[system].decision_distribution
        rows.append(
            (
                system,
                distribution[DEPLOY],
                distribution[CANARY],
                distribution[BLOCK],
            )
        )
    return markdown_table(("System", "DEPLOY", "CANARY", "BLOCK"), rows)


def research_interpretation(
    metrics: dict[str, SystemMetrics],
    policy_config: PolicyConfig,
) -> str:
    """Explain whether adaptive improved over static baseline."""

    static = metrics["Static"]
    adaptive = metrics["Adaptive"]
    risk_only = metrics["Risk-only"]

    improvements = []
    if adaptive.failure_rate < static.failure_rate:
        improvements.append("lower failure rate")
    if adaptive.mttr < static.mttr:
        improvements.append("lower MTTR")
    if adaptive.false_negative_rate < static.false_negative_rate:
        improvements.append("lower false negative rate")
    if adaptive.decision_accuracy > static.decision_accuracy:
        improvements.append("higher decision accuracy")

    if improvements:
        first_sentence = (
            "The adaptive controller improved over the static baseline on "
            + ", ".join(improvements)
            + "."
        )
    else:
        first_sentence = (
            "The adaptive controller did not improve over the static baseline "
            "on the primary reliability metrics in this run."
        )

    if policy_config.source != "learned_policy":
        second_sentence = (
            "The adaptive system used default thresholds because learned policy "
            "loading was not enabled for this run."
        )
    elif (
        policy_config.deploy_threshold == DEFAULT_DEPLOY_THRESHOLD
        and policy_config.block_threshold == DEFAULT_BLOCK_THRESHOLD
    ):
        second_sentence = (
            "The adaptive and risk-only systems match because the learned policy "
            "kept the default thresholds unchanged for this dataset."
        )
    else:
        second_sentence = (
            "The adaptive system differs from risk-only because it uses the "
            "learned MAPE-K thresholds."
        )

    if adaptive.failure_rate < risk_only.failure_rate:
        third_sentence = (
            "Compared with risk-only control, the adaptive policy reduces failure "
            f"rate from {format_percent(risk_only.failure_rate)} to "
            f"{format_percent(adaptive.failure_rate)} by blocking more risky "
            "deployments."
        )
    else:
        third_sentence = (
            "Compared with risk-only control, the adaptive policy does not reduce "
            "failure rate in this run."
        )

    return (
        f"{first_sentence} {second_sentence} {third_sentence} This provides the "
        "Phase 6 comparison needed to evaluate whether feedback-adjusted deployment "
        "control improves reliability compared to static CI/CD."
    )


def adaptive_behavior_markdown(
    metrics: dict[str, SystemMetrics],
    policy_config: PolicyConfig,
) -> str:
    """Format the adaptive threshold change and tradeoff."""

    risk_only = metrics["Risk-only"]
    adaptive = metrics["Adaptive"]
    rows = [
        ("Deploy threshold before", f"{policy_config.previous_deploy_threshold:.2f}"),
        ("Deploy threshold after", f"{policy_config.deploy_threshold:.2f}"),
        ("Block threshold before", f"{policy_config.previous_block_threshold:.2f}"),
        ("Block threshold after", f"{policy_config.block_threshold:.2f}"),
        ("Sensitivity threshold", format_optional_rate(policy_config.sensitivity_threshold)),
        ("Adjustment", policy_config.adjustment),
        (
            "Risk-only failure rate",
            format_percent(risk_only.failure_rate),
        ),
        (
            "Adaptive failure rate",
            format_percent(adaptive.failure_rate),
        ),
        (
            "Risk-only decision accuracy",
            format_percent(risk_only.decision_accuracy),
        ),
        (
            "Adaptive decision accuracy",
            format_percent(adaptive.decision_accuracy),
        ),
    ]
    return markdown_table(("Field", "Value"), rows)


def adaptive_behavior_payload(
    metrics: dict[str, SystemMetrics],
    policy_config: PolicyConfig,
) -> dict[str, Any]:
    """Return machine-readable adaptive behavior summary."""

    return {
        "policy": asdict(policy_config),
        "risk_only_failure_rate": metrics["Risk-only"].failure_rate,
        "adaptive_failure_rate": metrics["Adaptive"].failure_rate,
        "risk_only_accuracy": metrics["Risk-only"].decision_accuracy,
        "adaptive_accuracy": metrics["Adaptive"].decision_accuracy,
        "adaptive_differs_from_risk_only": (
            metrics["Adaptive"].decision_distribution
            != metrics["Risk-only"].decision_distribution
        ),
    }


def format_optional_rate(value: float | None) -> str:
    """Format an optional normalized rate."""

    if value is None:
        return "n/a"
    return format_percent(float(value))


def get_optional_recovery_time(row: Any) -> float | None:
    """Return existing MTTR/recovery time if the dataset includes one."""

    keys = row.keys() if hasattr(row, "keys") else row
    for name in ("mttr", "recovery_time"):
        if name in keys and row[name] is not None:
            return float(row[name])
    return None


def generate_graphs(
    metrics: dict[str, SystemMetrics],
    graphs_dir: str | Path = DEFAULT_GRAPHS_DIR,
) -> None:
    """Generate required matplotlib comparison graphs."""

    output_dir = Path(graphs_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    bar_chart(
        title="Success Rate Comparison",
        ylabel="Success Rate",
        values=[metrics[system].success_rate for system in SYSTEMS],
        output_path=output_dir / "success_rate_comparison.png",
        as_percent=True,
    )
    bar_chart(
        title="Failure Rate Comparison",
        ylabel="Failure Rate",
        values=[metrics[system].failure_rate for system in SYSTEMS],
        output_path=output_dir / "failure_rate_comparison.png",
        as_percent=True,
    )
    grouped_error_chart(
        metrics=metrics,
        output_path=output_dir / "error_rate_comparison.png",
    )
    bar_chart(
        title="MTTR Comparison",
        ylabel="Minutes",
        values=[metrics[system].mttr for system in SYSTEMS],
        output_path=output_dir / "mttr_comparison.png",
        as_percent=False,
    )
    decision_distribution_chart(
        metrics=metrics,
        output_path=output_dir / "decision_distribution_comparison.png",
    )
    adaptive_improvement_chart(
        metrics=metrics,
        output_path=output_dir / "adaptive_improvement_comparison.png",
    )


def bar_chart(
    title: str,
    ylabel: str,
    values: list[float],
    output_path: Path,
    as_percent: bool,
) -> None:
    """Create a simple bar chart."""

    fig, ax = plt.subplots(figsize=(8, 5))
    chart_values = [value * 100 for value in values] if as_percent else values
    ax.bar(SYSTEMS, chart_values, color=["#4c78a8", "#f58518", "#54a24b"])
    ax.set_title(title)
    ax.set_ylabel(ylabel + (" (%)" if as_percent else ""))
    ax.set_ylim(bottom=0)
    for index, value in enumerate(chart_values):
        label = f"{value:.2f}%" if as_percent else f"{value:.2f}"
        ax.text(index, value, label, ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def grouped_error_chart(
    metrics: dict[str, SystemMetrics],
    output_path: Path,
) -> None:
    """Create false positive vs false negative comparison chart."""

    fig, ax = plt.subplots(figsize=(8, 5))
    positions = range(len(SYSTEMS))
    width = 0.35
    false_positive_values = [
        metrics[system].false_positive_rate * 100 for system in SYSTEMS
    ]
    false_negative_values = [
        metrics[system].false_negative_rate * 100 for system in SYSTEMS
    ]
    ax.bar(
        [position - width / 2 for position in positions],
        false_positive_values,
        width,
        label="False Positive Rate",
        color="#e45756",
    )
    ax.bar(
        [position + width / 2 for position in positions],
        false_negative_values,
        width,
        label="False Negative Rate",
        color="#72b7b2",
    )
    ax.set_title("False Positive vs False Negative Comparison")
    ax.set_ylabel("Rate (%)")
    ax.set_xticks(list(positions))
    ax.set_xticklabels(SYSTEMS)
    ax.legend()
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def decision_distribution_chart(
    metrics: dict[str, SystemMetrics],
    output_path: Path,
) -> None:
    """Create grouped decision distribution chart."""

    fig, ax = plt.subplots(figsize=(9, 5))
    positions = range(len(SYSTEMS))
    width = 0.25
    deploy_values = [metrics[system].decision_distribution[DEPLOY] for system in SYSTEMS]
    canary_values = [metrics[system].decision_distribution[CANARY] for system in SYSTEMS]
    block_values = [metrics[system].decision_distribution[BLOCK] for system in SYSTEMS]
    ax.bar(
        [position - width for position in positions],
        deploy_values,
        width,
        label=DEPLOY,
        color="#4c78a8",
    )
    ax.bar(
        list(positions),
        canary_values,
        width,
        label=CANARY,
        color="#f58518",
    )
    ax.bar(
        [position + width for position in positions],
        block_values,
        width,
        label=BLOCK,
        color="#54a24b",
    )
    ax.set_title("Decision Distribution Comparison")
    ax.set_ylabel("Record Count")
    ax.set_xticks(list(positions))
    ax.set_xticklabels(SYSTEMS)
    ax.legend()
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def adaptive_improvement_chart(
    metrics: dict[str, SystemMetrics],
    output_path: Path,
) -> None:
    """Create focused risk-only vs adaptive tradeoff chart."""

    fig, ax = plt.subplots(figsize=(8, 5))
    labels = ("Failure Rate", "False Negative Rate", "False Positive Rate")
    risk_only_values = [
        metrics["Risk-only"].failure_rate * 100,
        metrics["Risk-only"].false_negative_rate * 100,
        metrics["Risk-only"].false_positive_rate * 100,
    ]
    adaptive_values = [
        metrics["Adaptive"].failure_rate * 100,
        metrics["Adaptive"].false_negative_rate * 100,
        metrics["Adaptive"].false_positive_rate * 100,
    ]
    positions = range(len(labels))
    width = 0.35
    ax.bar(
        [position - width / 2 for position in positions],
        risk_only_values,
        width,
        label="Risk-only",
        color="#f58518",
    )
    ax.bar(
        [position + width / 2 for position in positions],
        adaptive_values,
        width,
        label="Adaptive",
        color="#54a24b",
    )
    ax.set_title("Adaptive Improvement Comparison")
    ax.set_ylabel("Rate (%)")
    ax.set_xticks(list(positions))
    ax.set_xticklabels(labels)
    ax.legend()
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def markdown_table(headers: tuple[str, ...], rows: list[tuple[Any, ...]]) -> str:
    """Build a Markdown table."""

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def format_percent(value: float) -> str:
    """Format a ratio as a percentage."""

    return f"{value * 100:.2f}%"


def safe_divide(numerator: float, denominator: float) -> float:
    """Divide while returning 0 for empty denominators."""

    if denominator == 0:
        return 0.0
    return round(numerator / denominator, 4)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(
        description="Compare static, risk-only, and adaptive deployment control."
    )
    parser.add_argument(
        "--db",
        default=str(DEFAULT_DB_PATH),
        help="Path to the SQLite deployment database.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=200,
        help="Maximum deployment records to evaluate.",
    )
    parser.add_argument(
        "--policy",
        default=str(DEFAULT_POLICY_PATH),
        help="Path to learned policy JSON.",
    )
    parser.add_argument(
        "--use-adaptive-policy",
        action="store_true",
        help="Use thresholds from learned-policy.json for the adaptive system.",
    )
    return parser.parse_args()


def main() -> None:
    """Run Phase 6 evaluation and save all outputs."""

    args = parse_args()
    records = load_records(db_path=args.db, limit=args.limit)
    policy_config = load_policy_config(
        policy_path=args.policy,
        use_adaptive_policy=args.use_adaptive_policy,
    )
    decisions, metrics = evaluate_all_systems(
        records=records,
        deploy_threshold=policy_config.deploy_threshold,
        block_threshold=policy_config.block_threshold,
    )
    write_outputs(
        metrics=metrics,
        decisions=decisions,
        policy_config=policy_config,
    )

    print("# Experimentation & Evaluation Results\n")
    print(metrics_markdown(metrics))
    print("\n## Decision Distribution\n")
    print(decision_distribution_markdown(metrics))
    print(f"\nMarkdown saved to {DEFAULT_MARKDOWN_PATH}")
    print(f"JSON saved to {DEFAULT_JSON_PATH}")
    print(f"Graphs saved to {DEFAULT_GRAPHS_DIR}")


if __name__ == "__main__":
    main()
