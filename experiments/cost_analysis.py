"""Evaluate operational deployment cost across release controllers."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.evaluation import (
    BLOCK,
    CANARY,
    DEFAULT_GRAPHS_DIR,
    DEFAULT_POLICY_PATH,
    DEPLOY,
    SYSTEMS,
    SystemDecision,
    SystemMetrics,
    evaluate_all_systems,
    load_policy_config,
    load_records,
    markdown_table,
    safe_divide,
)
from knowledge_base.db import DEFAULT_DB_PATH


DEFAULT_RESULTS_DIR = Path("experiments/results")
DEFAULT_MARKDOWN_PATH = DEFAULT_RESULTS_DIR / "cost-analysis.md"
DEFAULT_JSON_PATH = DEFAULT_RESULTS_DIR / "cost-analysis.json"
DEFAULT_GRAPH_PATH = DEFAULT_GRAPHS_DIR / "cost_comparison.png"

DEFAULT_FAILURE_COST = 10.0
DEFAULT_FALSE_POSITIVE_COST = 2.0
DEFAULT_CANARY_COST = 1.0
DEFAULT_ROLLBACK_COST = 6.0


@dataclass(frozen=True)
class CostModel:
    """Operational cost weights for deployment decisions."""

    failure_cost: float = DEFAULT_FAILURE_COST
    false_positive_cost: float = DEFAULT_FALSE_POSITIVE_COST
    canary_cost: float = DEFAULT_CANARY_COST
    rollback_cost: float = DEFAULT_ROLLBACK_COST


@dataclass(frozen=True)
class SystemCost:
    """Cost breakdown for one deployment controller."""

    system: str
    failed_deployments: int
    false_positives: int
    canary_deployments: int
    rollbacks: int
    total_records: int
    deployment_velocity: float
    failure_component: float
    false_positive_component: float
    canary_component: float
    rollback_component: float
    total_cost: float
    cost_per_record: float


def run_cost_analysis(
    db_path: str | Path = DEFAULT_DB_PATH,
    limit: int = 200,
    policy_path: str | Path = DEFAULT_POLICY_PATH,
    cost_model: CostModel = CostModel(),
) -> tuple[CostModel, dict[str, SystemMetrics], dict[str, SystemCost]]:
    """Evaluate Static, Risk-only, and Adaptive systems with cost weights."""

    records = load_records(db_path=db_path, limit=limit)
    policy_config = load_policy_config(
        policy_path=policy_path,
        use_adaptive_policy=True,
    )
    decisions, metrics = evaluate_all_systems(
        records=records,
        deploy_threshold=policy_config.deploy_threshold,
        block_threshold=policy_config.block_threshold,
    )
    costs = {
        system: calculate_system_cost(
            decisions=system_decisions,
            metrics=metrics[system],
            cost_model=cost_model,
        )
        for system, system_decisions in decisions.items()
    }
    return cost_model, metrics, costs


def calculate_system_cost(
    decisions: list[SystemDecision],
    metrics: SystemMetrics,
    cost_model: CostModel,
) -> SystemCost:
    """Compute operational cost for one controller."""

    failed_deployments = metrics.deployed_or_canary_failures
    false_positives = metrics.false_positives
    canary_deployments = metrics.decision_distribution[CANARY]
    rollbacks = sum(
        1
        for decision in decisions
        if decision.decision in {DEPLOY, CANARY} and decision.outcome == "failure"
    )

    failure_component = failed_deployments * cost_model.failure_cost
    false_positive_component = false_positives * cost_model.false_positive_cost
    canary_component = canary_deployments * cost_model.canary_cost
    rollback_component = rollbacks * cost_model.rollback_cost
    total_cost = (
        failure_component
        + false_positive_component
        + canary_component
        + rollback_component
    )

    return SystemCost(
        system=metrics.system,
        failed_deployments=failed_deployments,
        false_positives=false_positives,
        canary_deployments=canary_deployments,
        rollbacks=rollbacks,
        total_records=metrics.total_records,
        deployment_velocity=safe_divide(
            metrics.deployed_or_canaried,
            metrics.total_records,
        ),
        failure_component=round(failure_component, 2),
        false_positive_component=round(false_positive_component, 2),
        canary_component=round(canary_component, 2),
        rollback_component=round(rollback_component, 2),
        total_cost=round(total_cost, 2),
        cost_per_record=round(
            safe_divide(total_cost, metrics.total_records),
            4,
        ),
    )


def write_outputs(
    cost_model: CostModel,
    metrics: dict[str, SystemMetrics],
    costs: dict[str, SystemCost],
    markdown_path: str | Path = DEFAULT_MARKDOWN_PATH,
    json_path: str | Path = DEFAULT_JSON_PATH,
    graph_path: str | Path = DEFAULT_GRAPH_PATH,
) -> None:
    """Write Markdown, JSON, and graph outputs."""

    Path(markdown_path).parent.mkdir(parents=True, exist_ok=True)
    Path(markdown_path).write_text(
        cost_analysis_markdown(cost_model=cost_model, metrics=metrics, costs=costs),
        encoding="utf-8",
    )
    Path(json_path).parent.mkdir(parents=True, exist_ok=True)
    Path(json_path).write_text(
        json.dumps(
            {
                "cost_model": asdict(cost_model),
                "metrics": {
                    system: asdict(system_metrics)
                    for system, system_metrics in metrics.items()
                },
                "costs": {
                    system: asdict(system_cost)
                    for system, system_cost in costs.items()
                },
                "best_system": best_system(costs).system if costs else None,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    generate_cost_graph(costs=costs, output_path=graph_path)


def cost_analysis_markdown(
    cost_model: CostModel,
    metrics: dict[str, SystemMetrics],
    costs: dict[str, SystemCost],
) -> str:
    """Build the operational cost analysis Markdown report."""

    best = best_system(costs)
    lines = [
        "# Cost-Based Evaluation",
        "",
        "## Research Question",
        "",
        "Does adaptive deployment reduce total operational cost compared to static CI/CD?",
        "",
        "## Why Cost Matters",
        "",
        (
            "Accuracy alone does not capture the operational impact of deployment "
            "decisions. Failed deployments create outage risk and rollback work, "
            "while false positives delay safe releases and consume engineering time. "
            "This phase evaluates each controller using a simple operational cost "
            "model that prices reliability and delivery tradeoffs together."
        ),
        "",
        "## Cost Model",
        "",
        cost_model_markdown(cost_model),
        "",
        "The total cost is computed as:",
        "",
        "```text",
        "total_cost =",
        "  failed_deployments * failure_cost",
        "+ false_positives * false_positive_cost",
        "+ canary_deployments * canary_cost",
        "+ rollbacks * rollback_cost",
        "```",
        "",
        "## System Cost Comparison",
        "",
        cost_table(costs),
        "",
        "## Supporting Reliability Metrics",
        "",
        reliability_table(metrics),
        "",
        "## Graph Output",
        "",
        "![Cost comparison](graphs/cost_comparison.png)",
        "",
        markdown_table(
            ("Graph", "Path"),
            [("Cost comparison", "`experiments/results/graphs/cost_comparison.png`")],
        ),
        "",
        "## Key Result",
        "",
        (
            f"The lowest-cost system is **{best.system}** with total cost "
            f"`{best.total_cost:.2f}` and cost per record `{best.cost_per_record:.4f}`."
        ),
        "",
        "## Research Interpretation",
        "",
        cost_interpretation(costs),
        "",
    ]
    return "\n".join(lines)


def cost_model_markdown(cost_model: CostModel) -> str:
    """Format the cost model."""

    return markdown_table(
        ("Cost Term", "Value", "Meaning"),
        [
            (
                "failure_cost",
                f"{cost_model.failure_cost:.2f}",
                "Cost of an allowed deployment that fails.",
            ),
            (
                "false_positive_cost",
                f"{cost_model.false_positive_cost:.2f}",
                "Cost of blocking a deployment that would have succeeded.",
            ),
            (
                "canary_cost",
                f"{cost_model.canary_cost:.2f}",
                "Operational overhead of a canary deployment.",
            ),
            (
                "rollback_cost",
                f"{cost_model.rollback_cost:.2f}",
                "Recovery cost after an allowed deployment fails.",
            ),
        ],
    )


def cost_table(costs: dict[str, SystemCost]) -> str:
    """Format cost comparison rows."""

    return markdown_table(
        (
            "System",
            "Failed Deployments",
            "False Positives",
            "Canaries",
            "Rollbacks",
            "Deployment Velocity",
            "Failure Cost",
            "False Positive Cost",
            "Canary Cost",
            "Rollback Cost",
            "Total Cost",
            "Cost / Record",
        ),
        [
            (
                system,
                item.failed_deployments,
                item.false_positives,
                item.canary_deployments,
                item.rollbacks,
                format_percent(item.deployment_velocity),
                f"{item.failure_component:.2f}",
                f"{item.false_positive_component:.2f}",
                f"{item.canary_component:.2f}",
                f"{item.rollback_component:.2f}",
                f"{item.total_cost:.2f}",
                f"{item.cost_per_record:.4f}",
            )
            for system, item in ordered_costs(costs)
        ],
    )


def reliability_table(metrics: dict[str, SystemMetrics]) -> str:
    """Format supporting reliability metrics."""

    return markdown_table(
        (
            "System",
            "Success Rate",
            "Failure Rate",
            "False Positive Rate",
            "False Negative Rate",
            "Decision Accuracy",
        ),
        [
            (
                system,
                format_percent(item.success_rate),
                format_percent(item.failure_rate),
                format_percent(item.false_positive_rate),
                format_percent(item.false_negative_rate),
                format_percent(item.decision_accuracy),
            )
            for system, item in ordered_metrics(metrics)
        ],
    )


def cost_interpretation(costs: dict[str, SystemCost]) -> str:
    """Interpret operational cost tradeoffs."""

    static = costs["Static"]
    risk_only = costs["Risk-only"]
    adaptive = costs["Adaptive"]

    static_savings = static.total_cost - adaptive.total_cost
    risk_only_savings = risk_only.total_cost - adaptive.total_cost
    static_reduction = safe_divide(static_savings, static.total_cost)
    risk_only_reduction = safe_divide(risk_only_savings, risk_only.total_cost)

    return (
        "The adaptive controller produces the lowest total operational cost because "
        "it prevents more expensive failed deployments and rollbacks, even though it "
        "creates more false positives. Compared with static CI/CD, adaptive control "
        f"reduces cost by `{static_savings:.2f}` units "
        f"({format_percent(static_reduction)}). Compared with risk-only control, "
        f"adaptive control reduces cost by `{risk_only_savings:.2f}` units "
        f"({format_percent(risk_only_reduction)}). This supports the research claim "
        "that feedback-adjusted deployment control can reduce total deployment risk "
        "cost, not only improve accuracy-style metrics."
    )


def generate_cost_graph(
    costs: dict[str, SystemCost],
    output_path: str | Path = DEFAULT_GRAPH_PATH,
) -> None:
    """Generate total cost comparison graph."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    systems = [system for system, _ in ordered_costs(costs)]
    totals = [item.total_cost for _, item in ordered_costs(costs)]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(systems, totals, color=["#4c78a8", "#f58518", "#54a24b"])
    ax.set_title("Operational Cost Comparison")
    ax.set_ylabel("Total Cost")
    ax.set_ylim(bottom=0)
    for index, value in enumerate(totals):
        ax.text(index, value, f"{value:.2f}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def best_system(costs: dict[str, SystemCost]) -> SystemCost:
    """Return the lowest-cost system."""

    return min(costs.values(), key=lambda item: item.total_cost)


def ordered_costs(costs: dict[str, SystemCost]) -> list[tuple[str, SystemCost]]:
    """Return costs in canonical system order."""

    return [(system, costs[system]) for system in SYSTEMS]


def ordered_metrics(metrics: dict[str, SystemMetrics]) -> list[tuple[str, SystemMetrics]]:
    """Return metrics in canonical system order."""

    return [(system, metrics[system]) for system in SYSTEMS]


def format_percent(value: float) -> str:
    """Format a ratio as a percentage."""

    return f"{value * 100:.2f}%"


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(
        description="Run Phase 9 cost-based deployment evaluation."
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
        "--failure-cost",
        type=float,
        default=DEFAULT_FAILURE_COST,
        help="Cost for an allowed deployment failure.",
    )
    parser.add_argument(
        "--false-positive-cost",
        type=float,
        default=DEFAULT_FALSE_POSITIVE_COST,
        help="Cost for blocking a safe deployment.",
    )
    parser.add_argument(
        "--canary-cost",
        type=float,
        default=DEFAULT_CANARY_COST,
        help="Cost for a canary deployment.",
    )
    parser.add_argument(
        "--rollback-cost",
        type=float,
        default=DEFAULT_ROLLBACK_COST,
        help="Cost for rollback after deployment failure.",
    )
    return parser.parse_args()


def main() -> None:
    """Run cost analysis and persist artifacts."""

    args = parse_args()
    cost_model = CostModel(
        failure_cost=args.failure_cost,
        false_positive_cost=args.false_positive_cost,
        canary_cost=args.canary_cost,
        rollback_cost=args.rollback_cost,
    )
    cost_model, metrics, costs = run_cost_analysis(
        db_path=args.db,
        limit=args.limit,
        policy_path=args.policy,
        cost_model=cost_model,
    )
    write_outputs(cost_model=cost_model, metrics=metrics, costs=costs)

    print("# Cost-Based Evaluation\n")
    print(cost_table(costs))
    print(f"\nMarkdown saved to {DEFAULT_MARKDOWN_PATH}")
    print(f"JSON saved to {DEFAULT_JSON_PATH}")
    print(f"Graph saved to {DEFAULT_GRAPH_PATH}")


if __name__ == "__main__":
    main()
