"""Reproducible matplotlib figure scripts for the paper; no interactive dependencies."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np


def plot_cumulative_cost(
    policy_costs: dict[str, np.ndarray],
    ci_lower: dict[str, np.ndarray],
    ci_upper: dict[str, np.ndarray],
    output_path: Path,
    title: str = "Cumulative Operational Cost",
) -> None:
    """Plot cumulative cost curves with 95% CI bands for each policy.

    Args:
        policy_costs: Dict policy_id → 1-D array of mean cumulative cost per step.
        ci_lower: Dict policy_id → lower CI bound per step.
        ci_upper: Dict policy_id → upper CI bound per step.
        output_path: Where to save the figure (PNG or PDF).
        title: Figure title.
    """
    # TODO: import matplotlib.pyplot; plot each policy; fill_between for CI; save
    raise NotImplementedError


def plot_cumulative_regret(
    policy_regrets: dict[str, np.ndarray],
    ci_lower: dict[str, np.ndarray],
    ci_upper: dict[str, np.ndarray],
    output_path: Path,
) -> None:
    """Plot cumulative regret vs. oracle for each policy."""
    # TODO: same structure as plot_cumulative_cost
    raise NotImplementedError


def plot_cost_cdf(
    policy_costs: dict[str, np.ndarray],
    output_path: Path,
) -> None:
    """Plot the CDF of per-step cost distributions across policies."""
    # TODO: compute empirical CDFs; overlay on one plot
    raise NotImplementedError


def plot_drift_recovery(
    costs_by_segment: dict[str, list[np.ndarray]],
    drift_steps: list[int],
    output_path: Path,
) -> None:
    """Plot per-segment instantaneous regret with vertical lines at drift events."""
    # TODO: segment costs; draw vlines at drift_steps; shade recovery windows
    raise NotImplementedError


def plot_cost_sweep(
    sweep_results: dict[str, dict[str, float]],
    output_path: Path,
) -> None:
    """Heatmap or line plot of cumulative cost as cost-matrix ratios are swept.

    Args:
        sweep_results: Dict cost_ratio_label → {policy_id → mean_cost}.
        output_path: Save path.
    """
    # TODO: render heatmap or grouped bar chart
    raise NotImplementedError
