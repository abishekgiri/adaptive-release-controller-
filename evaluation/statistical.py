"""Bootstrap confidence intervals, paired bootstrap tests, and Holm-Bonferroni correction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BootstrapConfig:
    """Controls bootstrap resampling behaviour."""

    n_resamples: int = 10_000    # number of bootstrap resamples
    confidence: float = 0.95     # desired CI coverage (two-sided)
    seed: int = 0                # RNG seed for reproducibility


def bootstrap_ci(
    values: np.ndarray,
    config: BootstrapConfig = BootstrapConfig(),
    statistic: str = "mean",
) -> tuple[float, float, float]:
    """Compute bootstrap confidence interval for the mean (or other statistic) of values.

    Args:
        values: 1-D array of per-seed scalar observations (e.g. cumulative costs).
        config: Bootstrap parameters.
        statistic: 'mean' or 'median'.

    Returns:
        (point_estimate, lower_bound, upper_bound)
    """
    # TODO: resample with replacement; compute statistic on each resample; percentile CI
    raise NotImplementedError


def paired_bootstrap_pvalue(
    values_a: np.ndarray,
    values_b: np.ndarray,
    config: BootstrapConfig = BootstrapConfig(),
) -> float:
    """Compute two-sided paired bootstrap p-value for H0: E[A] == E[B].

    Args:
        values_a: Per-seed results for policy A (same seeds as values_b).
        values_b: Per-seed results for policy B.
        config: Bootstrap parameters.

    Returns:
        Two-sided p-value.
    """
    # TODO: compute observed difference; resample pairs; count extreme differences
    raise NotImplementedError


def holm_bonferroni(p_values: list[float], alpha: float = 0.05) -> list[bool]:
    """Apply Holm-Bonferroni correction for multiple comparisons.

    Args:
        p_values: List of raw p-values from pairwise comparisons.
        alpha: Family-wise error rate.

    Returns:
        List of booleans: True if the corresponding null hypothesis is rejected.
    """
    # TODO: sort p-values; apply step-down correction; return reject flags in original order
    raise NotImplementedError


def summarise(
    policy_values: dict[str, np.ndarray],
    config: BootstrapConfig = BootstrapConfig(),
) -> dict[str, dict]:
    """Compute mean, CI, and pairwise p-values (with Holm-Bonferroni) for all policies.

    Args:
        policy_values: Dict mapping policy_id → 1-D array of per-seed cumulative costs.
        config: Bootstrap parameters.

    Returns:
        Dict mapping policy_id → {mean, ci_lower, ci_upper, p_values_vs_others}.
    """
    # TODO: call bootstrap_ci per policy; call paired_bootstrap_pvalue for all pairs; correct
    raise NotImplementedError
