"""Bootstrap confidence intervals for experiment summaries."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BootstrapConfig:
    """Controls bootstrap resampling behaviour."""

    n_resamples: int = 10_000
    confidence: float = 0.95
    seed: int = 0


def bootstrap_ci(
    values: list[float] | np.ndarray,
    config: BootstrapConfig = BootstrapConfig(),
    statistic: str = "mean",
) -> tuple[float, float, float]:
    """Compute a percentile bootstrap confidence interval.

    Returns ``(point_estimate, lower_bound, upper_bound)``.
    """

    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return 0.0, 0.0, 0.0
    if config.n_resamples <= 0:
        raise ValueError("n_resamples must be positive")
    if not 0.0 < config.confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1")

    stat_fn = _statistic_fn(statistic)
    point = float(stat_fn(array))
    rng = np.random.default_rng(config.seed)
    samples = np.empty(config.n_resamples, dtype=np.float64)
    for index in range(config.n_resamples):
        resample = rng.choice(array, size=array.size, replace=True)
        samples[index] = stat_fn(resample)

    alpha = 1.0 - config.confidence
    lower = float(np.quantile(samples, alpha / 2.0))
    upper = float(np.quantile(samples, 1.0 - alpha / 2.0))
    return point, lower, upper


def paired_bootstrap_pvalue(
    values_a: np.ndarray,
    values_b: np.ndarray,
    config: BootstrapConfig = BootstrapConfig(),
) -> float:
    """Compute a simple two-sided paired bootstrap p-value."""

    a = np.asarray(values_a, dtype=np.float64)
    b = np.asarray(values_b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("paired bootstrap requires arrays with the same shape")
    if a.size == 0:
        return 1.0

    diff = a - b
    observed = abs(float(np.mean(diff)))
    rng = np.random.default_rng(config.seed)
    extreme = 0
    centered = diff - np.mean(diff)
    for _ in range(config.n_resamples):
        resample = rng.choice(centered, size=centered.size, replace=True)
        if abs(float(np.mean(resample))) >= observed:
            extreme += 1
    return (extreme + 1) / (config.n_resamples + 1)


def holm_bonferroni(p_values: list[float], alpha: float = 0.05) -> list[bool]:
    """Apply Holm-Bonferroni correction and return reject flags."""

    indexed = sorted(enumerate(p_values), key=lambda item: item[1])
    reject = [False] * len(p_values)
    for rank, (index, p_value) in enumerate(indexed):
        threshold = alpha / (len(p_values) - rank)
        if p_value <= threshold:
            reject[index] = True
        else:
            break
    return reject


def summarise(
    policy_values: dict[str, list[float] | np.ndarray],
    config: BootstrapConfig = BootstrapConfig(),
) -> dict[str, dict[str, float]]:
    """Return mean and bootstrap CI per policy."""

    summary = {}
    for policy_id, values in policy_values.items():
        mean, lower, upper = bootstrap_ci(values, config=config)
        summary[policy_id] = {
            "mean": mean,
            "ci_lower": lower,
            "ci_upper": upper,
        }
    return summary


def _statistic_fn(statistic: str):
    if statistic == "mean":
        return np.mean
    if statistic == "median":
        return np.median
    raise ValueError("statistic must be 'mean' or 'median'")
