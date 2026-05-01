"""Tests for cost metrics and bootstrap summaries."""

from __future__ import annotations

import math

import numpy as np
import pytest

from data.schemas import Action
from evaluation.metrics import (
    EpisodeRecord,
    action_distribution,
    cumulative_cost,
    cumulative_regret,
    mean_operational_cost,
    total_operational_cost,
)
from evaluation.statistical import BootstrapConfig, bootstrap_ci, holm_bonferroni


def test_cumulative_cost_ignores_censored_nan_values() -> None:
    result = cumulative_cost([1.0, math.nan, 2.5])

    np.testing.assert_allclose(result, np.array([1.0, 3.5]))
    assert total_operational_cost([1.0, math.nan, 2.5]) == 3.5
    assert mean_operational_cost([1.0, math.nan, 2.5]) == 1.75


def test_cumulative_regret_uses_cost_minus_oracle_cost() -> None:
    record = EpisodeRecord(
        policy_id="p",
        seed=0,
        costs=[3.0, 4.0],
        oracle_costs=[1.0, 1.5],
    )

    np.testing.assert_allclose(cumulative_regret(record), np.array([2.0, 4.5]))


def test_action_distribution_returns_fractions_for_all_actions() -> None:
    distribution = action_distribution([Action.DEPLOY, Action.DEPLOY, Action.BLOCK])

    assert distribution[Action.DEPLOY] == pytest.approx(2 / 3)
    assert distribution[Action.CANARY] == 0.0
    assert distribution[Action.BLOCK] == pytest.approx(1 / 3)


def test_bootstrap_ci_is_deterministic_with_fixed_seed() -> None:
    config = BootstrapConfig(n_resamples=100, confidence=0.95, seed=7)

    first = bootstrap_ci([1.0, 2.0, 3.0, 4.0], config=config)
    second = bootstrap_ci([1.0, 2.0, 3.0, 4.0], config=config)

    assert first == second
    assert first[0] == 2.5
    assert first[1] <= first[0] <= first[2]


def test_holm_bonferroni_rejects_in_order() -> None:
    assert holm_bonferroni([0.001, 0.02, 0.2], alpha=0.05) == [
        True,
        True,
        False,
    ]
