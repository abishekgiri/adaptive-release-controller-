"""Tests for IPS/DR estimator unbiasedness on synthetic data with known true policy value."""

from __future__ import annotations

import pytest

from evaluation.replay_eval import IPSConfig, dr_estimate, ips_estimate, snips_estimate


def test_ips_recovers_true_value_under_uniform_logging() -> None:
    """When logging policy is uniform, IPS estimate must converge to the true policy value."""
    # TODO: construct synthetic trajectory where all propensities = 1/|A|;
    # true value is known analytically; assert |IPS - true| < tolerance
    pytest.skip("Implement after ReplayEnvironment and ips_estimate are filled in")


def test_dr_is_consistent_when_dm_is_misspecified() -> None:
    """DR must still recover true value when direct model is wrong but IPS is correct."""
    # TODO: construct trajectory with known logging propensities;
    # pass a misspecified constant direct_model; assert DR ≈ IPS ≈ true value
    pytest.skip("Implement after dr_estimate is filled in")


def test_ips_propensity_clipping_reduces_variance() -> None:
    """Clipped IPS must have lower variance than unclipped IPS across multiple trajectories."""
    # TODO: generate N trajectories with a high-variance logging policy;
    # compare variance of IPS(clip=1) vs IPS(clip=inf)
    pytest.skip("Implement after ips_estimate clipping is filled in")


def test_snips_closer_to_true_than_ips_on_small_samples() -> None:
    """SNIPS typically outperforms IPS in mean squared error on small trajectory samples."""
    # TODO: run on short trajectories; compare MSE of IPS vs SNIPS
    pytest.skip("Implement after snips_estimate is filled in")
