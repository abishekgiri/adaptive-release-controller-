"""Tests for hidden-state / observable-context separation in SyntheticEnvironment."""

from __future__ import annotations

import numpy as np
import pytest

from data.schemas import Action, Context
from environment.synthetic import HiddenState, SyntheticEnvironment


def test_hidden_context_field_disjointness() -> None:
    """HiddenState and Context must share no field names — the core validity invariant."""
    hidden_fields = set(HiddenState.__dataclass_fields__)
    context_fields = set(Context.__dataclass_fields__)
    overlap = hidden_fields & context_fields
    assert overlap == set(), (
        f"HiddenState and Context share fields {overlap}. "
        "This violates the hidden-state separation rule and invalidates the experiment."
    )


def test_observe_returns_context_type() -> None:
    """observe() must return a Context instance, not a HiddenState or raw dict."""
    # TODO: construct SyntheticEnvironment with fixed seed; call reset() then observe()
    # assert isinstance(result, Context)
    pytest.skip("Implement after SyntheticEnvironment.observe() is filled in")


def test_context_contains_no_outcome_fields() -> None:
    """Context fields must not include outcome, risk_score, decision, or reward."""
    forbidden = {"outcome", "risk_score", "decision", "reward", "cost"}
    context_fields = set(Context.__dataclass_fields__)
    leakage = context_fields & forbidden
    assert leakage == set(), (
        f"Context contains post-action fields {leakage}. "
        "Policy must not see outcome information at decision time."
    )


def test_step_does_not_expose_hidden_state() -> None:
    """step() must not return a HiddenState; it returns Optional[Reward]."""
    # TODO: construct env; take an action; assert return type is Reward or None
    pytest.skip("Implement after SyntheticEnvironment.step() is filled in")


def test_drift_schedule_changes_outcome_distribution() -> None:
    """After a drift event, the empirical failure rate must shift detectably."""
    # TODO: run env through drift boundary; compare failure rates pre/post drift
    pytest.skip("Implement after DriftSchedule and SyntheticEnvironment are filled in")
