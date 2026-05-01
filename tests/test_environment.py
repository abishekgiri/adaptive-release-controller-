"""Tests for hidden-state / observable-context separation in SyntheticEnvironment."""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from data.schemas import Action, Context, Outcome, Reward
from environment.synthetic import (
    DriftSchedule,
    HiddenState,
    SegmentParams,
    SyntheticEnvironment,
    default_drift_schedule,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_env(
    seed: int = 0,
    horizon: int = 500,
    drift_schedule: DriftSchedule | None = None,
) -> SyntheticEnvironment:
    rng = np.random.default_rng(seed)
    return SyntheticEnvironment(rng=rng, horizon=horizon, drift_schedule=drift_schedule)


def _run_and_collect_outcomes(
    env: SyntheticEnvironment,
    action: Action = Action.DEPLOY,
    n_steps: int = 200,
) -> list[Outcome]:
    """Run env for n_steps, collecting all matured reward outcomes."""
    outcomes: list[Outcome] = []
    for _ in range(n_steps):
        matured = env.advance_time()
        for r in matured:
            if not r.censored:
                outcomes.append(r.outcome)
        env.observe()
        env.step(action)
    return outcomes


# ---------------------------------------------------------------------------
# Structural invariants (no randomness needed)
# ---------------------------------------------------------------------------

def test_hidden_context_field_disjointness() -> None:
    """HiddenState and Context must share no field names — the core validity invariant."""
    hidden_fields = set(HiddenState.__dataclass_fields__)
    context_fields = set(Context.__dataclass_fields__)
    overlap = hidden_fields & context_fields
    assert overlap == set(), (
        f"HiddenState and Context share fields {overlap}. "
        "This violates the hidden-state separation rule and invalidates the experiment."
    )


def test_context_contains_no_outcome_fields() -> None:
    """Context fields must not include outcome, risk_score, decision, reward, or cost."""
    forbidden = {"outcome", "risk_score", "decision", "reward", "cost"}
    context_fields = set(Context.__dataclass_fields__)
    leakage = context_fields & forbidden
    assert leakage == set(), (
        f"Context contains post-action fields {leakage}. "
        "Policy must not see outcome information at decision time."
    )


def test_sample_outcome_signature_has_no_context_param() -> None:
    """_sample_outcome must accept only (self, hidden, action) — no Context parameter.

    This is the structural proof that outcomes cannot depend on Context features.
    """
    sig = inspect.signature(SyntheticEnvironment._sample_outcome)
    param_names = set(sig.parameters) - {"self"}
    assert "context" not in param_names, (
        "_sample_outcome accepts a 'context' parameter — outcome must not depend on Context."
    )
    assert "hidden" in param_names, "_sample_outcome must accept 'hidden: HiddenState'."
    assert "action" in param_names, "_sample_outcome must accept 'action: Action'."


# ---------------------------------------------------------------------------
# Functional correctness
# ---------------------------------------------------------------------------

def test_reset_returns_context_instance() -> None:
    """reset() must return a Context, not a HiddenState or raw dict."""
    env = _make_env()
    ctx = env.reset()
    assert isinstance(ctx, Context), f"reset() returned {type(ctx)}, expected Context"


def test_observe_returns_context_instance() -> None:
    """observe() must return a Context instance."""
    env = _make_env()
    env.reset()
    ctx = env.observe()
    assert isinstance(ctx, Context), f"observe() returned {type(ctx)}, expected Context"


def test_observe_step_field_matches_current_step() -> None:
    """Context.step must equal env.current_step at every point."""
    env = _make_env()
    env.reset()
    for _ in range(10):
        env.advance_time()
        ctx = env.observe()
        assert ctx.step == env.current_step, (
            f"Context.step={ctx.step} != env.current_step={env.current_step}"
        )
        env.step(Action.DEPLOY)


def test_step_always_returns_none() -> None:
    """step() must return None — rewards are always delayed in this environment."""
    env = _make_env()
    env.reset()
    result = env.step(Action.DEPLOY)
    assert result is None, f"step() returned {result!r}, expected None (all rewards are delayed)"


def test_advance_time_returns_list() -> None:
    """advance_time() must return a list (possibly empty)."""
    env = _make_env()
    env.reset()
    env.step(Action.DEPLOY)
    matured = env.advance_time()
    assert isinstance(matured, list), f"advance_time() returned {type(matured)}, expected list"


def test_advance_time_increments_step() -> None:
    """advance_time() must increment current_step by exactly 1."""
    env = _make_env()
    env.reset()
    assert env.current_step == 0
    env.advance_time()
    assert env.current_step == 1
    env.advance_time()
    assert env.current_step == 2


def test_matured_rewards_are_reward_instances() -> None:
    """Every element returned by advance_time() must be a Reward instance."""
    env = _make_env(horizon=100, seed=7)
    env.reset()
    for _ in range(50):
        env.step(Action.DEPLOY)
        matured = env.advance_time()
        for r in matured:
            assert isinstance(r, Reward), f"advance_time() returned {type(r)}, expected Reward"


def test_reward_cost_is_nan() -> None:
    """Rewards from this environment must have cost=nan (cost_model computes it, not env)."""
    import math
    env = _make_env(horizon=100)
    env.reset()
    for _ in range(50):
        env.step(Action.DEPLOY)
        for r in env.advance_time():
            if not r.censored:
                assert math.isnan(r.cost), (
                    f"Environment set cost={r.cost}; cost must be nan — "
                    "compute via rewards/cost_model.py, not here."
                )


def test_done_triggers_at_horizon() -> None:
    """done must become True exactly when current_step reaches horizon."""
    horizon = 20
    env = _make_env(horizon=horizon)
    env.reset()
    for _ in range(horizon):
        assert not env.done
        env.step(Action.DEPLOY)
        env.advance_time()
    assert env.done


def test_context_fields_are_valid_types() -> None:
    """All Context fields must have plausible types (no NaN, no negative counts)."""
    env = _make_env(seed=99)
    env.reset()
    for _ in range(30):
        env.advance_time()
        ctx = env.observe()
        assert ctx.files_changed >= 1
        assert ctx.lines_added >= 0
        assert ctx.lines_deleted >= 0
        assert ctx.src_churn >= 0
        assert ctx.tests_run >= 0
        assert ctx.tests_added >= 0
        assert ctx.build_duration_s >= 10.0
        assert ctx.author_experience >= 0
        assert 0.0 <= ctx.recent_failure_rate <= 1.0
        assert isinstance(ctx.is_pr, bool)
        assert isinstance(ctx.has_dependency_change, bool)
        assert isinstance(ctx.has_risky_path_change, bool)
        env.step(Action.DEPLOY)


# ---------------------------------------------------------------------------
# Drift detection
# ---------------------------------------------------------------------------

def test_drift_schedule_changes_outcome_distribution() -> None:
    """After a drift event, the empirical failure rate must shift detectably.

    Segment 0: base_failure_prob=0.10  →  expect ~10 % failures
    Segment 1: base_failure_prob=0.55  →  expect ~55 % failures
    The two means must be statistically separated by > 0.20.
    """
    segment_length = 200
    schedule = DriftSchedule(
        segment_length=segment_length,
        segments=(
            SegmentParams(
                base_failure_prob=0.10,
                infra_load_mean=0.2,
                change_complexity_mean=0.2,
                team_fatigue_mean=0.2,
            ),
            SegmentParams(
                base_failure_prob=0.55,
                infra_load_mean=0.7,
                change_complexity_mean=0.7,
                team_fatigue_mean=0.6,
            ),
        ),
    )

    env = _make_env(seed=42, horizon=segment_length * 2 + 50, drift_schedule=schedule)
    env.reset()

    # --- Segment 0: steps 0 .. segment_length-1 ---
    pre_drift_outcomes = _run_and_collect_outcomes(env, n_steps=segment_length)
    # --- Segment 1: steps segment_length .. 2*segment_length-1 ---
    post_drift_outcomes = _run_and_collect_outcomes(env, n_steps=segment_length)

    def failure_rate(outcomes: list[Outcome]) -> float:
        if not outcomes:
            return float("nan")
        return sum(o == Outcome.FAILURE for o in outcomes) / len(outcomes)

    pre_rate = failure_rate(pre_drift_outcomes)
    post_rate = failure_rate(post_drift_outcomes)

    assert pre_drift_outcomes, "No outcomes collected in pre-drift segment"
    assert post_drift_outcomes, "No outcomes collected in post-drift segment"
    assert post_rate - pre_rate > 0.20, (
        f"Drift did not shift failure rate enough: pre={pre_rate:.3f}, post={post_rate:.3f}. "
        f"Expected gap > 0.20."
    )


def test_no_drift_keeps_stable_failure_rate() -> None:
    """Without a drift event, failure rate should be stable across two equal windows."""
    schedule = DriftSchedule(
        segment_length=10_000,   # segment is much longer than the run
        segments=(
            SegmentParams(
                base_failure_prob=0.30,
                infra_load_mean=0.4,
                change_complexity_mean=0.4,
                team_fatigue_mean=0.3,
            ),
        ),
    )
    env = _make_env(seed=1, horizon=400, drift_schedule=schedule)
    env.reset()

    first_half = _run_and_collect_outcomes(env, n_steps=200)
    second_half = _run_and_collect_outcomes(env, n_steps=200)

    def failure_rate(outcomes: list[Outcome]) -> float:
        if not outcomes:
            return float("nan")
        return sum(o == Outcome.FAILURE for o in outcomes) / len(outcomes)

    rate_1 = failure_rate(first_half)
    rate_2 = failure_rate(second_half)

    # Both halves should be near 0.30; gap should be small (sampling noise only)
    assert abs(rate_1 - rate_2) < 0.20, (
        f"Unexpected rate shift without drift: first={rate_1:.3f}, second={rate_2:.3f}"
    )


def test_default_drift_schedule_produces_two_segments() -> None:
    """default_drift_schedule() must have exactly two segments."""
    schedule = default_drift_schedule()
    assert len(schedule.segments) == 2
    low_risk, high_risk = schedule.segments
    assert high_risk.base_failure_prob > low_risk.base_failure_prob + 0.20, (
        "High-risk segment failure prob must be substantially higher than low-risk."
    )
