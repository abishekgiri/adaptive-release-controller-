"""Tests for the cost-sensitive delayed contextual bandit contribution."""

from __future__ import annotations

import numpy as np

from data.schemas import Action, Context, Outcome, Reward
from drift.detectors import DriftDetector
from policies.base import FeatureEncoder
from policies.cost_sensitive_bandit import (
    CostSensitiveBandit,
    CostSensitiveBanditConfig,
)


def _ctx(step: int = 0, commit_sha: str = "abc123") -> Context:
    return Context(
        commit_sha=commit_sha,
        project_slug="test/repo",
        step=step,
        files_changed=3,
        lines_added=30,
        lines_deleted=10,
        src_churn=40,
        is_pr=False,
        tests_run=100,
        tests_added=2,
        build_duration_s=90.0,
        author_experience=5,
        recent_failure_rate=0.05,
        has_dependency_change=False,
        has_risky_path_change=False,
    )


def _reward(
    action: Action,
    cost: float,
    outcome: Outcome = Outcome.SUCCESS,
    observed_at_step: int = 3,
) -> Reward:
    return Reward(
        action_id=f"reward-{action.value}",
        outcome=outcome,
        cost=cost,
        delay_steps=observed_at_step,
        censored=False,
        observed_at_step=observed_at_step,
    )


def _policy(
    *,
    seed: int = 0,
    alpha: float = 1.0,
    detector: DriftDetector | None = None,
) -> CostSensitiveBandit:
    return CostSensitiveBandit(
        config=CostSensitiveBanditConfig(alpha=alpha, lambda_reg=1.0),
        feature_dim=FeatureEncoder.DIM,
        rng=np.random.default_rng(seed),
        detector=detector,
    )


class OneShotDriftDetector(DriftDetector):
    """Test detector that fires once cost reaches the configured threshold."""

    def __init__(self, threshold: float) -> None:
        self.threshold = threshold
        self.reset_calls = 0
        self._detected = False

    def update(self, value: float) -> bool:
        self._detected = value >= self.threshold
        return self._detected

    def reset(self) -> None:
        self.reset_calls += 1
        self._detected = False

    @property
    def drift_detected(self) -> bool:
        return self._detected


def test_no_update_before_delayed_reward_arrives() -> None:
    policy = _policy()
    ctx = _ctx()
    initial_A = policy._A[Action.DEPLOY].copy()

    policy.record_pending_reward(
        action_id="a1",
        context=ctx,
        action=Action.DEPLOY,
        outcome=Outcome.SUCCESS,
        cost=0.0,
        current_step=0,
        delay=3,
    )
    applied = policy.advance_to(2)

    assert applied == []
    np.testing.assert_array_equal(policy._A[Action.DEPLOY], initial_A)
    assert policy.stats.delayed_updates_applied == 0
    assert policy.stats.pending_rewards == 1


def test_update_after_delayed_reward_arrives() -> None:
    policy = _policy()
    ctx = _ctx()
    initial_A = policy._A[Action.DEPLOY].copy()

    policy.record_pending_reward(
        action_id="a1",
        context=ctx,
        action=Action.DEPLOY,
        outcome=Outcome.SUCCESS,
        cost=0.0,
        current_step=0,
        delay=3,
    )
    applied = policy.advance_to(3)

    assert [pending.action_id for pending in applied] == ["a1"]
    assert not np.array_equal(policy._A[Action.DEPLOY], initial_A)
    assert policy.stats.delayed_updates_applied == 1
    assert policy.stats.pending_rewards == 0


def test_high_cost_failure_reduces_future_preference() -> None:
    policy = _policy(alpha=0.0)
    ctx = _ctx()

    before, _ = policy.select_action(ctx)
    policy.update(
        ctx,
        Action.DEPLOY,
        _reward(Action.DEPLOY, cost=10.0, outcome=Outcome.FAILURE),
    )
    after, _ = policy.select_action(ctx)

    assert before == Action.DEPLOY
    assert after != Action.DEPLOY
    assert policy.stats.cumulative_cost == 10.0


def test_drift_reset_refreshes_model_state() -> None:
    detector = OneShotDriftDetector(threshold=5.0)
    policy = _policy(detector=detector)
    ctx = _ctx()

    policy.update(
        ctx,
        Action.DEPLOY,
        _reward(Action.DEPLOY, cost=10.0, outcome=Outcome.FAILURE),
    )

    np.testing.assert_array_equal(
        policy._A[Action.DEPLOY],
        np.eye(FeatureEncoder.DIM),
    )
    np.testing.assert_array_equal(
        policy._b[Action.DEPLOY],
        np.zeros(FeatureEncoder.DIM),
    )
    assert policy.stats.drift_resets == 1
    assert detector.reset_calls == 1


def test_deterministic_behavior_with_fixed_seed() -> None:
    first = _policy(seed=7)
    second = _policy(seed=7)
    contexts = [_ctx(step=index, commit_sha=f"c{index}") for index in range(5)]

    first_actions = []
    second_actions = []
    for index, ctx in enumerate(contexts):
        first_action, _ = first.select_action(ctx)
        second_action, _ = second.select_action(ctx)
        first_actions.append(first_action)
        second_actions.append(second_action)
        first.record_pending_reward(
            action_id=f"first-{index}",
            context=ctx,
            action=first_action,
            outcome=Outcome.FAILURE if index % 2 else Outcome.SUCCESS,
            cost=float(index),
            current_step=index,
        )
        second.record_pending_reward(
            action_id=f"second-{index}",
            context=ctx,
            action=second_action,
            outcome=Outcome.FAILURE if index % 2 else Outcome.SUCCESS,
            cost=float(index),
            current_step=index,
        )
        first.advance_to(index + 1)
        second.advance_to(index + 1)

    assert first_actions == second_actions
    assert first.stats == second.stats
    for action in Action:
        np.testing.assert_allclose(first._A[action], second._A[action])
        np.testing.assert_allclose(first._b[action], second._b[action])
