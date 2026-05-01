"""Tests for IPS replay evaluation."""

from __future__ import annotations

import pytest

from data.schemas import Action, Context, Outcome, Reward, Trajectory, TrajectoryStep
from evaluation.replay_eval import IPSConfig, evaluate_ips, ips_estimate
from policies.base import Policy


class FixedPolicy(Policy):
    def __init__(self, action: Action, propensity: float = 1.0) -> None:
        self.action = action
        self.propensity = propensity

    def select_action(self, context: Context) -> tuple[Action, float]:
        return self.action, self.propensity

    def update(self, context: Context, action: Action, reward: Reward) -> None:
        raise AssertionError("replay evaluation must not train policies")

    def reset(self) -> None:
        pass

    @property
    def policy_id(self) -> str:
        return f"fixed_{self.action.value}"


def _ctx(step: int = 0) -> Context:
    return Context(
        commit_sha=f"sha-{step}",
        project_slug="owner/repo",
        step=step,
        files_changed=1,
        lines_added=10,
        lines_deleted=2,
        src_churn=12,
        is_pr=False,
        tests_run=20,
        tests_added=1,
        build_duration_s=30.0,
        author_experience=3,
        recent_failure_rate=0.1,
        has_dependency_change=False,
        has_risky_path_change=False,
    )


def _reward(
    action_id: str,
    cost: float,
    outcome: Outcome = Outcome.SUCCESS,
    censored: bool = False,
) -> Reward:
    return Reward(
        action_id=action_id,
        outcome=outcome,
        cost=cost,
        delay_steps=1,
        censored=censored,
        observed_at_step=1,
    )


def _step(
    index: int,
    logged_action: Action,
    cost: float,
    propensity: float | None = 1.0,
    outcome: Outcome = Outcome.SUCCESS,
    censored: bool = False,
) -> TrajectoryStep:
    return TrajectoryStep(
        context=_ctx(index),
        action=logged_action,
        propensity=propensity,
        reward=_reward(
            action_id=f"a-{index}",
            cost=cost,
            outcome=outcome,
            censored=censored,
        ),
    )


def _trajectory(*steps: TrajectoryStep) -> Trajectory:
    return Trajectory(
        trajectory_id="t1",
        project_slug="owner/repo",
        policy_id="logged",
        drift_segment_id=None,
        steps=steps,
    )


def test_ips_matches_logged_policy_when_actions_match() -> None:
    trajectory = _trajectory(
        _step(0, Action.DEPLOY, cost=2.0, propensity=1.0),
        _step(1, Action.DEPLOY, cost=3.0, propensity=1.0),
    )

    result = evaluate_ips(FixedPolicy(Action.DEPLOY), trajectory)

    assert result.estimated_cumulative_cost == 5.0
    assert result.estimated_mean_cost == 2.5
    assert result.estimated_policy_value == -2.5
    assert result.matched_actions == 2
    assert result.effective_sample_size == 2.0
    assert ips_estimate(FixedPolicy(Action.DEPLOY), trajectory) == -2.5


def test_mismatched_actions_contribute_zero_weight() -> None:
    trajectory = _trajectory(
        _step(0, Action.DEPLOY, cost=10.0, propensity=1.0),
    )

    result = evaluate_ips(FixedPolicy(Action.BLOCK), trajectory)

    assert result.estimated_cumulative_cost == 0.0
    assert result.estimated_mean_cost == 0.0
    assert result.matched_actions == 0
    assert result.effective_sample_size == 0.0


def test_low_propensity_increases_weight() -> None:
    high_propensity = _trajectory(
        _step(0, Action.CANARY, cost=2.0, propensity=1.0),
    )
    low_propensity = _trajectory(
        _step(0, Action.CANARY, cost=2.0, propensity=0.25),
    )

    high_result = evaluate_ips(FixedPolicy(Action.CANARY), high_propensity)
    low_result = evaluate_ips(FixedPolicy(Action.CANARY), low_propensity)

    assert high_result.estimated_cumulative_cost == 2.0
    assert low_result.estimated_cumulative_cost == 8.0
    assert low_result.weight_sum == 4.0


def test_missing_propensity_path_is_explicit() -> None:
    trajectory = _trajectory(
        _step(0, Action.BLOCK, cost=2.0, propensity=None),
    )

    with pytest.raises(ValueError, match="Logged propensity is missing"):
        evaluate_ips(FixedPolicy(Action.BLOCK), trajectory)

    result = evaluate_ips(
        FixedPolicy(Action.BLOCK),
        trajectory,
        IPSConfig(allow_missing_propensity_for_deterministic=True),
    )

    assert result.used_default_propensity == 1
    assert result.estimated_cumulative_cost == 2.0


def test_censored_outcomes_are_skipped_consistently() -> None:
    trajectory = _trajectory(
        _step(
            0,
            Action.DEPLOY,
            cost=999.0,
            propensity=1.0,
            outcome=Outcome.CENSORED,
            censored=True,
        ),
        _step(1, Action.DEPLOY, cost=4.0, propensity=1.0),
    )

    result = evaluate_ips(FixedPolicy(Action.DEPLOY), trajectory)

    assert result.skipped_censored == 1
    assert result.evaluated_steps == 1
    assert result.estimated_cumulative_cost == 4.0
    assert result.estimated_policy_value == -4.0
