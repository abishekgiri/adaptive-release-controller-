"""Interface conformance tests and smoke runs for all Policy implementations."""

from __future__ import annotations

import pytest

from data.schemas import Action, Context, Outcome, Reward
from policies.base import Policy


# Shared fixture: a minimal valid Context for testing.
def _dummy_context() -> Context:
    return Context(
        commit_sha="abc123",
        project_slug="test/repo",
        step=0,
        files_changed=3,
        lines_added=50,
        lines_deleted=10,
        src_churn=40,
        is_pr=False,
        tests_run=100,
        tests_added=2,
        build_duration_s=90.0,
        author_experience=5,
        recent_failure_rate=0.1,
        has_dependency_change=False,
        has_risky_path_change=False,
    )


def _dummy_reward(action: Action) -> Reward:
    return Reward(
        action_id="test-action-0",
        outcome=Outcome.SUCCESS,
        cost=0.0,
        delay_steps=3,
        censored=False,
        observed_at_step=3,
    )


def _assert_policy_interface(policy: Policy) -> None:
    """Assert that a policy correctly implements the Policy interface contract."""
    ctx = _dummy_context()

    # select_action must return (Action, float)
    action, propensity = policy.select_action(ctx)
    assert isinstance(action, Action), "select_action must return an Action"
    assert 0.0 <= propensity <= 1.0, "propensity must be in [0, 1]"

    # update must not raise with a valid delayed reward
    reward = _dummy_reward(action)
    policy.update(ctx, action, reward)

    # reset must not raise
    policy.reset()

    # policy_id must be a non-empty string
    assert isinstance(policy.policy_id, str) and policy.policy_id, "policy_id must be a non-empty string"


def test_static_rules_interface() -> None:
    pytest.skip("Implement after StaticRulesPolicy.select_action() is filled in")


def test_heuristic_score_interface() -> None:
    pytest.skip("Implement after HeuristicScorePolicy.select_action() is filled in")


def test_linucb_interface() -> None:
    pytest.skip("Implement after LinUCBPolicy is filled in")


def test_thompson_interface() -> None:
    pytest.skip("Implement after ThompsonSamplingPolicy is filled in")


def test_epsilon_greedy_interface() -> None:
    pytest.skip("Implement after EpsilonGreedyPolicy is filled in")


def test_cost_sensitive_bandit_interface() -> None:
    pytest.skip("Implement after CostSensitiveBandit is filled in")


def test_no_policy_receives_immediate_reward() -> None:
    """Reward at step t must never be passed to update() at step t (delay invariant)."""
    # TODO: patch RewardBuffer to track delivery times; assert reward.observed_at_step > action step
    pytest.skip("Implement after RewardBuffer is filled in")
