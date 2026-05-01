"""Interface conformance tests and smoke runs for all Policy implementations."""

from __future__ import annotations

import pytest

from data.schemas import Action, Context, Outcome, Reward
import numpy as np

from policies.base import FeatureEncoder, Policy
from policies.linucb import LinUCBConfig, LinUCBPolicy
from policies.static_rules import StaticRulesPolicy


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _ctx(
    files_changed: int = 3,
    src_churn: int = 40,
    recent_failure_rate: float = 0.05,
    has_risky_path_change: bool = False,
    has_dependency_change: bool = False,
    author_experience: int = 5,
) -> Context:
    """Build a Context, parameterised on the fields StaticRulesPolicy uses."""
    lines_added = src_churn // 2
    lines_deleted = src_churn - lines_added
    return Context(
        commit_sha="abc123",
        project_slug="test/repo",
        step=0,
        files_changed=files_changed,
        lines_added=lines_added,
        lines_deleted=lines_deleted,
        src_churn=src_churn,
        is_pr=False,
        tests_run=100,
        tests_added=2,
        build_duration_s=90.0,
        author_experience=author_experience,
        recent_failure_rate=recent_failure_rate,
        has_dependency_change=has_dependency_change,
        has_risky_path_change=has_risky_path_change,
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
    ctx = _ctx()

    action, propensity = policy.select_action(ctx)
    assert isinstance(action, Action), "select_action must return an Action"
    assert 0.0 <= propensity <= 1.0, "propensity must be in [0, 1]"

    reward = _dummy_reward(action)
    policy.update(ctx, action, reward)

    policy.reset()

    assert isinstance(policy.policy_id, str) and policy.policy_id, (
        "policy_id must be a non-empty string"
    )


# ---------------------------------------------------------------------------
# StaticRulesPolicy — interface conformance
# ---------------------------------------------------------------------------

class TestStaticRulesPolicyInterface:
    """StaticRulesPolicy must satisfy the full Policy interface."""

    def test_interface_conformance(self) -> None:
        _assert_policy_interface(StaticRulesPolicy())

    def test_is_policy_subclass(self) -> None:
        assert isinstance(StaticRulesPolicy(), Policy)

    def test_default_policy_id(self) -> None:
        assert StaticRulesPolicy().policy_id == "static_rules"

    def test_custom_policy_id(self) -> None:
        p = StaticRulesPolicy(policy_id="my_static")
        assert p.policy_id == "my_static"

    def test_propensity_is_one(self) -> None:
        """Deterministic policy must always return propensity 1.0."""
        for ctx in [
            _ctx(),
            _ctx(recent_failure_rate=0.5),
            _ctx(has_risky_path_change=True),
        ]:
            _, propensity = StaticRulesPolicy().select_action(ctx)
            assert propensity == 1.0


# ---------------------------------------------------------------------------
# StaticRulesPolicy — valid action output
# ---------------------------------------------------------------------------

class TestStaticRulesPolicyValidActions:
    """select_action must always return one of the three valid Action values."""

    _policy = StaticRulesPolicy()

    def test_low_risk_returns_action(self) -> None:
        action, _ = self._policy.select_action(_ctx())
        assert action in Action

    def test_medium_risk_returns_action(self) -> None:
        action, _ = self._policy.select_action(_ctx(files_changed=12))
        assert action in Action

    def test_high_risk_returns_action(self) -> None:
        action, _ = self._policy.select_action(_ctx(recent_failure_rate=0.9))
        assert action in Action

    def test_action_is_action_enum(self) -> None:
        action, _ = self._policy.select_action(_ctx())
        assert isinstance(action, Action)


# ---------------------------------------------------------------------------
# StaticRulesPolicy — determinism
# ---------------------------------------------------------------------------

class TestStaticRulesPolicyDeterminism:
    """Calling select_action twice on the same context must yield the same action."""

    _policy = StaticRulesPolicy()

    def _both_calls_equal(self, ctx: Context) -> None:
        a1, p1 = self._policy.select_action(ctx)
        a2, p2 = self._policy.select_action(ctx)
        assert a1 == a2, f"Non-deterministic action: {a1} vs {a2}"
        assert p1 == p2

    def test_deterministic_low_risk(self) -> None:
        self._both_calls_equal(_ctx())

    def test_deterministic_medium_risk(self) -> None:
        self._both_calls_equal(_ctx(files_changed=15, recent_failure_rate=0.20))

    def test_deterministic_high_risk(self) -> None:
        self._both_calls_equal(_ctx(recent_failure_rate=0.50))

    def test_deterministic_after_update(self) -> None:
        """update() must not change the deterministic output."""
        ctx = _ctx()
        action_before, _ = self._policy.select_action(ctx)
        reward = _dummy_reward(action_before)
        self._policy.update(ctx, action_before, reward)
        action_after, _ = self._policy.select_action(ctx)
        assert action_before == action_after

    def test_deterministic_after_reset(self) -> None:
        """reset() must not change the deterministic output."""
        ctx = _ctx()
        action_before, _ = self._policy.select_action(ctx)
        self._policy.reset()
        action_after, _ = self._policy.select_action(ctx)
        assert action_before == action_after


# ---------------------------------------------------------------------------
# StaticRulesPolicy — does not require reward or outcome
# ---------------------------------------------------------------------------

class TestStaticRulesNeedsNoReward:
    """Policy must work without any reward information."""

    def test_select_action_needs_no_reward(self) -> None:
        """select_action must not raise when called with context only."""
        policy = StaticRulesPolicy()
        action, propensity = policy.select_action(_ctx())
        assert isinstance(action, Action)

    def test_update_with_any_outcome_does_not_change_action(self) -> None:
        """update() is a no-op; passing any outcome must not affect future actions."""
        policy = StaticRulesPolicy()
        ctx = _ctx()
        action_before, _ = policy.select_action(ctx)

        for outcome in (Outcome.SUCCESS, Outcome.FAILURE, Outcome.CENSORED):
            reward = Reward(
                action_id="x",
                outcome=outcome,
                cost=5.0,
                delay_steps=1,
                censored=(outcome == Outcome.CENSORED),
                observed_at_step=1,
            )
            policy.update(ctx, action_before, reward)

        action_after, _ = policy.select_action(ctx)
        assert action_before == action_after

    def test_update_with_no_prior_call_does_not_raise(self) -> None:
        policy = StaticRulesPolicy()
        ctx = _ctx()
        reward = _dummy_reward(Action.DEPLOY)
        policy.update(ctx, Action.DEPLOY, reward)  # must not raise


# ---------------------------------------------------------------------------
# StaticRulesPolicy — decision semantics
# ---------------------------------------------------------------------------

class TestStaticRulesDecisionSemantics:
    """Obvious risk signals must map to the correct action tier."""

    _policy = StaticRulesPolicy()

    # --- DEPLOY (low risk) ---

    def test_obvious_safe_change_deploys(self) -> None:
        """Small change, low failure rate, no risky paths → DEPLOY."""
        ctx = _ctx(
            files_changed=2,
            src_churn=30,
            recent_failure_rate=0.05,
            has_risky_path_change=False,
        )
        action, _ = self._policy.select_action(ctx)
        assert action == Action.DEPLOY

    def test_zero_failure_rate_deploys(self) -> None:
        ctx = _ctx(recent_failure_rate=0.0, files_changed=1, src_churn=10)
        action, _ = self._policy.select_action(ctx)
        assert action == Action.DEPLOY

    def test_small_pr_deploys(self) -> None:
        ctx = _ctx(files_changed=3, src_churn=60, recent_failure_rate=0.05)
        action, _ = self._policy.select_action(ctx)
        assert action == Action.DEPLOY

    # --- CANARY (medium risk) ---

    def test_risky_path_canaries(self) -> None:
        """has_risky_path_change alone (small change) → CANARY, not BLOCK."""
        ctx = _ctx(
            files_changed=5,
            src_churn=100,
            recent_failure_rate=0.05,
            has_risky_path_change=True,
        )
        action, _ = self._policy.select_action(ctx)
        assert action == Action.CANARY

    def test_medium_file_count_canaries(self) -> None:
        ctx = _ctx(files_changed=12, src_churn=200, recent_failure_rate=0.05)
        action, _ = self._policy.select_action(ctx)
        assert action == Action.CANARY

    def test_medium_churn_canaries(self) -> None:
        ctx = _ctx(files_changed=5, src_churn=500, recent_failure_rate=0.05)
        action, _ = self._policy.select_action(ctx)
        assert action == Action.CANARY

    def test_elevated_failure_rate_canaries(self) -> None:
        ctx = _ctx(recent_failure_rate=0.20, files_changed=3, src_churn=50)
        action, _ = self._policy.select_action(ctx)
        assert action == Action.CANARY

    # --- BLOCK (high risk) ---

    def test_high_failure_rate_blocks(self) -> None:
        """recent_failure_rate > 0.30 → BLOCK."""
        ctx = _ctx(recent_failure_rate=0.50)
        action, _ = self._policy.select_action(ctx)
        assert action == Action.BLOCK

    def test_risky_path_plus_large_files_blocks(self) -> None:
        """Risky path + large file count → BLOCK (not just CANARY)."""
        ctx = _ctx(
            files_changed=25,
            src_churn=200,
            recent_failure_rate=0.05,
            has_risky_path_change=True,
        )
        action, _ = self._policy.select_action(ctx)
        assert action == Action.BLOCK

    def test_risky_path_plus_large_churn_blocks(self) -> None:
        ctx = _ctx(
            files_changed=5,
            src_churn=1200,
            recent_failure_rate=0.05,
            has_risky_path_change=True,
        )
        action, _ = self._policy.select_action(ctx)
        assert action == Action.BLOCK

    def test_very_high_failure_rate_blocks(self) -> None:
        ctx = _ctx(recent_failure_rate=0.99)
        action, _ = self._policy.select_action(ctx)
        assert action == Action.BLOCK

    # --- Tier boundaries ---

    def test_failure_rate_at_high_threshold_blocks(self) -> None:
        """Exactly at the high threshold (>0.30) must block."""
        ctx = _ctx(recent_failure_rate=0.31, files_changed=2, src_churn=30)
        action, _ = self._policy.select_action(ctx)
        assert action == Action.BLOCK

    def test_failure_rate_below_high_threshold_does_not_block(self) -> None:
        """Just below 0.30 with no other risk signals → at most CANARY."""
        ctx = _ctx(recent_failure_rate=0.29, files_changed=2, src_churn=30)
        action, _ = self._policy.select_action(ctx)
        assert action != Action.BLOCK


# ---------------------------------------------------------------------------
# Skeletons for policies not yet implemented
# ---------------------------------------------------------------------------

def test_heuristic_score_interface() -> None:
    pytest.skip("Implement after HeuristicScorePolicy.select_action() is filled in")


def test_linucb_interface() -> None:
    config = LinUCBConfig(alpha=1.0)
    rng = np.random.default_rng(0)
    policy = LinUCBPolicy(config=config, feature_dim=FeatureEncoder.DIM, rng=rng)
    _assert_policy_interface(policy)


def test_thompson_interface() -> None:
    pytest.skip("Implement after ThompsonSamplingPolicy is filled in")


def test_epsilon_greedy_interface() -> None:
    pytest.skip("Implement after EpsilonGreedyPolicy is filled in")


def test_cost_sensitive_bandit_interface() -> None:
    pytest.skip("Implement after CostSensitiveBandit is filled in")


def test_no_policy_receives_immediate_reward() -> None:
    """Reward at step t must never be passed to update() at step t (delay invariant)."""
    pytest.skip("Implement after RewardBuffer is filled in")
