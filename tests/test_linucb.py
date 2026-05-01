"""Unit tests for policies/linucb.py — LinUCB disjoint contextual bandit."""

from __future__ import annotations

import numpy as np
import pytest

from data.schemas import Action, Context, Outcome, Reward
from policies.base import FeatureEncoder, Policy
from policies.linucb import LinUCBConfig, LinUCBPolicy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_policy(
    alpha: float = 1.0,
    lambda_reg: float = 1.0,
    feature_dim: int = FeatureEncoder.DIM,
    seed: int = 0,
    policy_id: str = "linucb",
) -> LinUCBPolicy:
    config = LinUCBConfig(alpha=alpha, lambda_reg=lambda_reg)
    rng = np.random.default_rng(seed)
    return LinUCBPolicy(config=config, feature_dim=feature_dim, rng=rng, policy_id=policy_id)


def _ctx(
    files_changed: int = 3,
    src_churn: int = 60,
    recent_failure_rate: float = 0.05,
    has_risky_path_change: bool = False,
    tests_run: int = 100,
) -> Context:
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
        tests_run=tests_run,
        tests_added=2,
        build_duration_s=90.0,
        author_experience=5,
        recent_failure_rate=recent_failure_rate,
        has_dependency_change=False,
        has_risky_path_change=has_risky_path_change,
    )


def _reward(action: Action, cost: float, outcome: Outcome = Outcome.SUCCESS) -> Reward:
    return Reward(
        action_id="test",
        outcome=outcome,
        cost=cost,
        delay_steps=3,
        censored=False,
        observed_at_step=3,
    )


# ---------------------------------------------------------------------------
# Policy interface conformance
# ---------------------------------------------------------------------------

class TestLinUCBInterface:
    """LinUCBPolicy must satisfy the full Policy interface contract."""

    def test_is_policy_subclass(self) -> None:
        assert isinstance(_make_policy(), Policy)

    def test_select_action_returns_action_and_propensity(self) -> None:
        action, propensity = _make_policy().select_action(_ctx())
        assert isinstance(action, Action)
        assert 0.0 <= propensity <= 1.0

    def test_update_does_not_raise(self) -> None:
        p = _make_policy()
        ctx = _ctx()
        action, _ = p.select_action(ctx)
        p.update(ctx, action, _reward(action, cost=2.0))

    def test_reset_does_not_raise(self) -> None:
        _make_policy().reset()

    def test_policy_id_is_string(self) -> None:
        p = _make_policy(policy_id="my_linucb")
        assert p.policy_id == "my_linucb"

    def test_propensity_is_one(self) -> None:
        """Greedy argmax policy must always report propensity 1.0."""
        p = _make_policy()
        for ctx in [_ctx(), _ctx(recent_failure_rate=0.5), _ctx(files_changed=30)]:
            _, prop = p.select_action(ctx)
            assert prop == 1.0


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

class TestLinUCBInit:
    """Per-arm matrices must be initialised correctly."""

    def test_A_initialised_as_lambda_identity(self) -> None:
        lam = 2.5
        p = _make_policy(lambda_reg=lam)
        d = FeatureEncoder.DIM
        for action in Action:
            np.testing.assert_array_equal(p._A[action], lam * np.eye(d))

    def test_b_initialised_as_zeros(self) -> None:
        p = _make_policy()
        d = FeatureEncoder.DIM
        for action in Action:
            np.testing.assert_array_equal(p._b[action], np.zeros(d))

    def test_all_three_arms_present(self) -> None:
        p = _make_policy()
        for action in Action:
            assert action in p._A
            assert action in p._b

    def test_custom_feature_dim(self) -> None:
        p = _make_policy(feature_dim=5)
        for action in Action:
            assert p._A[action].shape == (5, 5)
            assert p._b[action].shape == (5,)


# ---------------------------------------------------------------------------
# Valid action output
# ---------------------------------------------------------------------------

class TestLinUCBValidAction:
    """select_action must always return a valid Action, regardless of model state."""

    def test_fresh_model_returns_valid_action(self) -> None:
        action, _ = _make_policy().select_action(_ctx())
        assert action in Action

    def test_action_after_multiple_updates(self) -> None:
        p = _make_policy()
        ctx = _ctx()
        for _ in range(10):
            p.update(ctx, Action.DEPLOY, _reward(Action.DEPLOY, cost=0.0))
        action, _ = p.select_action(ctx)
        assert action in Action

    def test_action_after_reset(self) -> None:
        p = _make_policy()
        ctx = _ctx()
        p.update(ctx, Action.CANARY, _reward(Action.CANARY, cost=1.0))
        p.reset()
        action, _ = p.select_action(ctx)
        assert action in Action


# ---------------------------------------------------------------------------
# Matrix updates after reward
# ---------------------------------------------------------------------------

class TestLinUCBMatrixUpdates:
    """A and b must change only for the updated arm, and change correctly."""

    def test_A_grows_after_update(self) -> None:
        p = _make_policy()
        ctx = _ctx()
        A_before = p._A[Action.DEPLOY].copy()
        p.update(ctx, Action.DEPLOY, _reward(Action.DEPLOY, cost=0.0))
        assert not np.allclose(p._A[Action.DEPLOY], A_before), \
            "A[DEPLOY] must grow after an update"

    def test_b_changes_after_nonzero_reward(self) -> None:
        p = _make_policy()
        ctx = _ctx()
        p.update(ctx, Action.DEPLOY, _reward(Action.DEPLOY, cost=5.0))  # r = -5
        assert not np.allclose(p._b[Action.DEPLOY], np.zeros(FeatureEncoder.DIM)), \
            "b[DEPLOY] must change after a nonzero reward"

    def test_b_direction_reflects_reward_sign(self) -> None:
        """Positive reward (low cost) must push b in the direction of x."""
        p = _make_policy()
        ctx = _ctx()
        x = FeatureEncoder().encode(ctx)
        p.update(ctx, Action.DEPLOY, _reward(Action.DEPLOY, cost=-1.0))  # r = +1
        # b[DEPLOY] should equal x (lambda*I => only one update => b = r*x = x)
        np.testing.assert_allclose(p._b[Action.DEPLOY], x)

    def test_only_updated_arm_changes(self) -> None:
        """Updating DEPLOY must not touch CANARY or BLOCK matrices."""
        p = _make_policy()
        ctx = _ctx()
        A_canary_before = p._A[Action.CANARY].copy()
        b_canary_before = p._b[Action.CANARY].copy()
        A_block_before = p._A[Action.BLOCK].copy()
        b_block_before = p._b[Action.BLOCK].copy()

        p.update(ctx, Action.DEPLOY, _reward(Action.DEPLOY, cost=2.0))

        np.testing.assert_array_equal(p._A[Action.CANARY], A_canary_before)
        np.testing.assert_array_equal(p._b[Action.CANARY], b_canary_before)
        np.testing.assert_array_equal(p._A[Action.BLOCK], A_block_before)
        np.testing.assert_array_equal(p._b[Action.BLOCK], b_block_before)

    def test_A_update_formula(self) -> None:
        """A[action] must equal A_old + x x^T after one update."""
        p = _make_policy(lambda_reg=1.0)
        ctx = _ctx()
        x = FeatureEncoder().encode(ctx)
        A_before = p._A[Action.BLOCK].copy()
        p.update(ctx, Action.BLOCK, _reward(Action.BLOCK, cost=0.0))
        np.testing.assert_allclose(p._A[Action.BLOCK], A_before + np.outer(x, x))

    def test_b_update_formula(self) -> None:
        """b[action] must equal b_old + r * x after one update."""
        p = _make_policy()
        ctx = _ctx()
        x = FeatureEncoder().encode(ctx)
        cost = 3.0
        r = -cost
        b_before = p._b[Action.CANARY].copy()
        p.update(ctx, Action.CANARY, _reward(Action.CANARY, cost=cost))
        np.testing.assert_allclose(p._b[Action.CANARY], b_before + r * x)


# ---------------------------------------------------------------------------
# Exploration term
# ---------------------------------------------------------------------------

class TestLinUCBExploration:
    """α must meaningfully change which arm is selected."""

    def test_exploration_changes_selection_after_one_arm_updated(self) -> None:
        """After updating CANARY with a positive reward:
        - α=0 (exploitation only): CANARY wins (positive θ^T x vs 0 for others).
        - α=large (exploration dominant): an unexplored arm wins because its
          confidence interval (x^T A^{-1} x) is wider (A = I vs A > I for CANARY).
        """
        ctx = _ctx()

        p_exploit = _make_policy(alpha=0.0)
        p_explore = _make_policy(alpha=100.0)

        # Update CANARY with a positive reward (r = -cost = -(-1) = 1)
        good_canary = _reward(Action.CANARY, cost=-1.0)
        p_exploit.update(ctx, Action.CANARY, good_canary)
        p_explore.update(ctx, Action.CANARY, good_canary)

        action_exploit, _ = p_exploit.select_action(ctx)
        action_explore, _ = p_explore.select_action(ctx)

        # Exploitation picks CANARY: θ_canary^T x > 0, others have θ=0
        assert action_exploit == Action.CANARY, (
            f"α=0 should exploit the updated CANARY arm, got {action_exploit}"
        )
        # Exploration avoids CANARY: unexplored arms have larger confidence bounds
        assert action_explore != Action.CANARY, (
            f"α=100 should prefer an unexplored arm over CANARY, got {action_explore}"
        )

    def test_alpha_zero_is_pure_exploitation(self) -> None:
        """With α=0, the UCB score reduces to θ^T x with no exploration bonus."""
        ctx = _ctx()
        p = _make_policy(alpha=0.0)
        # Give BLOCK a strongly positive reward → θ_block^T x > 0
        p.update(ctx, Action.BLOCK, _reward(Action.BLOCK, cost=-5.0))  # r = 5
        action, _ = p.select_action(ctx)
        assert action == Action.BLOCK, (
            f"α=0: pure exploitation should pick BLOCK after high reward, got {action}"
        )

    def test_alpha_large_prefers_unexplored_arms(self) -> None:
        """With very large α, an arm never updated (A=I) beats one updated many times."""
        ctx = _ctx()
        p = _make_policy(alpha=1000.0)
        # Flood DEPLOY and CANARY with neutral updates → A grows, uncertainty shrinks
        for _ in range(50):
            p.update(ctx, Action.DEPLOY, _reward(Action.DEPLOY, cost=0.0))
            p.update(ctx, Action.CANARY, _reward(Action.CANARY, cost=0.0))
        # BLOCK has A=I (never updated) → maximal exploration bonus → should win
        action, _ = p.select_action(ctx)
        assert action == Action.BLOCK, (
            f"α=1000: BLOCK (never updated) should dominate via exploration, got {action}"
        )


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

class TestLinUCBDeterminism:
    """select_action uses no RNG → fully deterministic given the same model state."""

    def test_same_context_twice_gives_same_action(self) -> None:
        p = _make_policy(seed=42)
        ctx = _ctx()
        a1, p1 = p.select_action(ctx)
        a2, p2 = p.select_action(ctx)
        assert a1 == a2
        assert p1 == p2

    def test_different_seeds_give_same_action(self) -> None:
        """Action selection is deterministic and does not use the RNG."""
        ctx = _ctx()
        a1, _ = _make_policy(seed=0).select_action(ctx)
        a2, _ = _make_policy(seed=99).select_action(ctx)
        assert a1 == a2

    def test_deterministic_after_identical_update_sequence(self) -> None:
        p1 = _make_policy(seed=0)
        p2 = _make_policy(seed=7)
        ctx = _ctx()
        rwd = _reward(Action.DEPLOY, cost=2.0)
        p1.update(ctx, Action.DEPLOY, rwd)
        p2.update(ctx, Action.DEPLOY, rwd)
        a1, _ = p1.select_action(ctx)
        a2, _ = p2.select_action(ctx)
        assert a1 == a2


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------

class TestLinUCBReset:
    """reset() must restore the model to its initial state."""

    def test_reset_restores_A_to_identity(self) -> None:
        d = FeatureEncoder.DIM
        p = _make_policy(lambda_reg=1.0)
        ctx = _ctx()
        p.update(ctx, Action.DEPLOY, _reward(Action.DEPLOY, cost=3.0))
        p.reset()
        np.testing.assert_array_equal(p._A[Action.DEPLOY], np.eye(d))

    def test_reset_restores_b_to_zeros(self) -> None:
        d = FeatureEncoder.DIM
        p = _make_policy()
        ctx = _ctx()
        p.update(ctx, Action.CANARY, _reward(Action.CANARY, cost=1.0))
        p.reset()
        np.testing.assert_array_equal(p._b[Action.CANARY], np.zeros(d))

    def test_reset_preserves_policy_id(self) -> None:
        p = _make_policy(policy_id="test_linucb")
        p.reset()
        assert p.policy_id == "test_linucb"

    def test_action_after_reset_matches_fresh_policy(self) -> None:
        ctx = _ctx()
        fresh = _make_policy()
        updated = _make_policy()
        updated.update(ctx, Action.DEPLOY, _reward(Action.DEPLOY, cost=10.0))
        updated.reset()
        a_fresh, _ = fresh.select_action(ctx)
        a_reset, _ = updated.select_action(ctx)
        assert a_fresh == a_reset


# ---------------------------------------------------------------------------
# Feature encoder
# ---------------------------------------------------------------------------

class TestFeatureEncoder:
    """FeatureEncoder must produce consistently shaped, finite vectors."""

    def test_output_shape(self) -> None:
        x = FeatureEncoder().encode(_ctx())
        assert x.shape == (FeatureEncoder.DIM,)

    def test_output_dtype(self) -> None:
        x = FeatureEncoder().encode(_ctx())
        assert x.dtype == np.float64

    def test_bias_term_is_one(self) -> None:
        x = FeatureEncoder().encode(_ctx())
        assert x[-1] == 1.0, "Last element must be the bias term (1.0)"

    def test_all_finite(self) -> None:
        x = FeatureEncoder().encode(_ctx())
        assert np.all(np.isfinite(x)), "Feature vector must contain no NaN or Inf"

    def test_same_context_same_vector(self) -> None:
        enc = FeatureEncoder()
        ctx = _ctx()
        np.testing.assert_array_equal(enc.encode(ctx), enc.encode(ctx))
