"""Unit tests for rewards/cost_model.py — every (action, outcome) combination."""

from __future__ import annotations

import math

import pytest

from data.schemas import Action, Outcome
from rewards.cost_model import (
    CENSORED_COST,
    VALID_PAIRS,
    CostConfig,
    compute_cost,
    oracle_cost,
)


# ---------------------------------------------------------------------------
# CostConfig defaults
# ---------------------------------------------------------------------------

class TestCostConfigDefaults:
    """Default CostConfig values must match the paper's cost matrix exactly."""

    def test_deploy_success_default(self) -> None:
        assert CostConfig().deploy_success == 0.0

    def test_deploy_failure_default(self) -> None:
        assert CostConfig().deploy_failure == 10.0

    def test_canary_success_default(self) -> None:
        assert CostConfig().canary_success == 1.0

    def test_canary_failure_default(self) -> None:
        assert CostConfig().canary_failure == 4.0

    def test_block_safe_default(self) -> None:
        assert CostConfig().block_safe == 2.0

    def test_block_bad_default(self) -> None:
        assert CostConfig().block_bad == 0.5

    def test_block_unknown_default(self) -> None:
        assert CostConfig().block_unknown == 2.0

    def test_is_frozen(self) -> None:
        config = CostConfig()
        with pytest.raises((AttributeError, TypeError)):
            config.deploy_failure = 99.0  # type: ignore[misc]

    def test_negative_value_raises(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            CostConfig(deploy_failure=-1.0)

    def test_zero_value_is_valid(self) -> None:
        config = CostConfig(deploy_success=0.0)
        assert config.deploy_success == 0.0


# ---------------------------------------------------------------------------
# compute_cost — default config, every (action, outcome) pair
# ---------------------------------------------------------------------------

class TestComputeCostDefaults:
    """compute_cost with default CostConfig must return the paper's values."""

    _cfg = CostConfig()

    def test_deploy_success(self) -> None:
        assert compute_cost(Action.DEPLOY, Outcome.SUCCESS, self._cfg) == 0.0

    def test_deploy_failure(self) -> None:
        assert compute_cost(Action.DEPLOY, Outcome.FAILURE, self._cfg) == 10.0

    def test_canary_success(self) -> None:
        assert compute_cost(Action.CANARY, Outcome.SUCCESS, self._cfg) == 1.0

    def test_canary_failure(self) -> None:
        assert compute_cost(Action.CANARY, Outcome.FAILURE, self._cfg) == 4.0

    def test_block_safe_change(self) -> None:
        """block + SUCCESS = blocked a safe change → opportunity cost."""
        assert compute_cost(Action.BLOCK, Outcome.SUCCESS, self._cfg) == 2.0

    def test_block_bad_change(self) -> None:
        """block + FAILURE = blocked a risky change → small re-review overhead."""
        assert compute_cost(Action.BLOCK, Outcome.FAILURE, self._cfg) == 0.5

    def test_block_unknown(self) -> None:
        """block + BLOCKED = counterfactual unknown (replay) → conservative estimate."""
        assert compute_cost(Action.BLOCK, Outcome.BLOCKED, self._cfg) == 2.0

    def test_censored_returns_nan(self) -> None:
        for action in Action:
            result = compute_cost(action, Outcome.CENSORED, self._cfg)
            assert math.isnan(result), (
                f"compute_cost({action}, CENSORED) returned {result}, expected nan"
            )


# ---------------------------------------------------------------------------
# compute_cost — cost ordering invariants
# ---------------------------------------------------------------------------

class TestCostOrderingInvariants:
    """Structural relationships the cost matrix must satisfy regardless of config."""

    _cfg = CostConfig()

    def test_deploy_failure_most_expensive(self) -> None:
        """A production outage must be the highest cost across all valid pairs."""
        all_costs = [
            compute_cost(a, o, self._cfg)
            for a, o in VALID_PAIRS
            if o != Outcome.CENSORED and o != Outcome.BLOCKED
        ]
        assert compute_cost(Action.DEPLOY, Outcome.FAILURE, self._cfg) == max(all_costs)

    def test_deploy_success_cheapest(self) -> None:
        """Successful deploy should be the cheapest (or tied for cheapest)."""
        deploy_success = compute_cost(Action.DEPLOY, Outcome.SUCCESS, self._cfg)
        assert deploy_success <= compute_cost(Action.CANARY, Outcome.SUCCESS, self._cfg)
        assert deploy_success <= compute_cost(Action.BLOCK, Outcome.SUCCESS, self._cfg)

    def test_canary_failure_cheaper_than_deploy_failure(self) -> None:
        """Canary limits blast radius: canary_failure < deploy_failure."""
        assert (
            compute_cost(Action.CANARY, Outcome.FAILURE, self._cfg)
            < compute_cost(Action.DEPLOY, Outcome.FAILURE, self._cfg)
        )

    def test_block_bad_cheaper_than_block_safe(self) -> None:
        """Blocking a bad change correctly is cheaper than blocking a safe one."""
        assert (
            compute_cost(Action.BLOCK, Outcome.FAILURE, self._cfg)
            < compute_cost(Action.BLOCK, Outcome.SUCCESS, self._cfg)
        )

    def test_all_valid_costs_non_negative(self) -> None:
        """No valid (action, outcome) pair should produce a negative cost."""
        for action, outcome in VALID_PAIRS:
            if outcome in (Outcome.CENSORED, Outcome.BLOCKED):
                continue
            cost = compute_cost(action, outcome, self._cfg)
            assert cost >= 0.0, f"Negative cost for ({action}, {outcome}): {cost}"


# ---------------------------------------------------------------------------
# compute_cost — custom CostConfig (sweep)
# ---------------------------------------------------------------------------

class TestComputeCostCustomConfig:
    """compute_cost must respect a non-default CostConfig for all pairs."""

    def test_custom_deploy_failure(self) -> None:
        cfg = CostConfig(deploy_failure=20.0)
        assert compute_cost(Action.DEPLOY, Outcome.FAILURE, cfg) == 20.0

    def test_custom_canary_failure(self) -> None:
        cfg = CostConfig(canary_failure=8.0)
        assert compute_cost(Action.CANARY, Outcome.FAILURE, cfg) == 8.0

    def test_custom_block_safe(self) -> None:
        cfg = CostConfig(block_safe=5.0)
        assert compute_cost(Action.BLOCK, Outcome.SUCCESS, cfg) == 5.0

    def test_custom_block_bad(self) -> None:
        cfg = CostConfig(block_bad=0.0)
        assert compute_cost(Action.BLOCK, Outcome.FAILURE, cfg) == 0.0

    def test_symmetric_config(self) -> None:
        """When all costs are equal, every non-censored pair returns that value."""
        cfg = CostConfig(
            deploy_success=1.0, deploy_failure=1.0,
            canary_success=1.0, canary_failure=1.0,
            block_safe=1.0, block_bad=1.0, block_unknown=1.0,
        )
        for action, outcome in VALID_PAIRS:
            if outcome == Outcome.CENSORED:
                continue
            assert compute_cost(action, outcome, cfg) == 1.0


# ---------------------------------------------------------------------------
# compute_cost — invalid pairs
# ---------------------------------------------------------------------------

class TestComputeCostInvalidPairs:
    """Invalid (action, outcome) combinations must raise ValueError."""

    _cfg = CostConfig()

    def test_deploy_blocked_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid"):
            compute_cost(Action.DEPLOY, Outcome.BLOCKED, self._cfg)

    def test_canary_blocked_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid"):
            compute_cost(Action.CANARY, Outcome.BLOCKED, self._cfg)


# ---------------------------------------------------------------------------
# VALID_PAIRS exhaustiveness
# ---------------------------------------------------------------------------

class TestValidPairs:
    """VALID_PAIRS must cover exactly the pairs the cost matrix handles."""

    def test_every_valid_pair_computes_without_error(self) -> None:
        cfg = CostConfig()
        for action, outcome in VALID_PAIRS:
            if outcome == Outcome.CENSORED:
                continue
            cost = compute_cost(action, outcome, cfg)
            assert isinstance(cost, float) or isinstance(cost, int)

    def test_valid_pairs_contains_all_seven_semantic_entries(self) -> None:
        expected = {
            (Action.DEPLOY, Outcome.SUCCESS),
            (Action.DEPLOY, Outcome.FAILURE),
            (Action.CANARY, Outcome.SUCCESS),
            (Action.CANARY, Outcome.FAILURE),
            (Action.BLOCK,  Outcome.SUCCESS),
            (Action.BLOCK,  Outcome.FAILURE),
            (Action.BLOCK,  Outcome.BLOCKED),
        }
        assert expected <= VALID_PAIRS


# ---------------------------------------------------------------------------
# oracle_cost
# ---------------------------------------------------------------------------

class TestOracleCost:
    """Oracle always achieves the minimum cost for a given outcome."""

    _cfg = CostConfig()

    def test_oracle_success_is_deploy_success(self) -> None:
        """With defaults, cheapest action on a successful change is DEPLOY (cost 0)."""
        assert oracle_cost(Outcome.SUCCESS, self._cfg) == 0.0

    def test_oracle_failure_is_block_bad(self) -> None:
        """With defaults, cheapest action on a failing change is BLOCK (cost 0.5)."""
        assert oracle_cost(Outcome.FAILURE, self._cfg) == 0.5

    def test_oracle_censored_returns_nan(self) -> None:
        assert math.isnan(oracle_cost(Outcome.CENSORED, self._cfg))

    def test_oracle_blocked_returns_nan(self) -> None:
        assert math.isnan(oracle_cost(Outcome.BLOCKED, self._cfg))

    def test_oracle_cost_le_any_policy_cost_for_success(self) -> None:
        """Oracle is a lower bound: no policy can do better in expectation."""
        cfg = self._cfg
        for action in (Action.DEPLOY, Action.CANARY, Action.BLOCK):
            policy_cost = compute_cost(action, Outcome.SUCCESS, cfg)
            assert oracle_cost(Outcome.SUCCESS, cfg) <= policy_cost

    def test_oracle_cost_le_any_policy_cost_for_failure(self) -> None:
        cfg = self._cfg
        for action in (Action.DEPLOY, Action.CANARY, Action.BLOCK):
            policy_cost = compute_cost(action, Outcome.FAILURE, cfg)
            assert oracle_cost(Outcome.FAILURE, cfg) <= policy_cost
