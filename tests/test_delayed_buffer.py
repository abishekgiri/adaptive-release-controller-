"""Tests for RewardBuffer delay correctness, reward joins, and censoring behaviour."""

from __future__ import annotations

import pytest

from data.schemas import Action, Outcome, Reward
from delayed.buffer import PendingReward, RewardBuffer


def _make_pending(action_id: str, reveal_at: int) -> PendingReward:
    from data.schemas import Context
    ctx = Context(
        commit_sha="abc", project_slug="p/r", step=0,
        files_changed=1, lines_added=1, lines_deleted=0, src_churn=1,
        is_pr=False, tests_run=10, tests_added=0, build_duration_s=30.0,
        author_experience=1, recent_failure_rate=0.0,
        has_dependency_change=False, has_risky_path_change=False,
    )
    reward = Reward(
        action_id=action_id, outcome=Outcome.SUCCESS,
        cost=0.0, delay_steps=reveal_at, censored=False, observed_at_step=reveal_at,
    )
    return PendingReward(
        action_id=action_id, context=ctx, action=Action.DEPLOY,
        reward=reward, reveal_at_step=reveal_at,
    )


def test_flush_returns_only_matured_rewards() -> None:
    """flush(t) must return only rewards with reveal_at_step <= t."""
    # TODO: push rewards at steps 3, 5, 7; flush at step 5; assert returns steps 3 and 5 only
    pytest.skip("Implement after RewardBuffer.flush() is filled in")


def test_flush_removes_returned_rewards() -> None:
    """Rewards returned by flush() must not appear in a subsequent flush()."""
    # TODO: push one reward; flush once; flush again at same step; assert second flush is empty
    pytest.skip("Implement after RewardBuffer.flush() is filled in")


def test_push_overwrite_by_action_id() -> None:
    """Pushing a duplicate action_id must overwrite the previous entry."""
    # TODO: push action_id='x' at reveal=3; push action_id='x' at reveal=10;
    # flush at step 3 — should return nothing; flush at step 10 — should return 'x'
    pytest.skip("Implement after RewardBuffer.push() is filled in")


def test_cancel_removes_pending_reward() -> None:
    """cancel(action_id) must remove the entry and return it."""
    # TODO: push a reward; cancel it; assert pending_count == 0; assert flush returns nothing
    pytest.skip("Implement after RewardBuffer.cancel() is filled in")


def test_no_reward_delivered_at_action_step() -> None:
    """A reward with delay k > 0 must not be returned at the step the action was taken."""
    # TODO: push reward with reveal_at_step=5 at current_step=0; flush at step 0;
    # assert result is empty
    pytest.skip("Implement after RewardBuffer.flush() is filled in")
