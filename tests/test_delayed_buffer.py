"""Tests for delayed reward buffering and censoring behaviour."""

from __future__ import annotations

import numpy as np

from data.schemas import Action, Context, Outcome, Reward
from delayed.buffer import PendingReward, PendingRewardBuffer, RewardBuffer


def _context(step: int = 0) -> Context:
    return Context(
        commit_sha="abc",
        project_slug="p/r",
        step=step,
        files_changed=1,
        lines_added=1,
        lines_deleted=0,
        src_churn=1,
        is_pr=False,
        tests_run=10,
        tests_added=0,
        build_duration_s=30.0,
        author_experience=1,
        recent_failure_rate=0.0,
        has_dependency_change=False,
        has_risky_path_change=False,
    )


def _make_pending(action_id: str, reveal_at: int) -> PendingReward:
    reward = Reward(
        action_id=action_id,
        outcome=Outcome.SUCCESS,
        cost=0.0,
        delay_steps=reveal_at,
        censored=False,
        observed_at_step=reveal_at,
    )
    return PendingReward(
        action_id=action_id,
        context=_context(),
        action=Action.DEPLOY,
        reward=reward,
        reveal_at_step=reveal_at,
    )


def test_delayed_reward_not_available_immediately() -> None:
    buffer = PendingRewardBuffer(
        rng=np.random.default_rng(1),
        min_delay=2,
        max_delay=2,
    )

    buffer.add(
        "a1",
        _context(),
        Action.DEPLOY,
        Outcome.SUCCESS,
        current_step=0,
        cost=0.0,
    )

    assert buffer.pop_available(0) == []
    assert "a1" in buffer


def test_reward_becomes_available_after_delay() -> None:
    buffer = PendingRewardBuffer(
        rng=np.random.default_rng(1),
        min_delay=2,
        max_delay=2,
    )

    buffer.add(
        "a1",
        _context(),
        Action.DEPLOY,
        Outcome.SUCCESS,
        current_step=0,
        cost=0.0,
    )

    available = buffer.pop_available(2)

    assert len(available) == 1
    assert available[0].action_id == "a1"
    assert available[0].reward.observed_at_step == 2
    assert available[0].reward.cost == 0.0
    assert len(buffer) == 0


def test_multiple_rewards_return_in_order() -> None:
    buffer = PendingRewardBuffer(
        rng=np.random.default_rng(1),
        min_delay=1,
        max_delay=1,
    )

    buffer.add("late", _context(), Action.DEPLOY, Outcome.SUCCESS, 0, delay=3)
    buffer.add("early", _context(), Action.BLOCK, Outcome.BLOCKED, 0, delay=1)
    buffer.add("same_step", _context(), Action.CANARY, Outcome.FAILURE, 0, delay=1)

    available = buffer.pop_available(3)

    assert [item.action_id for item in available] == [
        "early",
        "same_step",
        "late",
    ]


def test_censored_reward_path() -> None:
    buffer = PendingRewardBuffer(
        rng=np.random.default_rng(1),
        min_delay=2,
        max_delay=2,
    )

    pending = buffer.add(
        "a1",
        _context(),
        Action.CANARY,
        Outcome.SUCCESS,
        current_step=0,
        cost=1.0,
    )
    censored = buffer.mark_censored("a1")

    assert pending.reveal_at_step == censored.reveal_at_step
    assert censored.reward.outcome == Outcome.CENSORED
    assert censored.reward.censored is True
    assert buffer.pop_available(1) == []

    available = buffer.pop_available(2)

    assert len(available) == 1
    assert available[0].action_id == "a1"
    assert available[0].reward.censored is True


def test_deterministic_behavior_with_fixed_seed() -> None:
    def sample_steps(seed: int) -> list[int]:
        buffer = PendingRewardBuffer(
            rng=np.random.default_rng(seed),
            min_delay=1,
            max_delay=5,
        )
        return [
            buffer.add(
                f"a{index}",
                _context(),
                Action.DEPLOY,
                Outcome.SUCCESS,
                current_step=0,
            ).reveal_at_step
            for index in range(8)
        ]

    assert sample_steps(123) == sample_steps(123)
    assert sample_steps(123) != sample_steps(456)


def test_reward_buffer_flush_and_cancel_compatibility() -> None:
    buffer = RewardBuffer()
    buffer.push(_make_pending("x", reveal_at=3))
    buffer.push(_make_pending("y", reveal_at=5))

    assert [item.action_id for item in buffer.flush(3)] == ["x"]
    assert buffer.cancel("y").action_id == "y"
    assert buffer.pending_count() == 0
