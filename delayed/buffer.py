"""Pending-reward buffer keyed by action_id; gates all policy updates on reward arrival."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from data.schemas import Action, Context, Reward


@dataclass
class PendingReward:
    """A reward that has been scheduled but not yet delivered to the policy."""

    action_id: str
    context: Context
    action: Action
    reward: Reward          # constructed at action time with censored=True if delay unknown
    reveal_at_step: int     # step index at which reward becomes observable


class RewardBuffer:
    """FIFO buffer that holds rewards until their delay has elapsed.

    The experiment loop calls flush(current_step) at the start of each step to
    retrieve all rewards that have matured. Policies must never receive rewards
    outside of this mechanism.

    Invariant: no reward is delivered to policy.update() before step t+k,
    where k is the sampled delay. Violations make evaluation invalid.
    """

    def __init__(self) -> None:
        # TODO: initialise internal pending dict keyed by action_id
        self._pending: dict[str, PendingReward] = {}

    def push(self, pending: PendingReward) -> None:
        """Register a pending reward; overwrites if action_id already present."""
        # TODO: validate action_id uniqueness; store pending reward
        raise NotImplementedError

    def flush(self, current_step: int) -> list[PendingReward]:
        """Return and remove all rewards with reveal_at_step <= current_step."""
        # TODO: filter _pending; remove matured entries; return sorted list
        raise NotImplementedError

    def cancel(self, action_id: str) -> Optional[PendingReward]:
        """Remove and return a pending reward by action_id (used for censoring)."""
        # TODO: pop from _pending; return None if not found
        raise NotImplementedError

    def pending_count(self) -> int:
        """Number of rewards still waiting to mature."""
        return len(self._pending)
