"""Pending-reward buffer keyed by action_id; gates policy updates on reward arrival.

This module intentionally contains no policy, environment, or cost-model logic.
Callers provide already-computed outcome/cost values, and the buffer only decides
when the delayed reward is visible.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace

import numpy as np

from data.schemas import Action, Context, Outcome, Reward


DelaySampler = Callable[[np.random.Generator], int]


@dataclass(frozen=True)
class PendingReward:
    """A reward that has been scheduled but not yet delivered to the policy."""

    action_id: str
    context: Context
    action: Action
    reward: Reward
    reveal_at_step: int

    @property
    def scheduled_step(self) -> int:
        """Alias used by the new delayed-feedback API."""

        return self.reveal_at_step

    @property
    def censored(self) -> bool:
        """Return whether the reward was censored before observation."""

        return self.reward.censored


class PendingRewardBuffer:
    """Buffer pending delayed rewards keyed by action id.

    The experiment loop should call :meth:`pop_available` or :meth:`flush` at the
    start of each step and only update policies with returned rewards.

    Delay model (choose one):
      - ``delay_p``: geometric delay k ~ Geom(p) as in the formal model. This is
        the canonical choice matching problem-formulation.md (default p=0.3).
      - ``min_delay`` / ``max_delay``: uniform integer delay for ablation studies.
      - ``delay_sampler``: arbitrary callable for custom distributions.
    """

    def __init__(
        self,
        *,
        rng: np.random.Generator,
        min_delay: int = 1,
        max_delay: int = 1,
        delay_sampler: DelaySampler | None = None,
        delay_p: float | None = None,
    ) -> None:
        if not isinstance(rng, np.random.Generator):
            raise TypeError("rng must be a numpy.random.Generator")
        if min_delay < 0:
            raise ValueError("min_delay must be non-negative")
        if max_delay < min_delay:
            raise ValueError("max_delay must be greater than or equal to min_delay")
        if delay_p is not None and not (0.0 < delay_p <= 1.0):
            raise ValueError("delay_p must be in (0, 1]")

        self._rng = rng
        self._min_delay = min_delay
        self._max_delay = max_delay
        self._delay_sampler = delay_sampler
        self._delay_p = delay_p
        self._pending: dict[str, PendingReward] = {}
        self._insertion_order: dict[str, int] = {}
        self._next_sequence = 0
        self.current_step = 0

    def add(
        self,
        action_id: str,
        context: Context,
        action: Action,
        outcome: Outcome,
        current_step: int,
        *,
        cost: float = 0.0,
        delay: int | None = None,
        censored: bool = False,
    ) -> PendingReward:
        """Schedule a computed reward for delayed observation."""

        if action_id in self._pending:
            raise ValueError(f"action_id already pending: {action_id}")
        if current_step < 0:
            raise ValueError("current_step must be non-negative")

        delay_steps = self._sample_delay() if delay is None else delay
        if delay_steps < 0:
            raise ValueError("delay must be non-negative")

        observed_at_step = current_step + delay_steps
        reward = Reward(
            action_id=action_id,
            outcome=outcome,
            cost=cost,
            delay_steps=delay_steps,
            censored=censored,
            observed_at_step=observed_at_step,
        )
        pending = PendingReward(
            action_id=action_id,
            context=context,
            action=action,
            reward=reward,
            reveal_at_step=observed_at_step,
        )
        self._store(pending, overwrite=False)
        return pending

    def advance_to(self, step: int) -> list[PendingReward]:
        """Advance logical time and return matured rewards without removing them."""

        if step < self.current_step:
            raise ValueError("step cannot move backwards")
        self.current_step = step
        return self.available(step)

    def available(self, step: int | None = None) -> list[PendingReward]:
        """Return matured rewards without removing them."""

        effective_step = self.current_step if step is None else step
        return self._matured(effective_step)

    def pop_available(self, step: int | None = None) -> list[PendingReward]:
        """Return and remove rewards whose scheduled step has arrived."""

        effective_step = self.current_step if step is None else step
        self.advance_to(effective_step)
        matured = self._matured(effective_step)
        for pending in matured:
            self._pending.pop(pending.action_id, None)
            self._insertion_order.pop(pending.action_id, None)
        return matured

    def mark_censored(self, action_id: str) -> PendingReward:
        """Mark a pending reward as censored without making it available early."""

        if action_id not in self._pending:
            raise KeyError(action_id)

        pending = self._pending[action_id]
        censored_reward = replace(
            pending.reward,
            outcome=Outcome.CENSORED,
            censored=True,
        )
        censored_pending = replace(pending, reward=censored_reward)
        self._pending[action_id] = censored_pending
        return censored_pending

    def push(self, pending: PendingReward) -> None:
        """Register a pre-built pending reward, overwriting duplicate action ids."""

        self._store(pending, overwrite=True)

    def flush(self, current_step: int) -> list[PendingReward]:
        """Compatibility wrapper for :meth:`pop_available`."""

        return self.pop_available(current_step)

    def cancel(self, action_id: str) -> PendingReward | None:
        """Remove and return a pending reward by action id."""

        self._insertion_order.pop(action_id, None)
        return self._pending.pop(action_id, None)

    def pending_count(self) -> int:
        """Number of rewards still waiting to mature."""

        return len(self._pending)

    def get(self, action_id: str) -> PendingReward | None:
        """Return a pending reward by action id without removing it."""

        return self._pending.get(action_id)

    def __contains__(self, action_id: object) -> bool:
        return action_id in self._pending

    def __len__(self) -> int:
        return len(self._pending)

    def _store(self, pending: PendingReward, *, overwrite: bool) -> None:
        if not overwrite and pending.action_id in self._pending:
            raise ValueError(f"action_id already pending: {pending.action_id}")

        self._pending[pending.action_id] = pending
        if pending.action_id not in self._insertion_order:
            self._insertion_order[pending.action_id] = self._next_sequence
            self._next_sequence += 1

    def _sample_delay(self) -> int:
        if self._delay_sampler is not None:
            delay = int(self._delay_sampler(self._rng))
        elif self._delay_p is not None:
            # Geometric delay k ~ Geom(p): canonical model from problem-formulation.md.
            # numpy geometric is 1-indexed; subtract 1 for 0-indexed steps.
            delay = int(self._rng.geometric(p=self._delay_p)) - 1
        elif self._min_delay == self._max_delay:
            delay = self._min_delay
        else:
            delay = int(self._rng.integers(self._min_delay, self._max_delay + 1))

        if delay < 0:
            raise ValueError("sampled delay must be non-negative")
        return delay

    def _matured(self, step: int) -> list[PendingReward]:
        return sorted(
            (
                pending
                for pending in self._pending.values()
                if pending.reveal_at_step <= step
            ),
            key=lambda pending: (
                pending.reveal_at_step,
                self._insertion_order[pending.action_id],
            ),
        )


class RewardBuffer(PendingRewardBuffer):
    """Backward-compatible delayed reward buffer.

    Invariant: no reward is delivered to policy.update() before step t+k,
    where k is the sampled delay. Violations make evaluation invalid.
    """

    def __init__(self, rng: np.random.Generator | None = None) -> None:
        if rng is None:
            raise TypeError(
                "RewardBuffer requires an explicit rng=np.random.default_rng(seed). "
                "Passing a fixed seed silently would violate the injected-randomness invariant."
            )
        super().__init__(rng=rng)
