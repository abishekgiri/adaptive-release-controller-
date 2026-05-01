"""Synthetic deployment environment with strictly separated hidden state and observable context."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from data.schemas import Action, Context, Outcome, Reward
from environment.base import DeploymentEnvironment


@dataclass(frozen=True)
class HiddenState:
    """Variables that determine the deployment outcome — NEVER exposed to the policy.

    No field in HiddenState may appear in Context. This separation is enforced by
    tests/test_environment.py which asserts field-level disjointness.
    """

    # TODO: define hidden state fields (e.g. true defect probability, infrastructure load)
    # These must share NO names or values with Context fields.
    true_failure_prob: float
    drift_segment: int


@dataclass(frozen=True)
class DriftSchedule:
    """Specifies when and how the hidden-state distribution shifts."""

    # TODO: define drift schedule fields
    # abrupt_steps: tuple[int, ...] — steps at which abrupt shifts occur
    # segment_params: tuple[dict, ...] — per-segment HiddenState distribution params
    pass


class SyntheticEnvironment(DeploymentEnvironment):
    """Synthetic environment; context contains only noisy correlates of the hidden state.

    Invariant: set(Context.__dataclass_fields__) ∩ set(HiddenState.__dataclass_fields__) == ∅
    This is verified by test_environment.py::test_hidden_context_field_disjointness.
    """

    def __init__(
        self,
        rng: np.random.Generator,
        horizon: int,
        drift_schedule: Optional[DriftSchedule] = None,
    ) -> None:
        # TODO: initialise hidden state, context generator, delay model
        self._rng = rng
        self._horizon = horizon
        self._drift_schedule = drift_schedule
        self._step = 0
        self._hidden: Optional[HiddenState] = None
        self._pending_rewards: list[tuple[int, Reward]] = []  # (reveal_at_step, reward)

    def observe(self) -> Context:
        # TODO: generate context from hidden state with noise; share NO fields
        raise NotImplementedError

    def step(self, action: Action) -> Optional[Reward]:
        # TODO: sample outcome from hidden state (not context); schedule delayed reveal
        raise NotImplementedError

    def advance_time(self) -> list[Reward]:
        # TODO: increment step; flush matured pending rewards
        raise NotImplementedError

    def reset(self) -> Context:
        # TODO: reset step counter, sample initial hidden state, return first context
        raise NotImplementedError

    @property
    def current_step(self) -> int:
        return self._step

    @property
    def done(self) -> bool:
        return self._step >= self._horizon

    def _sample_hidden_state(self) -> HiddenState:
        # TODO: draw hidden state params from current drift segment distribution
        raise NotImplementedError

    def _sample_outcome(self, hidden: HiddenState, action: Action) -> Outcome:
        # TODO: draw outcome from hidden state; cost depends on (action, outcome) pair
        raise NotImplementedError
