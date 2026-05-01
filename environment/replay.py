"""Offline replay environment over real CI/CD logs using logged propensities for IPS/DR evaluation."""

from __future__ import annotations

from typing import Iterator, Optional

from data.schemas import Action, Context, Reward, Trajectory, TrajectoryStep
from environment.base import DeploymentEnvironment


class ReplayEnvironment(DeploymentEnvironment):
    """Steps through a fixed logged trajectory; propensities enable unbiased off-policy evaluation.

    The logging policy propensity P(a_t | x_t) must be supplied per step. If propensities
    are unknown (e.g. TravisTorrent), a uniform assumption is used and flagged as a threat
    to validity in docs/evaluation-protocol.md.
    """

    def __init__(self, trajectory: Trajectory, default_propensity: float = 1.0) -> None:
        # TODO: store trajectory, set up step iterator
        self._trajectory = trajectory
        self._default_propensity = default_propensity
        self._step_index = 0
        self._iterator: Optional[Iterator[TrajectoryStep]] = None

    def observe(self) -> Context:
        # TODO: return context at current step without advancing
        raise NotImplementedError

    def step(self, action: Action) -> Optional[Reward]:
        # TODO: log chosen action against logged action; advance step; return logged reward
        # Note: reward is the logged reward regardless of the policy's chosen action.
        # Counterfactual correction (IPS/DR) happens in evaluation/replay_eval.py, not here.
        raise NotImplementedError

    def advance_time(self) -> list[Reward]:
        # TODO: in replay mode, time is discrete steps through the log; return matured rewards
        raise NotImplementedError

    def reset(self) -> Context:
        # TODO: rewind trajectory iterator to start
        raise NotImplementedError

    @property
    def current_step(self) -> int:
        return self._step_index

    @property
    def done(self) -> bool:
        return self._step_index >= len(self._trajectory.steps)

    def logged_propensity(self, step: int, action: Action) -> float:
        """Return the logging policy's propensity for the given action at the given step."""
        # TODO: look up propensity from trajectory step; fall back to default_propensity
        raise NotImplementedError
