"""Typed dataclasses for the unified trajectory format shared across all pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class Action(str, Enum):
    """The three deployment actions available to any policy."""

    DEPLOY = "deploy"
    CANARY = "canary"
    BLOCK = "block"


class Outcome(str, Enum):
    """Observed deployment outcome, available only after delay k steps."""

    SUCCESS = "success"
    FAILURE = "failure"
    CENSORED = "censored"  # reward window closed before outcome resolved


@dataclass(frozen=True)
class Context:
    """Pre-action observable features — contains NO outcome, risk score, or post-deploy signal."""

    # Identity
    commit_sha: str
    project_slug: str
    step: int  # discrete decision-time index within the trajectory

    # Commit metadata
    files_changed: int
    lines_added: int
    lines_deleted: int
    src_churn: int
    is_pr: bool

    # CI signal
    tests_run: int
    tests_added: int
    build_duration_s: float

    # Derived history (computed by loader / extractor, not post-action)
    author_experience: int       # prior commits by same author in project
    recent_failure_rate: float   # rolling 7-day build failure rate
    has_dependency_change: bool
    has_risky_path_change: bool


@dataclass(frozen=True)
class Reward:
    """Delayed reward observed k steps after the action."""

    action_id: str
    outcome: Outcome
    cost: float
    delay_steps: int          # k: number of steps between action and reward arrival
    censored: bool
    observed_at_step: int


@dataclass(frozen=True)
class Trajectory:
    """A time-ordered sequence of (context, action, reward) triples for one project run."""

    trajectory_id: str
    project_slug: str
    policy_id: str
    drift_segment_id: Optional[str]
    steps: tuple  # tuple[TrajectoryStep, ...]  — kept as tuple for hashability


@dataclass(frozen=True)
class TrajectoryStep:
    """One (context, action, reward) triple within a trajectory."""

    context: Context
    action: Action
    propensity: float         # P(action | context) under the logging policy; used for IPS
    reward: Optional[Reward]  # None until the delayed reward arrives
