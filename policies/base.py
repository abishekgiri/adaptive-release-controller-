"""Abstract Policy interface; all policies — baselines and bandits — implement this contract."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from data.schemas import Action, Context, Reward

# Normalisation denominators aligned with the synthetic environment's generation ranges.
# Chosen so that a typical feature value lands in roughly [0, 1].
_NORM_FACTORS: np.ndarray = np.array([
    50.0,    # files_changed      (Poisson ~5 + complexity*20; cap at 50 for normalisation)
    1000.0,  # lines_added        (files*15; large PRs can exceed but are clipped)
    500.0,   # lines_deleted
    1500.0,  # src_churn          (lines_added + lines_deleted)
    1.0,     # is_pr              (binary)
    200.0,   # tests_run          (Poisson 50 + complexity*100)
    10.0,    # tests_added        (Poisson 2 + complexity*5)
    180.0,   # build_duration_s   (30 + load*120; max ~150 s)
    10.0,    # author_experience  (0–10 range from env)
    1.0,     # recent_failure_rate (already [0, 1])
    1.0,     # has_dependency_change (binary)
    1.0,     # has_risky_path_change (binary)
], dtype=np.float64)


class Policy(ABC):
    """A stateful mapping from context to action, updated by delayed rewards.

    Invariant: update() must NEVER be called with a reward at the same step the action
    was taken. All reward delivery goes through delayed/buffer.py.
    """

    @abstractmethod
    def select_action(self, context: Context) -> tuple[Action, float]:
        """Select a deployment action for the given context.

        Args:
            context: Pre-action observable features. Must contain no outcome information.

        Returns:
            (action, propensity) — the chosen action and the probability assigned to it
            under the current policy. Propensity is required for IPS-based evaluation.
        """

    @abstractmethod
    def update(self, context: Context, action: Action, reward: Reward) -> None:
        """Update internal state given a (context, action, delayed_reward) triple.

        This method is called by the experiment loop only after delayed/buffer.py
        confirms the reward has arrived (k steps after the action).

        Args:
            context: The context observed at action time (not at reward time).
            action: The action taken at that step.
            reward: The delayed reward that just matured.
        """

    @abstractmethod
    def reset(self) -> None:
        """Reset all learned state; called by drift/adapt.py on detected drift."""

    @property
    @abstractmethod
    def policy_id(self) -> str:
        """Stable identifier used for logging and reproducibility."""


class FeatureEncoder:
    """Converts a Context dataclass into a fixed-length numpy feature vector.

    Feature order (12 context fields, normalised, then 1 bias term):
        0  files_changed          / 50
        1  lines_added            / 1000
        2  lines_deleted          / 500
        3  src_churn              / 1500
        4  is_pr                  (0/1)
        5  tests_run              / 200
        6  tests_added            / 10
        7  build_duration_s       / 180
        8  author_experience      / 10
        9  recent_failure_rate    (already [0,1])
        10 has_dependency_change  (0/1)
        11 has_risky_path_change  (0/1)
        12 bias                   = 1.0
    """

    DIM: int = 13  # 12 normalised context fields + 1 bias term

    def encode(self, context: Context) -> np.ndarray:
        """Return a 1-D float64 array of shape (DIM,) from a Context."""
        raw = np.array([
            context.files_changed,
            context.lines_added,
            context.lines_deleted,
            context.src_churn,
            float(context.is_pr),
            context.tests_run,
            context.tests_added,
            context.build_duration_s,
            context.author_experience,
            context.recent_failure_rate,
            float(context.has_dependency_change),
            float(context.has_risky_path_change),
        ], dtype=np.float64)
        normalised = raw / _NORM_FACTORS
        return np.append(normalised, 1.0)  # bias term last
