"""LinUCB contextual bandit (Li et al. 2010); first real bandit baseline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from data.schemas import Action, Context, Reward
from policies.base import FeatureEncoder, Policy


@dataclass(frozen=True)
class LinUCBConfig:
    """Hyperparameters for LinUCB."""

    alpha: float = 1.0       # exploration width; scales the UCB confidence interval
    lambda_reg: float = 1.0  # ridge regularisation; A initialised to lambda_reg * I_d


class LinUCBPolicy(Policy):
    """Disjoint LinUCB: one independent ridge-regression model per arm.

    Implements Algorithm 1 from Li et al. 2010. Regret bound: O(√(dT log T)).
    All three actions (deploy, canary, block) are treated as separate arms.

    Per-arm model:
        A_a ∈ R^{d×d}  initialised to lambda_reg * I   (grows with each update)
        b_a ∈ R^d      initialised to 0

    At action time:
        θ_a = A_a^{-1} b_a
        UCB(a) = θ_a^T x  +  α * sqrt(x^T A_a^{-1} x)

    At update time (reward r = -cost):
        A_a += x x^T
        b_a += r x

    Rewards are received via the experiment loop after delayed/buffer.py confirms
    maturity; this class never receives immediate (same-step) rewards.
    """

    def __init__(
        self,
        config: LinUCBConfig,
        feature_dim: int,
        rng: np.random.Generator,
        encoder: Optional[FeatureEncoder] = None,
        policy_id: str = "linucb",
    ) -> None:
        """
        Args:
            config:      Hyperparameters (alpha, lambda_reg).
            feature_dim: Dimensionality of the feature vector. Must match encoder output.
                         Use FeatureEncoder.DIM for the default encoder.
            rng:         Seeded RNG (reserved for future tie-breaking or Thompson variant).
            encoder:     Feature encoder; defaults to FeatureEncoder().
            policy_id:   Stable string identifier for logging.
        """
        self._config = config
        self._feature_dim = feature_dim
        self._rng = rng
        self._encoder = encoder or FeatureEncoder()
        self._policy_id = policy_id
        self._A: dict[Action, np.ndarray] = {}
        self._b: dict[Action, np.ndarray] = {}
        self._init_arms()

    # ------------------------------------------------------------------
    # Policy interface
    # ------------------------------------------------------------------

    def select_action(self, context: Context) -> tuple[Action, float]:
        """Return (best_arm, propensity=1.0) via greedy UCB maximisation.

        Ties are broken by the natural iteration order of Action (DEPLOY → CANARY → BLOCK);
        the first arm reaching the maximum score is chosen.
        """
        x = self._encoder.encode(context)
        best_action: Action = Action.DEPLOY
        best_score: float = -np.inf

        for action in Action:
            A_inv_x = np.linalg.solve(self._A[action], x)
            theta = np.linalg.solve(self._A[action], self._b[action])
            exploit = float(theta @ x)
            # clip negative rounding errors in the quadratic form before sqrt
            explore = self._config.alpha * float(np.sqrt(max(0.0, x @ A_inv_x)))
            score = exploit + explore
            if score > best_score:
                best_score = score
                best_action = action

        return best_action, 1.0

    def update(self, context: Context, action: Action, reward: Reward) -> None:
        """Update A_action and b_action given the delayed reward.

        Converts cost to reward: r = -reward.cost (we minimise cost).
        Called by the experiment loop only after the reward has matured.
        """
        x = self._encoder.encode(context)
        r = -reward.cost
        self._A[action] = self._A[action] + np.outer(x, x)
        self._b[action] = self._b[action] + r * x

    def reset(self) -> None:
        """Reinitialise all per-arm matrices; used by drift/adapt.py on drift detection."""
        self._init_arms()

    @property
    def policy_id(self) -> str:
        return self._policy_id

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _init_arms(self) -> None:
        """Set A_a = lambda_reg * I and b_a = 0 for every arm."""
        d = self._feature_dim
        lam = self._config.lambda_reg
        for action in Action:
            self._A[action] = lam * np.eye(d, dtype=np.float64)
            self._b[action] = np.zeros(d, dtype=np.float64)
