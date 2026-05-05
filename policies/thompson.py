"""Bayesian linear Thompson Sampling; bandit baseline 2.

Model: conjugate Gaussian linear regression per arm.
  Prior:      θ_a ~ N(0, v0 · I)          (prior_variance = v0)
  Likelihood: r | θ_a, x ~ N(θ_a^T x, σ²) (noise_variance = σ²)
  Posterior:  θ_a | data ~ N(μ_a, Λ_a^{-1})
    Λ_a = (1/v0)·I  +  (1/σ²)·Σ x_i x_i^T   (precision matrix)
    b_a = (1/σ²)·Σ r_i x_i
    μ_a = Λ_a^{-1} b_a

Action selection: sample θ_a from the posterior for each arm,
then pick argmax_a θ_a^T x_t (Thompson sampling).

Update: incremental Bayesian update via rank-1 precision update.
Exploration is implicit — no α parameter needed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from data.schemas import Action, Context, Reward
from policies.base import FeatureEncoder, Policy


@dataclass(frozen=True)
class ThompsonConfig:
    """Hyperparameters for Gaussian linear Thompson Sampling."""

    prior_variance: float = 1.0    # v0: variance of the Gaussian prior on weights
    noise_variance: float = 0.1    # σ²: assumed observation noise variance


class ThompsonSamplingPolicy(Policy):
    """Thompson Sampling with a Bayesian linear Gaussian reward model.

    Maintains a per-arm Gaussian posterior over weight vectors. Action
    selection samples one weight vector per arm from its posterior and
    picks the arm with the highest inner product with the current context.

    The posterior update uses the conjugate Gaussian-linear form:
        Λ_a ← Λ_a + (1/σ²) x x^T
        b_a ← b_a + (1/σ²) r x
        μ_a  = Λ_a^{-1} b_a
    which is structurally identical to the LinUCB sufficient statistics
    but drives exploration through posterior sampling rather than UCB.

    Rewards are gated by delayed/buffer.py; update() must only be called
    after the reward has matured (same invariant as all Policy subclasses).
    """

    def __init__(
        self,
        config: ThompsonConfig,
        feature_dim: int,
        rng: np.random.Generator,
        encoder: Optional[FeatureEncoder] = None,
        policy_id: str = "thompson",
    ) -> None:
        self._config = config
        self._d = feature_dim
        self._rng = rng
        self._encoder = encoder or FeatureEncoder()
        self._policy_id = policy_id
        self._Lambda: dict[Action, np.ndarray] = {}
        self._b: dict[Action, np.ndarray] = {}
        self._init_arms()

    # ------------------------------------------------------------------
    # Policy interface
    # ------------------------------------------------------------------

    def select_action(self, context: Context) -> tuple[Action, float]:
        """Sample one weight vector per arm from the posterior; pick greedy arm.

        Sampling: θ_a ~ N(μ_a, Λ_a^{-1})
            = μ_a + L_a^{-T} z,  z ~ N(0, I),  L_a L_a^T = Λ_a (Cholesky)

        Propensity returned is 1.0 — computing the true sampling probability
        requires integrating over the posterior, which is intractable here.
        This is consistent with the IPS limitation already documented for this dataset.
        """
        x = self._encoder.encode(context)
        best_action: Action = Action.DEPLOY
        best_val: float = -np.inf

        for action in Action:
            mu = np.linalg.solve(self._Lambda[action], self._b[action])
            L = np.linalg.cholesky(self._Lambda[action])
            z = self._rng.standard_normal(self._d)
            # θ = μ + Λ^{-1/2} z  via back-solve  L^T θ_delta = z
            theta = mu + np.linalg.solve(L.T, z)
            val = float(theta @ x)
            if val > best_val:
                best_val = val
                best_action = action

        return best_action, 1.0

    def update(self, context: Context, action: Action, reward: Reward) -> None:
        """Bayesian posterior update given a matured delayed reward.

        Uses r = −cost (reward = negative operational cost) so the policy
        learns to prefer arms with lower expected cost.
        """
        if reward.censored or not math.isfinite(reward.cost):
            return
        x = self._encoder.encode(context)
        r = -reward.cost
        inv_sigma2 = 1.0 / self._config.noise_variance
        self._Lambda[action] = self._Lambda[action] + inv_sigma2 * np.outer(x, x)
        self._b[action] = self._b[action] + inv_sigma2 * r * x

    def reset(self) -> None:
        """Reinitialise posteriors to prior; called between project trajectories."""
        self._init_arms()

    @property
    def policy_id(self) -> str:
        return self._policy_id

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _init_arms(self) -> None:
        """Set Λ_a = (1/v0)·I and b_a = 0 for every arm."""
        d = self._d
        inv_v0 = 1.0 / self._config.prior_variance
        for action in Action:
            self._Lambda[action] = inv_v0 * np.eye(d, dtype=np.float64)
            self._b[action] = np.zeros(d, dtype=np.float64)
