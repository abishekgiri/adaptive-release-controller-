"""Offline-trained classifier policy (logistic regression / gradient boosting); baseline 3."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from data.schemas import Action, Context, Reward
from policies.base import FeatureEncoder, Policy


class OfflineClassifierPolicy(Policy):
    """Policy backed by a classifier trained once on historical data; does not update online.

    Represents the 'train offline, deploy statically' paradigm. The bandit must beat this
    on cumulative cost to justify the online-learning overhead.
    propensity is the classifier's predicted class probability for the chosen action.
    """

    def __init__(
        self,
        model: Any,  # sklearn-compatible: has predict_proba(X)
        encoder: Optional[FeatureEncoder] = None,
        policy_id: str = "offline_classifier",
    ) -> None:
        # TODO: store model and encoder; model is fitted externally before injection
        self._model = model
        self._encoder = encoder or FeatureEncoder()
        self._policy_id = policy_id

    def select_action(self, context: Context) -> tuple[Action, float]:
        # TODO: encode context → feature vector; call model.predict_proba; argmax → action
        # Map binary classifier output to {DEPLOY, BLOCK}; CANARY not used by this baseline
        raise NotImplementedError

    def update(self, context: Context, action: Action, reward: Reward) -> None:
        # Offline model; no online update.
        pass

    def reset(self) -> None:
        # Model weights are frozen; nothing to reset.
        pass

    @property
    def policy_id(self) -> str:
        return self._policy_id

    @classmethod
    def fit(
        cls,
        contexts: list[Context],
        actions: list[Action],
        outcomes: list[bool],
        model_type: str = "logistic",
        policy_id: str = "offline_classifier",
    ) -> "OfflineClassifierPolicy":
        """Train and return a fitted policy from historical (context, outcome) pairs.

        Args:
            contexts: Historical pre-action contexts.
            actions: Actions taken (used only for propensity estimation).
            outcomes: True if deployment succeeded; used as the training label.
            model_type: 'logistic' or 'gbm'.
            policy_id: Identifier for logging.
        """
        # TODO: build sklearn pipeline; fit on feature matrix; return wrapped policy
        raise NotImplementedError
