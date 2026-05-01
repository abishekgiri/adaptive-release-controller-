"""Policy adaptation hooks triggered by drift detection; supports reset and windowed retraining."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from drift.detectors import DriftDetector
from policies.base import Policy


@dataclass(frozen=True)
class AdaptConfig:
    """Controls how aggressively the system responds to a detected drift event."""

    strategy: str = "reset"      # 'reset' or 'window'; reset is safer, window is faster
    window_size: int = 200       # number of recent steps to retain on windowed retraining
    cooldown_steps: int = 50     # minimum steps between consecutive adaptation events


class DriftAdaptor:
    """Monitors a policy's reward stream and triggers adaptation on detected drift.

    Wraps a DriftDetector and a Policy. The experiment loop calls observe(cost)
    at every step. When drift is detected, on_drift() is invoked and the policy
    is reset or retrained depending on AdaptConfig.strategy.
    """

    def __init__(
        self,
        policy: Policy,
        detector: DriftDetector,
        config: AdaptConfig = AdaptConfig(),
        on_drift_callback: Optional[Callable[[int], None]] = None,
    ) -> None:
        # TODO: store all fields; initialise cooldown counter
        self._policy = policy
        self._detector = detector
        self._config = config
        self._on_drift_callback = on_drift_callback
        self._steps_since_last_adapt: int = 0
        self._total_drift_events: int = 0

    def observe(self, cost: float, current_step: int) -> bool:
        """Feed cost observation to the detector; trigger adaptation if drift is signalled.

        Args:
            cost: Per-step operational cost from the most recent (action, outcome) pair.
            current_step: Current trajectory step index; used for cooldown enforcement.

        Returns:
            True if adaptation was triggered this step.
        """
        # TODO: call detector.update(cost); check cooldown; call _adapt() if needed
        raise NotImplementedError

    def _adapt(self, current_step: int) -> None:
        """Execute the configured adaptation strategy."""
        # TODO: if strategy == 'reset', call policy.reset() and detector.reset()
        # if strategy == 'window', retain last window_size updates and retrain
        raise NotImplementedError

    @property
    def drift_event_count(self) -> int:
        """Total number of drift events detected since construction."""
        return self._total_drift_events
