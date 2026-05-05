"""Concept drift detectors: ADWIN, Page-Hinkley, and DDM; wrappers or lightweight implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


class DriftDetector(ABC):
    """Abstract drift detector; consumes a scalar stream and signals when drift is detected."""

    @abstractmethod
    def update(self, value: float) -> bool:
        """Feed one observation to the detector.

        Args:
            value: Scalar observation (e.g. per-step cost or reward).

        Returns:
            True if drift is detected at this step; False otherwise.
        """
        # TODO: implement in subclasses

    @abstractmethod
    def reset(self) -> None:
        """Reset detector state; called after a confirmed drift event."""
        # TODO: implement in subclasses

    @property
    @abstractmethod
    def drift_detected(self) -> bool:
        """True if the most recent update() call signalled a drift event."""
        # TODO: implement in subclasses


@dataclass(frozen=True)
class ADWINConfig:
    """Hyperparameters for ADWIN (Bifet & Gavaldà 2007)."""

    delta: float = 0.002  # confidence parameter; smaller → slower but more accurate detection


class ADWINDetector(DriftDetector):
    """Adaptive Windowing (ADWIN) drift detector.

    Maintains a window of recent observations and detects when the mean in
    two sub-windows differs significantly. O(log W) memory where W is window size.
    Recommended wrapper: river.drift.ADWIN (if river is available) or a minimal
    reimplementation here.
    """

    def __init__(self, config: ADWINConfig = ADWINConfig()) -> None:
        # TODO: initialise ADWIN window structures; or import from river
        self._config = config
        self._detected = False

    def update(self, value: float) -> bool:
        # TODO: add value to window; check for distributional change; set _detected
        raise NotImplementedError

    def reset(self) -> None:
        # TODO: clear window state
        self._detected = False

    @property
    def drift_detected(self) -> bool:
        return self._detected


@dataclass(frozen=True)
class PageHinkleyConfig:
    """Hyperparameters for the Page-Hinkley test."""

    delta: float = 0.005    # minimum amplitude of change to detect
    lambda_: float = 50.0   # detection threshold; larger → fewer false positives
    alpha: float = 0.9999   # forgetting factor for the running mean


class PageHinkleyDetector(DriftDetector):
    """Page-Hinkley test for detecting upward shifts in a scalar stream.

    Algorithm (Mouss et al. 2004):
        Running mean m_t = alpha * m_{t-1} + (1-alpha) * x_t
        Cumulative sum S_t = S_{t-1} + (x_t - m_t - delta)
        M_t = min(S_0, ..., S_t)
        Drift if S_t - M_t > lambda_
    """

    def __init__(self, config: PageHinkleyConfig = PageHinkleyConfig()) -> None:
        self._config = config
        self._detected = False
        self._cum_sum: float = 0.0
        self._min_sum: float = 0.0
        self._mean: float = 0.0
        self._n: int = 0

    def update(self, value: float) -> bool:
        """Feed one observation; returns True on the step drift is first detected."""
        self._n += 1
        alpha = self._config.alpha
        self._mean = alpha * self._mean + (1.0 - alpha) * value
        self._cum_sum += value - self._mean - self._config.delta
        if self._cum_sum < self._min_sum:
            self._min_sum = self._cum_sum
        self._detected = (self._cum_sum - self._min_sum) > self._config.lambda_
        return self._detected

    def reset(self) -> None:
        self._detected = False
        self._cum_sum = 0.0
        self._min_sum = 0.0
        self._mean = 0.0
        self._n = 0

    @property
    def drift_detected(self) -> bool:
        return self._detected


def make_detector(name: str) -> DriftDetector:
    """Factory function; name ∈ {'adwin', 'page_hinkley'}."""
    # TODO: add DDM when implemented
    _registry = {
        "adwin": lambda: ADWINDetector(),
        "page_hinkley": lambda: PageHinkleyDetector(),
    }
    if name not in _registry:
        raise ValueError(f"Unknown detector '{name}'. Available: {list(_registry)}")
    return _registry[name]()
