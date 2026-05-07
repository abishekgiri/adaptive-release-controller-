"""Concept drift detectors: ADWIN, Page-Hinkley, and DDM; wrappers or lightweight implementations."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections import deque
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
    """Adaptive Windowing (ADWIN) drift detector (Bifet & Gavaldà 2007).

    Maintains a compressed sliding window and detects when the mean of two
    adjacent sub-windows W0 (older) and W1 (newer) differs by more than the
    Hoeffding bound ε_cut = sqrt((1/(2m)) * ln(4n/δ)), where m = n0*n1/(n0+n1)
    and n = n0+n1.

    Memory: O(M log W) where W is the window size and M = _MAX_BUCKETS.
    On detection the older sub-window is dropped; the detector never resets
    the full window — it only shrinks it to the most recent coherent portion.

    Usage::
        detector = ADWINDetector()
        for cost in cost_stream:
            if detector.update(cost):
                print("drift detected")
    """

    _MAX_BUCKETS: int = 5  # M in the paper; max buckets kept per level

    def __init__(self, config: ADWINConfig = ADWINConfig()) -> None:
        self._config = config
        # _levels[i]: deque of (bsum, bcount) at level i, newest appended right
        self._levels: list[deque[tuple[float, int]]] = []
        self._total: float = 0.0
        self._n: int = 0
        self._detected: bool = False

    def update(self, value: float) -> bool:
        """Feed one observation; returns True on the step drift is first detected."""
        self._detected = False
        if not self._levels:
            self._levels.append(deque())
        self._levels[0].append((value, 1))
        self._total += value
        self._n += 1
        self._compress()
        self._detected = self._check_drift()
        return self._detected

    def reset(self) -> None:
        self._levels = []
        self._total = 0.0
        self._n = 0
        self._detected = False

    @property
    def drift_detected(self) -> bool:
        return self._detected

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _compress(self) -> None:
        """Merge the two oldest buckets at level i into one at level i+1 whenever
        level i holds more than _MAX_BUCKETS buckets (propagates up the chain)."""
        for i in range(len(self._levels)):
            while len(self._levels[i]) > self._MAX_BUCKETS:
                s1, c1 = self._levels[i].popleft()   # oldest
                s2, c2 = self._levels[i].popleft()   # second oldest
                if i + 1 >= len(self._levels):
                    self._levels.append(deque())
                self._levels[i + 1].append((s1 + s2, c1 + c2))

    def _check_drift(self) -> bool:
        """Scan from newest bucket to oldest; signal drift and shrink window if found."""
        if self._n < 2:
            return False
        n = self._n
        n1: int = 0
        s1: float = 0.0
        # Iterate newest → oldest: levels 0, 1, 2, … within each level right → left
        for level_idx, level in enumerate(self._levels):
            for bucket_pos, (bsum, bcount) in enumerate(reversed(level)):
                n1 += bcount
                s1 += bsum
                n0 = n - n1
                if n0 <= 0:
                    continue
                s0 = self._total - s1
                mu0 = s0 / n0
                mu1 = s1 / n1
                # Harmonic-mean form of the Hoeffding bound (Theorem 2)
                m = n0 * n1 / (n0 + n1)
                eps = math.sqrt((1.0 / (2.0 * m)) * math.log(4.0 * n / self._config.delta))
                if abs(mu0 - mu1) >= eps:
                    # Drop everything older than the current cut point
                    n_older_in_level = len(level) - 1 - bucket_pos
                    self._drop_older(level_idx, n_older_in_level)
                    return True
        return False

    def _drop_older(self, cut_level: int, n_to_drop_from_cut_level: int) -> None:
        """Remove the older sub-window from the internal bucket store."""
        for i, level in enumerate(self._levels):
            if i < cut_level:
                pass  # all newer — keep
            elif i == cut_level:
                for _ in range(n_to_drop_from_cut_level):
                    bs, bc = level.popleft()
                    self._total -= bs
                    self._n -= bc
            else:
                for bs, bc in level:
                    self._total -= bs
                    self._n -= bc
                self._levels[i] = deque()
        # Trim empty trailing levels
        while self._levels and not self._levels[-1]:
            self._levels.pop()


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
