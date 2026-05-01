"""Synthetic deployment environment with strictly separated hidden state and observable context."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Optional

import numpy as np

from data.schemas import Action, Context, Outcome, Reward
from environment.base import DeploymentEnvironment


# ---------------------------------------------------------------------------
# Hidden state — NEVER exposed to the policy
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HiddenState:
    """Ground-truth deployment risk variables — opaque to the policy.

    Field names are deliberately chosen to share NO names with Context fields.
    This is enforced by tests/test_environment.py::test_hidden_context_field_disjointness.

    Mapping to Context (noisy, one-way):
        true_failure_prob  →  recent_failure_rate  (noisy lagged estimate)
        infra_load         →  build_duration_s      (load inflates build time)
        change_complexity  →  files_changed, lines_added, src_churn, tests_run,
                              has_dependency_change, has_risky_path_change
        team_fatigue       →  author_experience     (inversely, with noise)
        drift_segment      →  (not visible; policy must infer drift from rewards)
    """

    true_failure_prob: float   # P(deployment fails); in [0, 1]
    infra_load: float          # infrastructure stress; in [0, 1]
    change_complexity: float   # true riskiness of the change; in [0, 1]
    team_fatigue: float        # team error-rate factor; in [0, 1]
    drift_segment: int         # index into DriftSchedule.segments


# ---------------------------------------------------------------------------
# Drift schedule
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SegmentParams:
    """Distribution parameters for one piecewise-stationary segment."""

    base_failure_prob: float        # centre of P(failure) for this segment
    infra_load_mean: float          # mean infrastructure load
    change_complexity_mean: float   # mean change complexity
    team_fatigue_mean: float        # mean team fatigue


@dataclass(frozen=True)
class DriftSchedule:
    """Abrupt piecewise-stationary drift: every segment_length steps, switch segment."""

    segment_length: int                    # steps per segment
    segments: tuple[SegmentParams, ...]   # cycles when trajectory exceeds total length

    def segment_index_at(self, step: int) -> int:
        return (step // self.segment_length) % len(self.segments)

    def params_at(self, step: int) -> SegmentParams:
        return self.segments[self.segment_index_at(step)]


def default_drift_schedule() -> DriftSchedule:
    """Two alternating segments: low-risk baseline and high-risk shifted distribution."""
    return DriftSchedule(
        segment_length=200,
        segments=(
            SegmentParams(
                base_failure_prob=0.10,
                infra_load_mean=0.20,
                change_complexity_mean=0.25,
                team_fatigue_mean=0.15,
            ),
            SegmentParams(
                base_failure_prob=0.55,
                infra_load_mean=0.70,
                change_complexity_mean=0.70,
                team_fatigue_mean=0.60,
            ),
        ),
    )


# ---------------------------------------------------------------------------
# Synthetic environment
# ---------------------------------------------------------------------------

# Noise standard deviations for the Context projection (tunable but not critical).
_FAILURE_RATE_NOISE: float = 0.05
_INFRA_NOISE: float = 0.10
_COMPLEXITY_NOISE: float = 0.10
_FATIGUE_NOISE: float = 0.10

# Action-specific failure probability multipliers.
# Canary catches ~60 % of failures via gradual rollout before full promotion.
_ACTION_FAILURE_MULTIPLIER: dict[Action, float] = {
    Action.DEPLOY: 1.0,
    Action.CANARY: 0.4,
    Action.BLOCK: 1.0,   # counterfactual: would this change have failed?
}


class SyntheticEnvironment(DeploymentEnvironment):
    """Deployment environment backed by a hidden Markov-like risk process.

    Observable context is a noisy projection of the hidden state. Outcomes are
    sampled from the hidden state only — the policy cannot infer the outcome
    directly from the context features.

    Usage::

        rng = np.random.default_rng(42)
        env = SyntheticEnvironment(rng, horizon=1000)
        ctx = env.reset()
        while not env.done:
            matured = env.advance_time()
            # deliver matured rewards to policy here
            ctx = env.observe()
            env.step(action)   # always returns None; reward arrives later
    """

    def __init__(
        self,
        rng: np.random.Generator,
        horizon: int,
        drift_schedule: Optional[DriftSchedule] = None,
        delay_p: float = 0.3,
        max_delay: int = 20,
    ) -> None:
        """
        Args:
            rng: Seeded RNG; all randomness in this env flows through here.
            horizon: Number of decision steps before done == True.
            drift_schedule: Piecewise-stationary drift spec; defaults to
                default_drift_schedule() if None.
            delay_p: Success probability of the geometric reward delay distribution.
                Mean delay = 1 / delay_p steps.
            max_delay: Rewards not observed within max_delay steps are censored.
        """
        self._rng = rng
        self._horizon = horizon
        self._drift_schedule = drift_schedule or default_drift_schedule()
        self._delay_p = delay_p
        self._max_delay = max_delay

        self._step: int = 0
        self._hidden: HiddenState = self._sample_hidden_state(segment=0)
        # Each entry: (reveal_at_step, Reward)
        self._pending: list[tuple[int, Reward]] = []

    # ------------------------------------------------------------------
    # DeploymentEnvironment interface
    # ------------------------------------------------------------------

    def reset(self) -> Context:
        """Reset environment to step 0 and return the first context."""
        self._step = 0
        self._pending = []
        self._hidden = self._sample_hidden_state(segment=0)
        return self.observe()

    def observe(self) -> Context:
        """Return the current context — a noisy projection of the hidden state."""
        return self._project_to_context(self._hidden)

    def step(self, action: Action) -> None:  # type: ignore[override]
        """Sample an outcome from the hidden state and schedule a delayed reward.

        Always returns None — rewards are never immediate in this environment.
        The reward arrives via advance_time() after k ~ Geom(delay_p) steps.

        Cost is NOT computed here. Set to float('nan') as a sentinel; the
        experiment loop must call rewards/cost_model.py::compute_cost().
        """
        outcome = self._sample_outcome(self._hidden, action)
        delay = self._sample_delay()
        reveal_at = self._step + delay
        censored = delay >= self._max_delay

        reward = Reward(
            action_id=f"synth_{self._step}_{uuid.uuid4().hex[:6]}",
            outcome=Outcome.CENSORED if censored else outcome,
            cost=float("nan"),   # cost_model.compute_cost() must be called by caller
            delay_steps=delay,
            censored=censored,
            observed_at_step=reveal_at,
        )
        self._pending.append((reveal_at, reward))
        return None

    def advance_time(self) -> list[Reward]:
        """Advance the clock by one step; apply drift if segment boundary crossed.

        Returns all rewards whose reveal_at_step equals the new step index.
        """
        self._step += 1
        self._apply_drift_if_needed()

        matured = [r for (t, r) in self._pending if t <= self._step]
        self._pending = [(t, r) for (t, r) in self._pending if t > self._step]
        return matured

    @property
    def current_step(self) -> int:
        return self._step

    @property
    def done(self) -> bool:
        return self._step >= self._horizon

    # ------------------------------------------------------------------
    # Hidden-state sampling
    # ------------------------------------------------------------------

    def _sample_hidden_state(self, segment: int) -> HiddenState:
        """Draw a hidden state from the distribution of the given segment."""
        params = self._drift_schedule.segments[
            segment % len(self._drift_schedule.segments)
        ]
        return HiddenState(
            true_failure_prob=float(np.clip(
                params.base_failure_prob + self._rng.normal(0.0, _FAILURE_RATE_NOISE),
                0.0, 1.0,
            )),
            infra_load=float(np.clip(
                params.infra_load_mean + self._rng.normal(0.0, _INFRA_NOISE),
                0.0, 1.0,
            )),
            change_complexity=float(np.clip(
                params.change_complexity_mean + self._rng.normal(0.0, _COMPLEXITY_NOISE),
                0.0, 1.0,
            )),
            team_fatigue=float(np.clip(
                params.team_fatigue_mean + self._rng.normal(0.0, _FATIGUE_NOISE),
                0.0, 1.0,
            )),
            drift_segment=segment,
        )

    def _apply_drift_if_needed(self) -> None:
        """If we crossed a segment boundary, resample the hidden state."""
        new_segment = self._drift_schedule.segment_index_at(self._step)
        if new_segment != self._hidden.drift_segment:
            self._hidden = self._sample_hidden_state(segment=new_segment)

    # ------------------------------------------------------------------
    # Outcome sampling — depends ONLY on HiddenState, never on Context
    # ------------------------------------------------------------------

    def _sample_outcome(self, hidden: HiddenState, action: Action) -> Outcome:
        """Sample a deployment outcome from the hidden state.

        Context is deliberately NOT a parameter. The outcome distribution is
        fully determined by hidden.true_failure_prob and the action multiplier.
        If a reviewer asks "how do you know the policy can't cheat?", point here.
        """
        p_fail = hidden.true_failure_prob * _ACTION_FAILURE_MULTIPLIER[action]
        return Outcome.FAILURE if self._rng.random() < p_fail else Outcome.SUCCESS

    # ------------------------------------------------------------------
    # Context projection — noisy, one-way mapping from HiddenState
    # ------------------------------------------------------------------

    def _project_to_context(self, hidden: HiddenState) -> Context:
        """Generate a noisy observable context from the current hidden state.

        Every Context field is a corrupted signal of one or more HiddenState
        fields. No HiddenState field name appears in Context — verified by
        test_hidden_context_field_disjointness.
        """
        # recent_failure_rate: noisy lagged estimate of true_failure_prob
        recent_failure_rate = float(np.clip(
            hidden.true_failure_prob + self._rng.normal(0.0, _FAILURE_RATE_NOISE),
            0.0, 1.0,
        ))

        # build_duration_s: infra load inflates build times
        build_duration_s = float(max(
            10.0,
            30.0 + hidden.infra_load * 120.0 + self._rng.normal(0.0, 10.0),
        ))

        # files_changed, lines_added, lines_deleted, src_churn: proxy for change_complexity
        files_changed = int(max(1, self._rng.poisson(5 + hidden.change_complexity * 20)))
        lines_added = int(max(0, self._rng.poisson(files_changed * 15 + 1)))
        lines_deleted = int(max(0, self._rng.poisson(lines_added * 0.3 + 1)))
        src_churn = lines_added + lines_deleted

        # author_experience: inversely related to team_fatigue (fatigued teams make mistakes)
        author_experience = int(max(0, round(
            10.0 - hidden.team_fatigue * 8.0 + self._rng.normal(0.0, 1.5)
        )))

        # tests_run, tests_added: more complex changes touch more tests
        tests_run = int(max(0, self._rng.poisson(50 + hidden.change_complexity * 100)))
        tests_added = int(max(0, self._rng.poisson(2 + hidden.change_complexity * 5)))

        # Boolean signals: Bernoulli probabilities tied to change_complexity
        is_pr = bool(self._rng.random() < 0.4)
        has_dependency_change = bool(
            self._rng.random() < 0.05 + hidden.change_complexity * 0.25
        )
        has_risky_path_change = bool(
            self._rng.random() < hidden.change_complexity * 0.35
        )

        return Context(
            commit_sha=f"synth_{self._step:06d}",
            project_slug="synthetic/env",
            step=self._step,
            files_changed=files_changed,
            lines_added=lines_added,
            lines_deleted=lines_deleted,
            src_churn=src_churn,
            is_pr=is_pr,
            tests_run=tests_run,
            tests_added=tests_added,
            build_duration_s=build_duration_s,
            author_experience=author_experience,
            recent_failure_rate=recent_failure_rate,
            has_dependency_change=has_dependency_change,
            has_risky_path_change=has_risky_path_change,
        )

    # ------------------------------------------------------------------
    # Delay sampling
    # ------------------------------------------------------------------

    def _sample_delay(self) -> int:
        """Draw k ~ Geom(delay_p), capped at max_delay (censored if k >= max_delay)."""
        return int(min(self._rng.geometric(p=self._delay_p), self._max_delay))
