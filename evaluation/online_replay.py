"""Online replay evaluation for learning policies.

WARNING — SIMULATION, NOT CAUSAL INFERENCE
==========================================
Each step computes cost as ``compute_cost(policy_action, logged_outcome)`` using
the logged CI outcome as a counterfactual proxy for the policy's chosen action.
This assumes an action-independent CI proxy label, observed only after completion.
Decisions are simulated at CI start; CI failure is not a deployment incident.
The mapping supplies counterfactual costs unavailable in a real deployment log.

Do NOT report these numbers as unbiased estimates of real-world cost.
Use them to:
  - Verify that policies actually learn (weights change, actions diverge).
  - Debug the delayed-update pipeline.
  - Compare learning curves across policies on the same trajectory.

IPS additionally requires actual logging propensities and action support; these CI files lack both.
"""

from __future__ import annotations

import math
import bisect
import hashlib
from dataclasses import asdict
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from data.loaders import TravisTorrentRecord
from data.schemas import Action, Outcome, Reward
from delayed.buffer import PendingRewardBuffer
from policies.base import Policy
from rewards.cost_model import CostConfig, compute_cost


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class OnlineStepRecord:
    """Per-step snapshot from one online replay pass."""

    step: int
    policy_action: Action
    logged_outcome: Outcome
    effective_outcome: Outcome   # outcome used for cost lookup (may differ for BLOCK)
    cost: float                  # NaN when censored
    delay_steps: int
    updates_applied: int         # rewards matured and applied to policy at this step
    pending_count_before: int    # pending rewards in buffer before advancing
    recent_failure_rate: float = 0.0
    reveal_at_step: int = 0


@dataclass
class OnlineTrajectoryResult:
    """Aggregated result for one policy over one project trajectory.

    NOTE: costs here are simulation artefacts, not causal estimates.
    """

    policy_id: str
    trajectory_id: str
    project_slug: str
    total_steps: int
    total_updates: int           # delayed reward applications (policy.update() calls)
    total_censored_skipped: int  # steps where cost was NaN; update skipped
    cumulative_cost: float       # sum of finite per-step costs
    action_counts: dict[str, int]
    step_records: list[OnlineStepRecord] = field(default_factory=list)

    @property
    def mean_cost(self) -> float:
        finite = self.total_steps - self.total_censored_skipped
        return self.cumulative_cost / finite if finite > 0 else 0.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _effective_outcome(policy_action: Action, logged_outcome: Outcome) -> Outcome:
    """Keep the same observed/censored CI label for every simulated action."""
    # Use the same evaluation cohort for every policy. Unknown CI outcomes
    # cannot be charged to BLOCK while silently dropping them for other arms.
    return logged_outcome


def _delay_from_record(record: TravisTorrentRecord, delay_step_seconds: int) -> int:
    """Convert build duration to a discrete delay step count (minimum 1)."""
    if delay_step_seconds <= 0:
        raise ValueError("delay_step_seconds must be positive")
    duration = ((record.finished_at - record.started_at).total_seconds()
                if record.finished_at is not None and record.started_at is not None
                else record.context.build_duration_s)
    if duration <= 0:
        return 1
    return max(1, math.ceil(duration / delay_step_seconds))


def reveal_steps(records, timing_mode, delay_step_seconds):
    """Translate actual finish times or an explicitly artificial delay to decisions."""
    if timing_mode == "duration_steps":
        return [i + _delay_from_record(r, delay_step_seconds) for i, r in enumerate(records)]
    if timing_mode != "event_time":
        raise ValueError("timing_mode must be event_time or duration_steps")
    starts = [r.started_at for r in records]
    if any(t is None for t in starts) or starts != sorted(starts):
        raise ValueError("event_time requires ordered, observed start timestamps")
    if any(r.finished_at is None and r.outcome != Outcome.CENSORED for r in records):
        raise ValueError("event_time requires finish times for observed outcomes")
    return [bisect.bisect_left(starts, r.finished_at, lo=i + 1)
            if r.finished_at is not None else len(records)
            for i, r in enumerate(records)]


# ---------------------------------------------------------------------------
# Core replay loop
# ---------------------------------------------------------------------------

def run_online_trajectory(
    policy: Policy,
    records: Sequence[TravisTorrentRecord],
    *,
    cost_config: CostConfig,
    rng: np.random.Generator,
    delay_step_seconds: int = 60,
    trajectory_id: str = "",
    flush_at_end: bool = True,
    timing_mode: str = "duration_steps",
) -> OnlineTrajectoryResult:
    """Run one online learning pass over a project trajectory.

    Records must already be in chronological order; the caller is responsible
    for ordering.  The policy is updated in-place; reset it before calling
    if you need a fresh model.

    Loop at step t:
      1. Advance the buffer to t — matured rewards call ``policy.update()``.
      2. Policy selects action for the current context (no outcome peeking).
      3. Cost = compute_cost(policy_action, effective_outcome, cost_config).
      4. A pending reward is queued for delivery at step t + delay.

    When ``flush_at_end`` is True, all remaining pending rewards are applied
    after the last record.  This lets callers compare final model states
    without an arbitrary trailing buffer.

    Args:
        policy:              Policy to evaluate (updated in-place).
        records:             Chronologically ordered TravisTorrentRecord list.
        cost_config:         Operational cost matrix.
        rng:                 Seeded RNG for the internal delay buffer.
        delay_step_seconds:  Denominator for build-duration → step conversion.
        trajectory_id:       Label used in action_id keys and result.
        flush_at_end:        Apply remaining buffer after the final step.

    Returns:
        OnlineTrajectoryResult with per-step records and aggregated stats.
    """
    records = list(records)
    reveals = reveal_steps(records, timing_mode, delay_step_seconds)
    project_slug = records[0].context.project_slug if records else ""

    if not records:
        return OnlineTrajectoryResult(
            policy_id=policy.policy_id,
            trajectory_id=trajectory_id,
            project_slug=project_slug,
            total_steps=0,
            total_updates=0,
            total_censored_skipped=0,
            cumulative_cost=0.0,
            action_counts={a.value: 0 for a in Action},
        )

    # Explicit delays are drawn from build duration; the buffer's own sampler
    # is irrelevant (we pass delay= explicitly in every add() call).
    buffer = PendingRewardBuffer(rng=rng, min_delay=1, max_delay=1)

    finish_by_id = {}
    def in_observation_order(items):
        return sorted(items, key=lambda p: finish_by_id[p.reward.action_id]) if timing_mode == "event_time" else items

    step_records: list[OnlineStepRecord] = []
    cumulative_cost = 0.0
    total_updates = 0
    total_censored = 0
    action_counts: dict[str, int] = {a.value: 0 for a in Action}

    for step, record in enumerate(records):
        # 1. Release matured rewards and apply them to the policy.
        pending_before = len(buffer)
        matured = in_observation_order(buffer.pop_available(step))
        updates_this_step = 0
        for pending in matured:
            if pending.reward.censored or not math.isfinite(pending.reward.cost):
                continue
            policy.update(pending.context, pending.action, pending.reward)
            updates_this_step += 1
        total_updates += updates_this_step

        # 2. Policy decides without seeing the outcome.
        policy_action, _ = policy.select_action(record.context)
        action_counts[policy_action.value] += 1

        # 3. Compute counterfactual cost from logged outcome.
        effective = _effective_outcome(policy_action, record.outcome)
        cost = compute_cost(policy_action, effective, cost_config)
        is_censored = not math.isfinite(cost)
        if not is_censored:
            cumulative_cost += cost
        else:
            total_censored += 1

        # 4. Queue the reward; it will mature after `delay` steps.
        delay = reveals[step] - step
        action_id = f"{trajectory_id}:{step}:{record.context.commit_sha}"
        finish_by_id[action_id] = record.finished_at or record.started_at
        # Pass censored=True and cost=0.0 for NaN costs so the buffer doesn't
        # propagate NaN; the censored flag causes update() to skip.
        buffer.add(
            action_id=action_id,
            context=record.context,
            action=policy_action,
            outcome=effective,
            current_step=step,
            cost=cost if math.isfinite(cost) else 0.0,
            delay=delay,
            censored=is_censored,
        )

        step_records.append(OnlineStepRecord(
            step=step,
            policy_action=policy_action,
            logged_outcome=record.outcome,
            effective_outcome=effective,
            cost=cost,
            delay_steps=delay,
            updates_applied=updates_this_step,
            pending_count_before=pending_before,
            recent_failure_rate=record.context.recent_failure_rate,
            reveal_at_step=reveals[step],
        ))

    # 5. Flush remaining pending rewards so the policy has seen all feedback.
    if flush_at_end and len(buffer) > 0:
        # The furthest-future reveal step is at most:
        #   last_step_index + max(delay_from_record over all records)
        flush_step = max(reveals)
        for pending in in_observation_order(buffer.pop_available(flush_step)):
            if pending.reward.censored or not math.isfinite(pending.reward.cost):
                continue
            policy.update(pending.context, pending.action, pending.reward)
            total_updates += 1

    return OnlineTrajectoryResult(
        policy_id=policy.policy_id,
        trajectory_id=trajectory_id,
        project_slug=project_slug,
        total_steps=len(records),
        total_updates=total_updates,
        total_censored_skipped=total_censored,
        cumulative_cost=cumulative_cost,
        action_counts=action_counts,
        step_records=step_records,
    )


def run_online_experiment(
    policies: list[Policy],
    records_by_project: dict[str, list[TravisTorrentRecord]],
    *,
    cost_config: CostConfig,
    rng: np.random.Generator,
    delay_step_seconds: int = 60,
    flush_at_end: bool = True,
    timing_mode: str = "duration_steps",
) -> dict[str, list[OnlineTrajectoryResult]]:
    """Run online replay for each policy over every project trajectory.

    Each policy is reset before processing each project so project histories
    are independent.  Deterministic per-project RNG seeds are derived from
    the master ``rng`` so project ordering does not affect reproducibility.

    Returns:
        Dict mapping policy_id → list of OnlineTrajectoryResult (one per project).
    """
    results: dict[str, list[OnlineTrajectoryResult]] = {
        p.policy_id: [] for p in policies
    }
    project_keys = sorted(records_by_project)
    master_seed = int(rng.integers(0, 2**31))
    project_seeds = [int.from_bytes(hashlib.sha256(f"{master_seed}:{key}".encode()).digest()[:8], "little")
                     for key in project_keys]

    for project_key, project_seed in zip(project_keys, project_seeds):
        records = records_by_project[project_key]
        for policy in policies:
            policy.reset()
            # Project-local stochastic streams: adding another project cannot
            # consume this project's Thompson draws.
            if hasattr(policy, "_rng"):
                identity = f"{int(project_seed)}:{project_key}:{policy.policy_id}"
                local_seed = int.from_bytes(hashlib.sha256(identity.encode()).digest()[:8], "little")
                policy._rng = np.random.default_rng(local_seed)
            result = run_online_trajectory(
                policy=policy,
                records=records,
                cost_config=cost_config,
                rng=np.random.default_rng(int(project_seed)),
                delay_step_seconds=delay_step_seconds,
                trajectory_id=f"online:{project_key}",
                flush_at_end=flush_at_end,
                timing_mode=timing_mode,
            )
            results[policy.policy_id].append(result)

    return results
