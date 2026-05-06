"""Online replay evaluation for learning policies.

WARNING — SIMULATION, NOT CAUSAL INFERENCE
==========================================
Each step computes cost as ``compute_cost(policy_action, logged_outcome)`` using
the logged CI outcome as a counterfactual proxy for the policy's chosen action.
This is valid only under the assumption that the CI outcome is independent of
the deployment action (i.e. the build result was determined before the policy
acted).  This holds approximately for TravisTorrent because CI runs before
deployment decisions are made.

Do NOT report these numbers as unbiased estimates of real-world cost.
Use them to:
  - Verify that policies actually learn (weights change, actions diverge).
  - Debug the delayed-update pipeline.
  - Compare learning curves across policies on the same trajectory.

For unbiased offline policy evaluation see evaluation/replay_eval.py (IPS).
"""

from __future__ import annotations

import math
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
    """Map (policy_action, logged_outcome) to a valid cost-matrix outcome.

    The only special case: BLOCK + CENSORED → BLOCKED, because the canonical
    cost matrix has no (BLOCK, CENSORED) entry — a blocked commit's counterfactual
    is unknown rather than simply unobserved.
    """
    if policy_action == Action.BLOCK and logged_outcome == Outcome.CENSORED:
        return Outcome.BLOCKED
    return logged_outcome


def _delay_from_record(record: TravisTorrentRecord, delay_step_seconds: int) -> int:
    """Convert build duration to a discrete delay step count (minimum 1)."""
    if record.context.build_duration_s <= 0:
        return 1
    return max(1, math.ceil(record.context.build_duration_s / delay_step_seconds))


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

    step_records: list[OnlineStepRecord] = []
    cumulative_cost = 0.0
    total_updates = 0
    total_censored = 0
    action_counts: dict[str, int] = {a.value: 0 for a in Action}

    for step, record in enumerate(records):
        # 1. Release matured rewards and apply them to the policy.
        pending_before = len(buffer)
        matured = buffer.pop_available(step)
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
        delay = _delay_from_record(record, delay_step_seconds)
        action_id = f"{trajectory_id}:{step}:{record.context.commit_sha}"
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
        ))

    # 5. Flush remaining pending rewards so the policy has seen all feedback.
    if flush_at_end and len(buffer) > 0:
        # The furthest-future reveal step is at most:
        #   last_step_index + max(delay_from_record over all records)
        max_delay_seen = max(
            _delay_from_record(r, delay_step_seconds) for r in records
        )
        flush_step = len(records) + max_delay_seen
        for pending in buffer.pop_available(flush_step):
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
    project_seeds = rng.integers(0, 2**31, size=len(project_keys))

    for project_key, project_seed in zip(project_keys, project_seeds):
        records = records_by_project[project_key]
        for policy in policies:
            policy.reset()
            result = run_online_trajectory(
                policy=policy,
                records=records,
                cost_config=cost_config,
                rng=np.random.default_rng(int(project_seed)),
                delay_step_seconds=delay_step_seconds,
                trajectory_id=f"online:{project_key}",
                flush_at_end=flush_at_end,
            )
            results[policy.policy_id].append(result)

    return results
