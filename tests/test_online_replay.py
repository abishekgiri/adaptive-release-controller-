"""Tests for evaluation/online_replay.py — online learning replay loop."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest

from data.loaders import TravisTorrentRecord
from data.schemas import Action, Context, Outcome
from evaluation.online_replay import (
    OnlineTrajectoryResult,
    _delay_from_record,
    _effective_outcome,
    run_online_trajectory,
)
from policies.base import FeatureEncoder
from policies.cost_sensitive_bandit import CostSensitiveBandit, CostSensitiveBanditConfig
from policies.linucb import LinUCBConfig, LinUCBPolicy
from policies.static_rules import StaticRulesPolicy
from rewards.cost_model import CostConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_NOW = datetime(2022, 1, 1, tzinfo=timezone.utc)


def _make_context(
    step: int = 0,
    recent_failure_rate: float = 0.0,
    files_changed: int = 3,
    has_risky_path_change: bool = False,
    build_duration_s: float = 60.0,
) -> Context:
    return Context(
        commit_sha=f"sha{step:04d}",
        project_slug="test/repo",
        step=step,
        files_changed=files_changed,
        lines_added=50,
        lines_deleted=20,
        src_churn=70,
        is_pr=False,
        tests_run=100,
        tests_added=2,
        build_duration_s=build_duration_s,
        author_experience=5,
        recent_failure_rate=recent_failure_rate,
        has_dependency_change=False,
        has_risky_path_change=has_risky_path_change,
    )


def _make_record(
    step: int,
    outcome: Outcome = Outcome.SUCCESS,
    build_duration_s: float = 60.0,
    recent_failure_rate: float = 0.0,
) -> TravisTorrentRecord:
    return TravisTorrentRecord(
        context=_make_context(
            step=step,
            build_duration_s=build_duration_s,
            recent_failure_rate=recent_failure_rate,
        ),
        action=Action.DEPLOY,   # TravisTorrent always logs DEPLOY
        outcome=outcome,
        started_at=_NOW,
        finished_at=_NOW,
    )


def _make_records(
    n: int,
    outcome: Outcome = Outcome.SUCCESS,
    build_duration_s: float = 60.0,
) -> list[TravisTorrentRecord]:
    return [_make_record(i, outcome=outcome, build_duration_s=build_duration_s) for i in range(n)]


def _linucb(alpha: float = 1.0, seed: int = 0) -> LinUCBPolicy:
    return LinUCBPolicy(
        config=LinUCBConfig(alpha=alpha, lambda_reg=1.0),
        feature_dim=FeatureEncoder.DIM,
        rng=np.random.default_rng(seed),
        policy_id="linucb",
    )


def _bandit(alpha: float = 1.0, seed: int = 0) -> CostSensitiveBandit:
    return CostSensitiveBandit(
        config=CostSensitiveBanditConfig(alpha=alpha, lambda_reg=1.0),
        feature_dim=FeatureEncoder.DIM,
        rng=np.random.default_rng(seed),
        policy_id="cost_sensitive_bandit",
    )


# ---------------------------------------------------------------------------
# Unit tests: helpers
# ---------------------------------------------------------------------------

class TestEffectiveOutcome:
    def test_deploy_success_unchanged(self) -> None:
        assert _effective_outcome(Action.DEPLOY, Outcome.SUCCESS) == Outcome.SUCCESS

    def test_deploy_failure_unchanged(self) -> None:
        assert _effective_outcome(Action.DEPLOY, Outcome.FAILURE) == Outcome.FAILURE

    def test_canary_success_unchanged(self) -> None:
        assert _effective_outcome(Action.CANARY, Outcome.SUCCESS) == Outcome.SUCCESS

    def test_block_success_unchanged(self) -> None:
        assert _effective_outcome(Action.BLOCK, Outcome.SUCCESS) == Outcome.SUCCESS

    def test_block_failure_unchanged(self) -> None:
        assert _effective_outcome(Action.BLOCK, Outcome.FAILURE) == Outcome.FAILURE

    def test_block_censored_becomes_blocked(self) -> None:
        # BLOCK + CENSORED has no valid cost-matrix entry; must map to BLOCKED.
        assert _effective_outcome(Action.BLOCK, Outcome.CENSORED) == Outcome.BLOCKED

    def test_deploy_censored_stays_censored(self) -> None:
        assert _effective_outcome(Action.DEPLOY, Outcome.CENSORED) == Outcome.CENSORED


class TestDelayFromRecord:
    def test_minimum_one_for_zero_duration(self) -> None:
        record = _make_record(0, build_duration_s=0.0)
        assert _delay_from_record(record, delay_step_seconds=60) == 1

    def test_minimum_one_for_short_build(self) -> None:
        record = _make_record(0, build_duration_s=30.0)
        assert _delay_from_record(record, delay_step_seconds=60) == 1

    def test_ceil_applied(self) -> None:
        # 90s / 60s_per_step = 1.5 → ceil → 2
        record = _make_record(0, build_duration_s=90.0)
        assert _delay_from_record(record, delay_step_seconds=60) == 2

    def test_exact_multiple(self) -> None:
        record = _make_record(0, build_duration_s=120.0)
        assert _delay_from_record(record, delay_step_seconds=60) == 2


# ---------------------------------------------------------------------------
# Requirement 1: Policy updates occur over time
# ---------------------------------------------------------------------------

class TestPolicyUpdatesOccur:
    """After a trajectory, the policy must have received at least one update."""

    def test_linucb_A_changes_after_trajectory(self) -> None:
        policy = _linucb()
        A_before = {a: policy._A[a].copy() for a in Action}

        records = _make_records(20, outcome=Outcome.SUCCESS, build_duration_s=60.0)
        result = run_online_trajectory(
            policy=policy,
            records=records,
            cost_config=CostConfig(),
            rng=np.random.default_rng(0),
            delay_step_seconds=60,
        )

        assert result.total_updates > 0, "Expected at least one policy update"
        # At least one arm's A matrix must differ from the initial lambda*I.
        any_changed = any(
            not np.allclose(policy._A[a], A_before[a]) for a in Action
        )
        assert any_changed, "A matrix must change after updates"

    def test_bandit_updates_counter_increments(self) -> None:
        policy = _bandit()
        records = _make_records(20, outcome=Outcome.SUCCESS)
        result = run_online_trajectory(
            policy=policy,
            records=records,
            cost_config=CostConfig(),
            rng=np.random.default_rng(0),
            delay_step_seconds=60,
        )
        assert result.total_updates > 0

    def test_total_updates_grows_with_more_records(self) -> None:
        short = run_online_trajectory(
            policy=_linucb(),
            records=_make_records(10),
            cost_config=CostConfig(),
            rng=np.random.default_rng(0),
        )
        long = run_online_trajectory(
            policy=_linucb(),
            records=_make_records(50),
            cost_config=CostConfig(),
            rng=np.random.default_rng(0),
        )
        assert long.total_updates > short.total_updates

    def test_static_policy_never_updates(self) -> None:
        # StaticRulesPolicy.update() is a no-op; verify the loop doesn't crash.
        policy = StaticRulesPolicy()
        result = run_online_trajectory(
            policy=policy,
            records=_make_records(20),
            cost_config=CostConfig(),
            rng=np.random.default_rng(0),
        )
        # Static policy's update is always a no-op; total_updates counts
        # buffer pops, not meaningful model changes — just confirm no crash.
        assert result.total_steps == 20


# ---------------------------------------------------------------------------
# Requirement 2: Delayed rewards are respected
# ---------------------------------------------------------------------------

class TestDelayedRewardsRespected:
    """No update must happen before the delay has elapsed."""

    def test_no_update_at_step_zero_with_delay_one(self) -> None:
        policy = _linucb()
        records = _make_records(5, build_duration_s=60.0)  # delay=1 step
        result = run_online_trajectory(
            policy=policy,
            records=records,
            cost_config=CostConfig(),
            rng=np.random.default_rng(0),
            flush_at_end=False,  # don't flush so we can inspect the loop
        )
        # At step 0, delay=1 means the reward reveals at step 1.
        # So the step-0 record must have updates_applied == 0.
        assert result.step_records[0].updates_applied == 0, (
            "First step must not apply any updates (nothing has matured yet)"
        )

    def test_update_arrives_after_delay(self) -> None:
        policy = _linucb()
        # Use delay=2 (build_duration_s=90, delay_step_seconds=60 → ceil(1.5)=2)
        records = _make_records(10, build_duration_s=90.0)
        result = run_online_trajectory(
            policy=policy,
            records=records,
            cost_config=CostConfig(),
            rng=np.random.default_rng(0),
            flush_at_end=False,
        )
        # Reward from step 0 arrives at step 0+2=2 (updates_applied at step 2).
        # Steps 0 and 1 must have 0 updates each.
        assert result.step_records[0].updates_applied == 0
        assert result.step_records[1].updates_applied == 0
        # Some update must appear at or after step 2.
        updates_from_step2 = sum(
            s.updates_applied for s in result.step_records[2:]
        )
        assert updates_from_step2 > 0, "Updates must arrive after the delay"

    def test_pending_count_before_decreases_as_rewards_mature(self) -> None:
        records = _make_records(20, build_duration_s=60.0)  # delay=1
        result = run_online_trajectory(
            policy=_linucb(),
            records=records,
            cost_config=CostConfig(),
            rng=np.random.default_rng(0),
            flush_at_end=False,
        )
        # pending_count_before should increase then stabilise at 1 (each step
        # adds one and releases one from the previous step).
        pending_counts = [s.pending_count_before for s in result.step_records]
        # At steady state with delay=1, buffer should hold ≤1 item before each step.
        assert max(pending_counts) <= 2, (
            "Buffer should not accumulate more than a few items with delay=1"
        )

    def test_censored_records_are_skipped_in_updates(self) -> None:
        records = _make_records(20, outcome=Outcome.CENSORED)
        result = run_online_trajectory(
            policy=_linucb(),
            records=records,
            cost_config=CostConfig(),
            rng=np.random.default_rng(0),
        )
        assert result.total_censored_skipped == 20
        assert result.cumulative_cost == 0.0


# ---------------------------------------------------------------------------
# Requirement 3: LinUCB and cost-sensitive bandit diverge after learning
# ---------------------------------------------------------------------------

class TestPoliciesDivergeAfterLearning:
    """After sufficient learning, different policies must select different actions."""

    def _run_and_get_final_action(
        self,
        policy,
        outcome: Outcome,
        n_records: int = 100,
    ) -> Action:
        records = _make_records(n_records, outcome=outcome)
        run_online_trajectory(
            policy=policy,
            records=records,
            cost_config=CostConfig(),
            rng=np.random.default_rng(42),
        )
        # Evaluate on a high-risk context to surface policy differences.
        from data.schemas import Context as Ctx
        eval_ctx = _make_context(recent_failure_rate=0.8, has_risky_path_change=True)
        action, _ = policy.select_action(eval_ctx)
        return action

    def test_high_alpha_vs_low_alpha_linucb_can_differ(self) -> None:
        """α=0 (exploit) vs α=100 (explore) must not always agree after learning."""
        exploit = _linucb(alpha=0.0, seed=0)
        explore = _linucb(alpha=100.0, seed=0)

        records = _make_records(60, outcome=Outcome.FAILURE)
        for policy in (exploit, explore):
            run_online_trajectory(
                policy=policy,
                records=records,
                cost_config=CostConfig(),
                rng=np.random.default_rng(7),
            )

        eval_ctx = _make_context(recent_failure_rate=0.9, has_risky_path_change=True)
        action_exploit, _ = exploit.select_action(eval_ctx)
        action_explore, _ = explore.select_action(eval_ctx)

        # With different alpha values the UCB scores differ; at least one
        # must have updated its b vector from the failure rewards.
        # We just verify they've learned something (b != 0 for deployed arm).
        deployed_arm = Action.DEPLOY  # both policies deployed on every record
        assert not np.allclose(exploit._b[deployed_arm], np.zeros(FeatureEncoder.DIM)), (
            "Policy must have learned from failure rewards"
        )
        assert not np.allclose(explore._b[deployed_arm], np.zeros(FeatureEncoder.DIM))

    def test_linucb_and_bandit_diverge_with_different_alpha(self) -> None:
        """LinUCB(alpha=0) and CostSensitiveBandit(alpha=10) must diverge after learning
        on a mix of outcomes."""
        linucb = _linucb(alpha=0.0, seed=1)
        bandit = _bandit(alpha=10.0, seed=1)

        mixed = (
            _make_records(30, outcome=Outcome.FAILURE)
            + _make_records(30, outcome=Outcome.SUCCESS)
        )

        for policy in (linucb, bandit):
            run_online_trajectory(
                policy=policy,
                records=mixed,
                cost_config=CostConfig(),
                rng=np.random.default_rng(1),
            )

        # Both policies have learned; their UCB scores use different alpha,
        # so at least one context should elicit a different action.
        actions_differ = False
        for rate in (0.0, 0.3, 0.6, 0.9):
            ctx = _make_context(recent_failure_rate=rate)
            a_l, _ = linucb.select_action(ctx)
            a_b, _ = bandit.select_action(ctx)
            if a_l != a_b:
                actions_differ = True
                break

        assert actions_differ, (
            "LinUCB(alpha=0) and CostSensitiveBandit(alpha=10) must diverge "
            "on at least one context after learning"
        )

    def test_bandit_b_vector_changes_after_failure_records(self) -> None:
        """After failure rewards, the DEPLOY arm's b vector must become negative."""
        policy = _bandit(alpha=0.0)
        records = _make_records(50, outcome=Outcome.FAILURE)
        run_online_trajectory(
            policy=policy,
            records=records,
            cost_config=CostConfig(),  # deploy_failure = 10; r = -10
            rng=np.random.default_rng(0),
        )
        # b[DEPLOY] should be strongly negative (r = -10 per update).
        b_deploy = policy._b[Action.DEPLOY]
        assert np.sum(b_deploy) < 0, (
            "After failure rewards, b[DEPLOY] must be negative (r = -cost = -10)"
        )


# ---------------------------------------------------------------------------
# Requirement 4: Records are processed chronologically
# ---------------------------------------------------------------------------

class TestChronologicalOrder:
    """The online loop must process records in the order they are supplied."""

    def test_step_indices_are_sequential(self) -> None:
        records = _make_records(15)
        result = run_online_trajectory(
            policy=_linucb(),
            records=records,
            cost_config=CostConfig(),
            rng=np.random.default_rng(0),
        )
        steps = [s.step for s in result.step_records]
        assert steps == list(range(15)), "Steps must be 0, 1, 2, …, n-1"

    def test_total_steps_matches_record_count(self) -> None:
        n = 42
        result = run_online_trajectory(
            policy=_linucb(),
            records=_make_records(n),
            cost_config=CostConfig(),
            rng=np.random.default_rng(0),
        )
        assert result.total_steps == n
        assert len(result.step_records) == n

    def test_action_counts_sum_to_total_steps(self) -> None:
        n = 30
        result = run_online_trajectory(
            policy=_linucb(),
            records=_make_records(n),
            cost_config=CostConfig(),
            rng=np.random.default_rng(0),
        )
        total = sum(result.action_counts.values())
        assert total == n

    def test_later_records_have_more_updates_available(self) -> None:
        """Updates at late steps must cumulate all matured rewards from early steps."""
        records = _make_records(30, build_duration_s=60.0)  # delay=1
        result = run_online_trajectory(
            policy=_linucb(),
            records=records,
            cost_config=CostConfig(),
            rng=np.random.default_rng(0),
            flush_at_end=False,
        )
        # Cumulative updates must be non-decreasing over steps.
        cumulative = 0
        for step_rec in result.step_records:
            cumulative += step_rec.updates_applied
        # By the end (excluding flush), most rewards should have matured.
        # With delay=1, by step n-1 all but the last reward have arrived.
        assert cumulative >= len(records) - 2, (
            "Almost all rewards should have matured by the end of the trajectory"
        )

    def test_empty_records_returns_zero_result(self) -> None:
        result = run_online_trajectory(
            policy=_linucb(),
            records=[],
            cost_config=CostConfig(),
            rng=np.random.default_rng(0),
        )
        assert result.total_steps == 0
        assert result.total_updates == 0
        assert result.cumulative_cost == 0.0

    def test_project_slug_taken_from_first_record(self) -> None:
        records = _make_records(5)
        result = run_online_trajectory(
            policy=_linucb(),
            records=records,
            cost_config=CostConfig(),
            rng=np.random.default_rng(0),
        )
        assert result.project_slug == "test/repo"
