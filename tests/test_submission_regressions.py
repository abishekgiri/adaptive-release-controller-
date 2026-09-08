"""Adversarial checks added during the submission audit (2026-09-05)."""
import csv
import math
from pathlib import Path

import numpy as np
import pytest

from data.loaders import TravisTorrentLoader
from data.schemas import Action, Outcome, Reward
from drift.detectors import PageHinkleyDetector
from evaluation.metrics import EpisodeRecord, cumulative_regret
from tests.test_replay_eval import FixedPolicy, _trajectory, _step
from evaluation.replay_eval import evaluate_ips
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from drift.detectors import PageHinkleyConfig
from evaluation.online_replay import reveal_steps, run_online_trajectory, run_online_experiment
from tests.test_online_replay import _make_record, _make_context
from experiments.run_bandits import OnlineExperimentConfig, build_policies
from rewards.cost_model import CostConfig


def test_overlapping_build_cannot_reveal_prior_outcome(tmp_path):
    path = tmp_path / "overlap.csv"
    rows = [
        dict(git_trigger_commit="a", gh_project_name="p", tr_status="failed",
             tr_started_at="2020-01-01T00:00:00Z", tr_finished_at="2020-01-01T01:00:00Z"),
        dict(git_trigger_commit="b", gh_project_name="p", tr_status="passed",
             tr_started_at="2020-01-01T00:01:00Z", tr_finished_at="2020-01-01T00:02:00Z"),
        dict(git_trigger_commit="c", gh_project_name="p", tr_status="passed",
             tr_started_at="2020-01-01T02:00:00Z", tr_finished_at="2020-01-01T02:01:00Z"),
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    records = list(TravisTorrentLoader(path, min_builds=1, min_history_days=0))
    assert records[1].context.recent_failure_rate == 0.0
    assert records[2].context.recent_failure_rate == 0.5


def test_constant_positive_stream_has_no_page_hinkley_alarms():
    detector = PageHinkleyDetector()
    alarms = 0
    for _ in range(1150):
        if detector.update(2.0):
            alarms += 1
            detector.reset()
    assert alarms == 0


def test_regret_keeps_joint_observation_alignment():
    record = EpisodeRecord("p", 0, costs=[10, math.nan, 4], oracle_costs=[1, 2, 3])
    np.testing.assert_allclose(cumulative_regret(record), [9, 10])


def test_ips_rejects_zero_support_instead_of_reporting_free_blocking():
    with pytest.raises(ValueError, match="support"):
        evaluate_ips(FixedPolicy(Action.BLOCK), _trajectory(_step(0, Action.DEPLOY, 10)))


def test_page_hinkley_matches_reference_upward_statistic():
    # Independent batch-mean calculation of River's documented one-sided
    # Page-Hinkley recurrence, including the first-observation minimum.
    stream = np.r_[np.full(100, 2.0), np.full(100, 7.0), np.full(100, 1.0)]
    detector = PageHinkleyDetector(PageHinkleyConfig(min_instances=30))
    history, sums, stat, detected = [], [], 0.0, False
    for value in stream:
        if detected:
            history, sums, stat = [], [], 0.0
        history.append(value)
        stat = 0.9999*stat + value - np.mean(history) - 0.005
        sums.append(stat)
        detected = len(history) >= 30 and stat-min(sums) > 50
        assert detector.update(value) == detected
        assert detector._mean == pytest.approx(np.mean(history))


def test_event_clock_uses_actual_completion_not_duration_divided_by_minutes():
    t = datetime(2020, 1, 1, tzinfo=timezone.utc)
    records = [replace(_make_record(i), started_at=t+timedelta(minutes=start),
                       finished_at=t+timedelta(minutes=finish))
               for i, (start, finish) in enumerate([(0,60), (1,2), (120,121)])]
    assert reveal_steps(records, "event_time", 60) == [2,2,3]


def test_censoring_is_identical_for_every_action():
    class NoUpdate(FixedPolicy):
        def update(self, *args):
            pass
    records = [_make_record(0, Outcome.SUCCESS), _make_record(1, Outcome.CENSORED)]
    for action in Action:
        result = run_online_trajectory(NoUpdate(action), records,
            cost_config=CostConfig(), rng=np.random.default_rng(0))
        assert result.total_censored_skipped == 1
        assert result.total_updates == 1


def test_project_stream_is_invariant_to_unrelated_project():
    records = [_make_record(i, Outcome.FAILURE if i%3==0 else Outcome.SUCCESS) for i in range(30)]
    cfg = OnlineExperimentConfig("test", Path("unused"))
    def actions(groups):
        ts = next(p for p in build_policies(cfg, 1) if p.policy_id == "thompson")
        rs = run_online_experiment([ts], groups, cost_config=CostConfig(), rng=np.random.default_rng(1))
        return [s.policy_action for s in rs["thompson"][-1].step_records]
    assert actions({"z": records}) == actions({"a": records, "z": records})


def test_drift_runner_decides_on_current_context_for_exact_horizon():
    from experiments.run_drift_eval import run_drift_trajectory
    class Spy(FixedPolicy):
        def __init__(self):
            super().__init__(Action.DEPLOY)
            self.steps = []
        def select_action(self, context):
            self.steps.append(context.step)
            return super().select_action(context)
        def update(self, *args):
            pass
    policy = Spy()
    result = run_drift_trajectory(policy, "abrupt", 0, 10, CostConfig(), delay_p=1.0)
    assert policy.steps == list(range(10))
    assert len(result.step_costs) == 10
    assert all(math.isfinite(c) for c in result.step_costs)
    assert result.total_censored == 0  # tail rewards are evaluated, not discarded


def test_cost_threshold_canary_interval_and_cost_rescaling():
    from policies.cost_rules import expected_costs
    assert min(expected_costs(.15, CostConfig()), key=expected_costs(.15, CostConfig()).get) == Action.CANARY
    for p in np.linspace(0, 1, 101):
        costs = expected_costs(p, CostConfig())
        scaled = expected_costs(p, CostConfig(**{k:7*v for k,v in CostConfig().__dict__.items()}))
        assert min(costs, key=costs.get) == min(scaled, key=scaled.get)


def test_artificial_delay_also_delays_history_features(tmp_path):
    path=tmp_path/'delayed.csv'
    rows=[dict(git_trigger_commit=str(i),gh_project_name='p',tr_status='failed' if i==0 else 'passed',
               tr_started_at=f'2020-01-0{i+1}T00:00:00Z',tr_duration='180',gh_tests_run='42')
          for i in range(4)]
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)
    records=list(TravisTorrentLoader(path,min_builds=1,min_history_days=0,
                 timing_mode='duration_steps',delay_step_seconds=60))
    assert records[1].context.recent_failure_rate==0
    assert records[1].context.tests_run==0
    assert records[3].context.recent_failure_rate==1
    assert records[3].context.tests_run==42


def test_github_pagination_does_not_change_page_size_midstream():
    from ingestion.github_client import GitHubClient
    client=GitHubClient('owner','repo')
    def page(per_page,page,branch=None):
        return [{'sha':str(i)} for i in range((page-1)*per_page,page*per_page)]
    client.list_commits=page
    client.list_workflow_runs=page
    assert [r['sha'] for r in client._collect_commits(150)]==[str(i) for i in range(150)]
    assert [r['sha'] for r in client._collect_workflow_runs(150)]==[str(i) for i in range(150)]


def test_window_rate_forgets_old_evidence_and_adjusts_canary_exposure():
    from experiments.run_drift_eval import MomentRatePolicy
    policy=MomentRatePolicy(CostConfig(),window=2)
    ctx=_make_context(0)
    for i,(action,outcome) in enumerate([(Action.DEPLOY,Outcome.FAILURE),(Action.CANARY,Outcome.SUCCESS),(Action.CANARY,Outcome.SUCCESS)]):
        policy.update(ctx,action,Reward(str(i),outcome,1,1,False,i+1))
    assert policy.failures==1  # only the pseudo-count; old failure aged out
    assert policy.exposure==pytest.approx(2.8)  # prior 2 + two canary exposures


def test_wrapper_and_plain_linucb_match_without_drift():
    from experiments.run_drift_eval import build_policies,run_drift_trajectory
    policies={p.policy_id:p for p in build_policies(3,CostConfig())}
    a=run_drift_trajectory(policies['linucb'],'abrupt',3,100,CostConfig())
    b=run_drift_trajectory(policies['linucb_with_drift_no_reset'],'abrupt',3,100,CostConfig())
    assert a.action_counts==b.action_counts
    np.testing.assert_allclose(a.step_costs,b.step_costs,equal_nan=True)
    assert a.cumulative_regret==b.cumulative_regret
