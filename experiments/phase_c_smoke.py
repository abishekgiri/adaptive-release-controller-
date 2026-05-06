"""Phase C smoke experiment: synthetic env -> static policy -> cost model."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from data.schemas import Action, Context, Outcome, Reward
from environment.synthetic import SyntheticEnvironment
from policies.static_rules import StaticRulesPolicy
from rewards.cost_model import CostConfig, compute_cost


EXPECTED_CUMULATIVE_COST = 15.5
DEFAULT_RESULTS_ROOT = Path("experiments/results")


@dataclass(frozen=True)
class SmokeStep:
    """One hand-checkable step in the Phase C smoke trajectory."""

    step: int
    action: str
    outcome: str
    cost: float


@dataclass(frozen=True)
class SmokeResult:
    """Smoke run summary."""

    cumulative_cost: float
    expected_cumulative_cost: float
    steps: tuple[SmokeStep, ...]


class DeterministicSmokeEnvironment(SyntheticEnvironment):
    """Five-step synthetic fixture with fixed contexts and outcomes.

    The class inherits from SyntheticEnvironment so the smoke path still uses
    the Phase C environment interface, but fixes the trajectory so the expected
    cumulative cost can be calculated by hand:

        deploy + success = 0.0
        canary + failure = 4.0
        block + failure = 0.5
        deploy + failure = 10.0
        canary + success = 1.0
        total = 15.5
    """

    def __init__(self, seed: int) -> None:
        super().__init__(
            rng=np.random.default_rng(seed),
            horizon=5,
            delay_p=1.0,
            max_delay=10,
        )
        self._contexts = (
            _context(step=0, files_changed=2, src_churn=40, recent_failure_rate=0.05),
            _context(step=1, files_changed=12, src_churn=260, recent_failure_rate=0.10),
            _context(step=2, files_changed=25, src_churn=1200, recent_failure_rate=0.40),
            _context(step=3, files_changed=1, src_churn=20, recent_failure_rate=0.01),
            _context(
                step=4,
                files_changed=4,
                src_churn=80,
                recent_failure_rate=0.04,
                has_risky_path_change=True,
            ),
        )
        self._outcomes = (
            Outcome.SUCCESS,
            Outcome.FAILURE,
            Outcome.FAILURE,
            Outcome.FAILURE,
            Outcome.SUCCESS,
        )

    def observe(self) -> Context:
        """Return the fixed context for the current smoke step."""
        index = min(self.current_step, len(self._contexts) - 1)
        return self._contexts[index]

    def step(self, action: Action) -> None:  # type: ignore[override]
        """Schedule the fixed outcome for the current smoke step."""
        if self.done:
            raise RuntimeError("Cannot step after smoke trajectory is done")

        index = self.current_step
        reward = Reward(
            action_id=f"phase_c_smoke_{index}",
            outcome=self._outcomes[index],
            cost=float("nan"),
            delay_steps=1,
            censored=False,
            observed_at_step=index + 1,
        )
        self._pending.append((index + 1, reward))
        return None


def run_smoke(
    *,
    config_name: str = "phase_c_smoke",
    seed: int = 0,
    results_root: Path = DEFAULT_RESULTS_ROOT,
    write_results: bool = True,
) -> SmokeResult:
    """Run the five-step smoke trajectory and return the cumulative cost."""

    env = DeterministicSmokeEnvironment(seed=seed)
    policy = StaticRulesPolicy()
    cost_config = CostConfig()
    env.reset()

    steps: list[SmokeStep] = []
    for _ in range(5):
        context = env.observe()
        action, _ = policy.select_action(context)
        env.step(action)
        matured = env.advance_time()

        if len(matured) != 1:
            raise RuntimeError(f"Expected one matured reward, got {len(matured)}")

        reward = matured[0]
        cost = compute_cost(action, reward.outcome, cost_config)
        steps.append(
            SmokeStep(
                step=context.step,
                action=action.value,
                outcome=reward.outcome.value,
                cost=cost,
            )
        )

    cumulative_cost = round(sum(step.cost for step in steps), 6)
    result = SmokeResult(
        cumulative_cost=cumulative_cost,
        expected_cumulative_cost=EXPECTED_CUMULATIVE_COST,
        steps=tuple(steps),
    )

    if cumulative_cost != EXPECTED_CUMULATIVE_COST:
        raise AssertionError(
            f"Smoke cumulative cost {cumulative_cost} != manual "
            f"{EXPECTED_CUMULATIVE_COST}"
        )

    if write_results:
        _write_result(result, config_name=config_name, seed=seed, results_root=results_root)

    return result


def _context(
    *,
    step: int,
    files_changed: int,
    src_churn: int,
    recent_failure_rate: float,
    has_risky_path_change: bool = False,
) -> Context:
    """Build a minimal pre-action context for the smoke trajectory."""

    lines_added = src_churn // 2
    lines_deleted = src_churn - lines_added
    return Context(
        commit_sha=f"phase-c-smoke-{step}",
        project_slug="synthetic/smoke",
        step=step,
        files_changed=files_changed,
        lines_added=lines_added,
        lines_deleted=lines_deleted,
        src_churn=src_churn,
        is_pr=False,
        tests_run=100,
        tests_added=2,
        build_duration_s=90.0,
        author_experience=5,
        recent_failure_rate=recent_failure_rate,
        has_dependency_change=False,
        has_risky_path_change=has_risky_path_change,
    )


def _write_result(
    result: SmokeResult,
    *,
    config_name: str,
    seed: int,
    results_root: Path,
) -> None:
    """Write the smoke result under experiments/results/<config>/<seed>/."""

    output_dir = results_root / config_name / str(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = asdict(result)
    (output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )


def _config_name(value: str) -> str:
    """Treat --config as either a simple name or a config path."""

    path = Path(value)
    return path.stem if path.suffix else value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="phase_c_smoke")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    result = run_smoke(config_name=_config_name(args.config), seed=args.seed)
    print(f"{result.cumulative_cost:.4f}")


if __name__ == "__main__":
    main()
