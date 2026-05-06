"""Phase C end-to-end smoke test."""

from __future__ import annotations

from experiments.phase_c_smoke import EXPECTED_CUMULATIVE_COST, run_smoke


def test_phase_c_smoke_matches_manual_five_step_cost() -> None:
    """Synthetic env, static policy, and cost model must produce the manual cost."""

    result = run_smoke(seed=0, write_results=False)

    assert result.cumulative_cost == EXPECTED_CUMULATIVE_COST
    assert [step.action for step in result.steps] == [
        "deploy",
        "canary",
        "block",
        "deploy",
        "canary",
    ]
    assert [step.outcome for step in result.steps] == [
        "success",
        "failure",
        "failure",
        "failure",
        "success",
    ]
    assert [step.cost for step in result.steps] == [0.0, 4.0, 0.5, 10.0, 1.0]
