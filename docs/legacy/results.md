# Phase 17 Results: First Controlled Evidence Run

## Status

This is a preliminary smoke-test result, not a final research result.

The configured real-data input `data/raw/travistorrent.csv` was not present locally. The first attempt to run the existing pipeline with `experiments/configs/first_real_result.json` failed with:

```text
FileNotFoundError: data/raw/travistorrent.csv
```

To exercise the existing pipeline end to end, a gitignored TravisTorrent-shaped smoke CSV was generated locally at `data/raw/travistorrent.csv`. It contains 600 synthetic rows for one project over more than 365 days so it satisfies the current config filters. This file is not committed and should be replaced with the real TravisTorrent dump before making paper claims.

## Exact Commands Used

Initial real-data attempt:

```bash
python3 -m experiments.run_baselines --config experiments/configs/first_real_result.json --seed 0
```

Smoke data generation:

```bash
python3 -c '... wrote data/raw/travistorrent.csv with 600 synthetic TravisTorrent-shaped rows ...'
```

Experiment run:

```bash
for seed in $(seq 0 29); do python3 -m experiments.run_baselines --config experiments/configs/first_real_result.json --seed "$seed" >/tmp/phase17-seed-${seed}.log; done
```

Raw outputs were written to:

```text
experiments/results/first_real_result/<seed>/summary.json
experiments/results/first_real_result/<seed>/summary.md
```

The `experiments/results/` directory remains gitignored and is not committed.

## Dataset And Config

| Item | Value |
| --- | --- |
| Config | `experiments/configs/first_real_result.json` |
| Dataset path used by config | `data/raw/travistorrent.csv` |
| Actual dataset for this run | locally generated synthetic TravisTorrent-shaped smoke CSV |
| Rows | 600 |
| Projects / trajectories | 1 |
| Seeds | 0-29 |
| Delay setting | `delay_steps = max(1, ceil(tr_duration / 60))` |
| Bootstrap resamples | 1000 per seed in pipeline; 10000 for aggregate table below |
| Propensity clip | 20.0 |

## Policies Compared

| Policy | Status |
| --- | --- |
| `static_rules` | evaluated |
| `linucb` | evaluated |
| `cost_sensitive_bandit` | evaluated |
| `heuristic-score` | TODO placeholder |
| `offline-classifier` | TODO placeholder |
| `thompson` | TODO placeholder |

## Metrics Table

Bootstrap confidence intervals are computed across the 30 seed-level summaries. Because this smoke run is deterministic for the implemented replay policies, the intervals collapse to a single value.

| Policy | Mean cumulative cost | Cost 95% CI | Mean IPS value | Value 95% CI | Mean matched actions | Mean ESS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `static_rules` | 600.0000 | [600.0000, 600.0000] | -1.0000 | [-1.0000, -1.0000] | 397.00 | 397.00 |
| `linucb` | 980.0000 | [980.0000, 980.0000] | -1.6333 | [-1.6333, -1.6333] | 600.00 | 600.00 |
| `cost_sensitive_bandit` | 980.0000 | [980.0000, 980.0000] | -1.6333 | [-1.6333, -1.6333] | 600.00 | 600.00 |

Additional diagnostics:

| Metric | Value |
| --- | ---: |
| Evaluated steps per seed | 600 |
| Skipped censored rewards | 0 |
| Seed count | 30 |
| Output directories | 30 |

## Short Interpretation

This run proves that the current Phase 14/15 pipeline can produce raw per-seed outputs and aggregate cost-first evidence tables for the three implemented policies.

It does not prove the research claim yet. In this smoke replay, `linucb` and `cost_sensitive_bandit` produce identical results because `evaluation/replay_eval.py` evaluates candidate policies with IPS but does not perform online training during replay. Both bandit policies therefore act from their initial model state and choose `DEPLOY` for every logged context. Since the synthetic smoke log also records `DEPLOY` as the logged action for every row, both bandits match all 600 actions and receive the same cost.

`static_rules` has lower cumulative observed IPS cost in this smoke run because it deploys only on lower-risk rows and therefore matches fewer logged actions. This is a useful evaluator diagnostic, but not a final comparison: lower matched-action coverage changes what evidence the IPS estimate can use.

## Limitations

- This is synthetic TravisTorrent-shaped smoke data, not the real TravisTorrent dataset.
- The run used one project trajectory, so project-level uncertainty is not represented.
- The aggregate bootstrap CI is degenerate because the implemented replay path is deterministic across seeds for this dataset.
- IPS replay currently evaluates policies but does not train LinUCB or the cost-sensitive bandit online before scoring later steps.
- The logging policy is effectively deploy-only in this smoke data, so `CANARY` and `BLOCK` have limited counterfactual coverage.
- These numbers must not be used as final paper evidence.

## Next Evidence Step

Before making research claims, replace the smoke CSV with the real TravisTorrent dump and add an online replay/simulation experiment path that applies delayed updates during the trajectory. The final evidence table should then be regenerated with real multi-project trajectories and non-degenerate uncertainty estimates.
