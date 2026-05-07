# Source of Truth — Numerical Claims

Maps every numerical claim in `paper/adaptive-deployment-control.md` to the config,
seed(s), and result file that produced it. Update this file whenever a number changes.

Generated: 2026-05-06  
Paper commit: `9012cd3`  
Last updated for: fix-1 (Thompson ±58→±72), fix-2 (drift_resets now stored in ablation JSON), OPTION-A-1 (30-seed re-run; Thompson synthetic 1877±72→1903±54, Thompson real 585±99→624±72)

---

## Hyperparameters (all experiments unless overridden)

| Parameter | Value | Source |
|---|---|---|
| LinUCB α | 1.0 | `experiments/configs/online_smoke.json` |
| LinUCB λ | 1.0 | `experiments/configs/online_smoke.json` |
| Delay step | 60 s | `experiments/configs/online_smoke.json` |
| deploy\_failure cost | 10.0 | `experiments/configs/online_smoke.json → cost_config` |
| block\_bad cost | 0.5 | `experiments/configs/online_smoke.json → cost_config` |
| Asymmetry ratio | 20:1 | derived: 10.0 / 0.5 |
| PageHinkley λ\_PH | 50.0 | `drift/detectors.py:72–76` (PageHinkleyConfig default) |
| Feature dim | 13 | `policies/base.py::FeatureEncoder.DIM` |

---

## §5.1 — Main Results (Synthetic Smoke)

Config: `experiments/configs/online_smoke.json`  
Dataset: `data/raw/travistorrent_smoke.csv` (1 150 rows, 2 projects)  
Seeds: 0–29 (n=30); bootstrap seed 42  
Result files: `experiments/results/online_smoke/{0..29}/online_summary.json`

| Claim | Seeds | Raw values | Computed | Paper text |
|---|---|---|---|---|
| static\_rules cost | 0–29 (deterministic) | 1878.0 (all seeds identical) | — | 1878 |
| linucb cost | 0–29 (deterministic) | 1879.0 (all seeds identical) | — | 1879 |
| heuristic\_score cost | 0–29 (deterministic) | 2319.0 (all seeds identical) | — | 2319 |
| thompson cost range | 0–29 | 1814.5–2010.0 | — | 1814–2010 |
| Thompson mean | 0–29 | all 30 values in result files | **1903.4 ≈ 1903** | 1903 |
| Thompson std (sample, ddof=1) | 0–29 | sample\_std | **54.1 ≈ 54** | ±54 |
| Thompson 95% CI (bootstrap seed 42) | 0–29 | boot percentiles 2.5/97.5 | **[1884.5, 1921.9] ≈ [1884, 1922]** | [1884, 1922] |
| Thompson mean/step | 0–29 | 1903.4/1150 | **1.6551 ≈ 1.655** | 1.655 |
| Thompson std/step | 0–29 | 54.1/1150 | **0.0470 ≈ 0.047** | ±0.047 |
| Thompson action fractions (mean) | 0–29 | per-file fracs | deploy=6.5%, canary=22.4%, block=71.1% | 6.5%/22.4%/71.1% |
| LinUCB action fractions (mean) | 0–29 (deterministic) | per-file fracs | deploy=8.2%, canary=5.7%, block=86.3% | 5.7%/86.3% (§5.1 body) |
| Updates per arm — smoke/alpha | derived | 600 steps ÷ 3 arms | **200** | 200 (§1.3 supp., §5.1) |
| Updates per arm — smoke/beta | derived | 550 steps ÷ 3 arms | **183** | 183 (§1.3 supp., §5.1) |
| O(d²) convergence threshold | derived | d=13; 13²=169 | **169** | 169 (§1.3 Condition 2, §5.1) |
| ~~Thompson mean (n=5 pilot)~~ | ~~0–4~~ | ~~1877.7~~ | ~~SUPERSEDED by n=30~~ | ~~retired~~ |

---

## §5.2 — Robustness Analysis

All configs share `data/raw/travistorrent_smoke.csv`, delay=60 s, seeds 0–29 (n=30); bootstrap seed 42.  
Note: all 30 seeds return identical values for robustness configs (deterministic under fixed dataset). Robustness bootstrap CI = [value, value] (zero width, as expected).

### High-failure scenario (deploy\_failure=20)

Config: `experiments/configs/robustness_high_failure.json`  
Result files: `experiments/results/robustness_high_failure/{0..4}/online_summary.json`

| Claim | Seeds | static | linucb | Gain | Paper text |
|---|---|---|---|---|---|
| Cumulative cost | 0–4 | 2564.0 | 2078.5 | (2564−2078.5)/2564 = **18.9% ≈ 19%** | 19% |

### Low block-penalty scenario (block\_bad=1)

Config: `experiments/configs/robustness_low_block.json`  
Result files: `experiments/results/robustness_low_block/{0..4}/online_summary.json`

| Claim | Seeds | static | linucb | Gain | Paper text |
|---|---|---|---|---|---|
| Cumulative cost | 0–4 | 1584.0 | 1163.5 | (1584−1163.5)/1584 = **26.5% ≈ 27%** | 27% |

### Short-delay scenario (delay=30 s)

Config: `experiments/configs/robustness_short_delay.json`  
Result files: `experiments/results/robustness_short_delay/{0..4}/online_summary.json`

| Claim | Seeds | static | linucb | Diff | Paper text |
|---|---|---|---|---|---|
| Cumulative cost | 0–4 | 1878.0 | 1849.5 | 1878−1849.5 = **28.5 ≈ 29** | saves 29 units |

### Long-delay scenario (delay=120 s)

Config: `experiments/configs/robustness_long_delay.json`  
Result files: `experiments/results/robustness_long_delay/{0..4}/online_summary.json`

| Claim | Seeds | static | linucb | Diff | Paper text |
|---|---|---|---|---|---|
| Cumulative cost | 0–4 | 1878.0 | 1896.5 | 1896.5−1878 = **18.5 ≈ 19** | costs 19 more |

---

## §5.3 — Ablation Study

Config: `AblationConfig` defaults in `experiments/run_ablations.py` (no JSON config file)  
Dataset: `data/raw/travistorrent_smoke.csv`  
Seed: 0 only  
Result file: `experiments/results/ablation_smoke/0/ablation_summary.json`

| Claim | Key | Raw value | Computed | Paper text |
|---|---|---|---|---|
| full variant cost | full.cumulative\_cost | 2383.0 | — | 2383 |
| no\_delay variant cost | no\_delay.cumulative\_cost | 1857.5 | — | 1857.5 |
| no\_cost variant cost | no\_cost.cumulative\_cost | 2469.0 | — | 2469 |
| no\_drift variant cost | no\_drift.cumulative\_cost | 1879.0 | — | 1879 |
| Binary-reward degradation | (no\_cost−no\_drift)/no\_drift | (2469−1879)/1879 | **31.4% ≈ 31%** | 31% |
| Buffer delay overhead | (no\_drift−no\_delay)/no\_drift | (1879−1857.5)/1879 | **1.14% ≈ 1.1%** | 1.1% |
| Drift overhead | (full−no\_drift)/no\_drift | (2383−1879)/1879 | **26.8% ≈ 27%** | 27% |
| Drift resets ("false alarms") | full.drift\_resets | **44** | — | 44 |
| No-reset detector fires | no\_drift.drift\_resets | 36 | — | (not stated in paper) |

---

## §5.4 — Real-Data Evaluation (GitHub Actions)

Config: `experiments/configs/real_github_actions.json`  
Dataset: `data/raw/github_actions_real.csv` (600 rows, 2 projects)  
Seeds: 0–29 (n=30); bootstrap seed 42  
Result files: `experiments/results/real_github_actions/{0..29}/online_summary.json`

| Claim | Seeds | Key | Raw value(s) | Computed | Paper text |
|---|---|---|---|---|---|
| static\_rules cost | 0–29 (deterministic) | static\_rules.cumulative\_cost | 644.5 (all seeds identical) | — | 644.5 |
| heuristic\_score cost | 0–29 (deterministic) | heuristic\_score.cumulative\_cost | 860.0 (all seeds identical) | — | 860 |
| linucb cost | 0–29 (deterministic) | linucb.cumulative\_cost | 669.5 (all seeds identical) | — | 669.5 |
| linucb 3.8% worse than static | 0–29 | (669.5−644.5)/644.5 | **3.88% ≈ 3.8%** | 3.8% |
| thompson cost range | 0–29 | thompson.cumulative\_cost | 445.5–725.5 | — | 445.5–725.5 |
| Thompson mean | 0–29 | — | **624.2 ≈ 624** | 624 |
| Thompson std (sample, ddof=1) | 0–29 | — | **71.8 ≈ 72** | ±72 |
| Thompson 95% CI (bootstrap seed 42) | 0–29 | boot percentiles | **[598.2, 648.4] ≈ [598, 648]** | [598, 648] |
| Thompson vs LinUCB (6.8% cheaper) | 0–29 | (669.5−624.2)/669.5 | **6.77% ≈ 6.8%** | 6.8% |
| Thompson vs static (tied) | 0–29 | static=644.5 inside CI [598,648]? | **YES** | tied |
| Thompson p-value vs LinUCB | 0–29 | paired bootstrap, seed 42 | **p < 0.0001** | p < 0.01 |
| Thompson mean/step | 0–29 | 624.2/600 | **1.0403 ≈ 1.040** | 1.040 |
| Thompson std/step | 0–29 | 71.8/600 | **0.1197 ≈ 0.120** | ±0.120 |
| Thompson action fractions (mean) | 0–29 | per-file fracs | deploy=50.6%, canary=15.5%, block=33.9% | 50.6%/15.5%/33.9% |
| psf/requests failure rate | — | CSV: 16 failures / 300 rows | 5.33% | 5.3% |
| pallets/flask failure rate | — | CSV: 70 failures / 300 rows | 23.3% | 23.3% |
| Censored rewards | 0 | \*.total\_censored\_skipped | 17 (linucb\_with\_drift), 21 (static), 22 (heuristic) | — | 17–22 |
| ~~Thompson mean (n=5 pilot)~~ | ~~0–4~~ | ~~585.3~~ | ~~SUPERSEDED by n=30~~ | ~~retired~~ |

---

## §3.4 — Drift Detector Parameters

| Claim | Source |
|---|---|
| PageHinkley λ\_PH = 50 | `drift/detectors.py:72` — `PageHinkleyConfig(lambda_=50.0)` |
| ADWIN (not yet implemented) | `drift/detectors.py::ADWINDetector.update()` raises `NotImplementedError` |

---

## §6.4 — Cost-Ratio Sweep (F11)

Config: `experiments/run_cost_sweep.py` (5 inline CostLevel entries)  
Dataset: `data/raw/travistorrent_smoke.csv`  
Seeds: 0–29 (n=30); bootstrap seed 42  
Result file: `experiments/results/cost_sweep/cost_sweep_summary.json`

| Claim | Level | static\_rules | linucb | Δ | Paper text |
|---|---|---|---|---|---|
| 5:1 | deploy\_failure=5, block\_bad=1 | 1598 | 1486 | −7.0% | −7.0% |
| 10:1 | deploy\_failure=5, block\_bad=0.5 | 1535 | 1440 | −6.2% | −6.2% |
| 20:1 (default) | deploy\_failure=10, block\_bad=0.5 | 1878 | 1879 | +0.05% | tied |
| 40:1 | deploy\_failure=20, block\_bad=0.5 | 2564 | 2079 | −18.9% | −18.9% |
| 100:1 | deploy\_failure=50, block\_bad=0.5 | 4622 | 2330 | −49.6% | −49.6% |

---

## §6.4 — Drift-Mode Evaluation (F12)

Config: `experiments/run_drift_eval.py` (inline LOW\_RISK/HIGH\_RISK SegmentParams)  
Dataset: Synthetic (SyntheticEnvironment), horizon=500  
Seeds: 0–29 (n=30); bootstrap seed 42  
Result file: `experiments/results/drift_eval/drift_eval_summary.json`

| Claim | Drift mode | linucb mean | linucb\_with\_drift\_no\_reset mean | linucb\_with\_drift\_full mean | static mean | Paper text |
|---|---|---|---|---|---|---|
| Stationary | none | 536.9 | 536.9 | 581.9 | 540.1 | +8.4% overhead (linucb\_with\_drift\_full vs linucb) |
| Abrupt drift | abrupt | 602.7 | 602.7 | 768.5 | 572.5 | +27.5% overhead |
| Gradual drift | gradual | 685.8 | 685.8 | 874.1 | 651.1 | +27.4% overhead |
| linucb\_with\_drift\_full resets — none | — | — | — | 10.6 resets/traj | — | 10.6 |
| linucb\_with\_drift\_full resets — abrupt | — | — | — | 25.3 resets/traj | — | 25.3 |
| linucb\_with\_drift\_full resets — gradual | — | — | — | 42.2 resets/traj | — | 42.2 |

---

## Figure → Generator Mapping

All figures generated by `evaluation/plots.py::generate_all()`. Output: `paper/figures/<name>.pdf` and `.png`.

| Fig | Paper section | Generator function | Data source | Seeds |
|---|---|---|---|---|
| 1 | §5.1 above Table 1 | `fig_cumulative_cost_synthetic` | `experiments/results/online_smoke/{0..29}/online_summary.json` | 0–29 |
| 2 | §5.4 above Table 3 | `fig_cumulative_cost_real` | `experiments/results/real_github_actions/{0..29}/online_summary.json` | 0–29 |
| 3 | §5.4 above Table 3 | `fig_action_distribution` | both online\_smoke and real\_github\_actions summary JSONs | 0–29 |
| 4 | §6.4 F11 | `fig_cost_sweep` | `experiments/results/cost_sweep/cost_sweep_summary.json` | 0–29 |
| 5 | §6.4 F12 | `fig_drift_mode_bars` | `experiments/results/drift_eval/drift_eval_summary.json` | 0–29 |
| 6 | §5.3 above Table 2 | `fig_ablation_bars` | `experiments/results/ablation_smoke/0/ablation_summary.json` | 0 only |
| 7 | §5.4 above Table 3 | `fig_thompson_seed_distribution` | per-seed `thompson.cumulative_cost` from summary JSONs | 0–29 |
| 8 | §5.1 above Table 1 | `fig_cumulative_cost_curves` | `experiments/results/online_smoke/{seed}/step_costs_{pid}.npy` | 0–29 |
| 9 | §6.4 F12 | `fig_drift_recovery_curves` | `experiments/results/drift_eval/step_regrets_{mode}_{pid}.npy` | 0–29 (mean) |
| 10 | §5.4 above Table 3 | `fig_cost_cdf_per_step` | `experiments/results/real_github_actions/{seed}/step_costs_{pid}.npy` | 0–29 |

---

## Bootstrap / Statistical Parameters

| Claim | Source |
|---|---|
| 10 000 resamples | `evaluation/statistical.py::BootstrapConfig(n_resamples=10000)` |
| Seed 42 (robustness script) | `experiments/run_robustness.py:38` — `BOOTSTRAP_SEED = 42` |
| Seed 0 (BootstrapConfig default) | `evaluation/statistical.py::BootstrapConfig(seed=0)` — **discrepancy; run\_robustness.py overrides to 42** |

---

## Known Discrepancies / Open Items

| Item | Status |
|---|---|
| Bootstrap seed: paper says 42, default is 0 | `run_robustness.py` overrides to 42; paper claim correct for robustness section. Default seed=0 is unused for any reported number. |
| Ablation uses seed 0 only, not 5 seeds | Intentional (ablation is deterministic on this dataset; all seeds produce same values for non-Thompson policies). |
| Smoke dataset project-level breakdown (alpha=600/15%, beta=550/35%) | Not verified from CSV at time of audit; requires `data/raw/travistorrent_smoke.csv` inspection. |
| ADWIN implemented | `drift/detectors.py::ADWINDetector` fully implemented (Bifet & Gavaldà 2007, bucket compression). Not used in any reported experiment — PageHinkley is the paper's detector. |
| `run_drift_eval.py`, `run_cost_sweep.py` | Implemented and run (30 seeds each). Results in `experiments/results/drift_eval/` and `experiments/results/cost_sweep/`. F11 and F12 added to §6.4. |
