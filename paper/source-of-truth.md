# Source of Truth — Numerical Claims

Maps every numerical claim in `paper/adaptive-deployment-control.md` to the config,
seed(s), and result file that produced it. Update this file whenever a number changes.

Generated: 2026-05-06  
Paper commit: `9012cd3`  
Last updated for: fix-1 (Thompson ±58→±72), fix-2 (drift_resets now stored in ablation JSON)

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
Seeds: 0–4  
Result files: `experiments/results/online_smoke/{0..4}/online_summary.json`

| Claim | Seeds | Raw values | Computed | Paper text |
|---|---|---|---|---|
| static\_rules cost (seed 0) | 0 | 1878.0 | — | 1878 |
| linucb cost (seed 0) | 0 | 1879.0 | — | 1879 |
| heuristic\_score cost (seed 0) | 0 | 2319.0 | — | 2319 |
| thompson cost per seed | 0–4 | 1814.5, 1970.5, 1844.5, 1937.5, 1821.5 | — | 1814–1970 |
| Thompson mean | 0–4 | sum/5 | **1877.7 ≈ 1877** | 1877 |
| Thompson std (sample, ddof=1) | 0–4 | sample\_std | **71.5 ≈ 72** | ±72 |

---

## §5.2 — Robustness Analysis

All configs share `data/raw/travistorrent_smoke.csv`, delay=60 s, seeds 0–4.  
Note: all five seeds return identical values for robustness configs (deterministic under fixed dataset).

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
Seeds: 0–4  
Result files: `experiments/results/real_github_actions/{0..4}/online_summary.json`

| Claim | Seeds | Key | Raw value(s) | Computed | Paper text |
|---|---|---|---|---|---|
| static\_rules cost (seed 0) | 0 | static\_rules.cumulative\_cost | 644.5 | — | 644.5 |
| heuristic\_score cost (seed 0) | 0 | heuristic\_score.cumulative\_cost | 860.0 | — | 860 |
| linucb cost (seed 0) | 0 | linucb.cumulative\_cost | 669.5 | — | 669.5 |
| linucb 3.8% worse than static | 0 | (669.5−644.5)/644.5 | **3.88% ≈ 3.8%** | 3.8% |
| thompson cost per seed | 0–4 | thompson.cumulative\_cost | 445.5, 520.0, 649.5, 631.5, 680.0 | — | 445.5–680.0 |
| Thompson mean | 0–4 | — | **585.3 ≈ 585** | 585 |
| Thompson std (sample, ddof=1) | 0–4 | — | **98.8 ≈ 99** | ±99 |
| psf/requests failure rate | — | CSV: 16 failures / 300 rows | 5.33% | 5.3% |
| pallets/flask failure rate | — | CSV: 70 failures / 300 rows | 23.3% | 23.3% |
| Censored rewards | 0 | \*.total\_censored\_skipped | 17 (csb), 21 (static), 22 (heuristic) | — | 17–22 |

---

## §3.4 — Drift Detector Parameters

| Claim | Source |
|---|---|
| PageHinkley λ\_PH = 50 | `drift/detectors.py:72` — `PageHinkleyConfig(lambda_=50.0)` |
| ADWIN (not yet implemented) | `drift/detectors.py::ADWINDetector.update()` raises `NotImplementedError` |

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
| ADWIN not implemented | `drift/detectors.py::ADWINDetector` raises `NotImplementedError`. UNBLOCKED-5 will implement it. |
| `run_drift_eval.py`, `run_cost_sweep.py` | Not yet implemented. UNBLOCKED-4 will implement them. |
