# Repo Audit for paper/adaptive-deployment-control.md

Generated: 2026-05-06  
Paper version commit: `9012cd3` (Reviewer polish: title, abstract, table captions, terminology)

---

## 1. Inventory Summary

| Paper Claim / Artifact | Paper Section | Backing Artifact | Status |
|---|---|---|---|
| **POLICIES** | | | |
| LinUCB — disjoint, 3-arm, r = −cost | §3.1, §3.5 | `policies/linucb.py` (127 lines) | **ok** |
| Thompson Sampling — Bayesian linear, Cholesky | §3.2, §3.5 | `policies/thompson.py` (137 lines) | **ok** |
| CostSensitiveBandit — LinUCB + delayed buffer + drift | §3.1, §3.4 | `policies/cost_sensitive_bandit.py` (230 lines) | **ok** |
| StaticRules policy | §3.5 | `policies/static_rules.py` | **ok** |
| HeuristicScore policy | §3.5 | `policies/heuristic_score.py` | **ok** |
| linucb and cost_sensitive_bandit identical at α=λ=1.0 (‖b-vector‖₂=0) | §3.5, §5.1 | Confirmed: both produce identical action_counts and cumulative_cost in all result files | **ok** |
| **FEATURE VECTOR** | | | |
| 12 features + bias = ℝ^13 | §2.2 | `policies/base.py::FeatureEncoder` (DIM=13) | **ok** |
| Feature order: files\_changed ÷50 at dim 0 | §2.2 | `policies/base.py:13–26, 94–109` | **ok** |
| All 12 normalisations match table in §2.2 | §2.2 | `policies/base.py:13–26` — normalisations match exactly | **ok** |
| Bias = 1.0 appended last (dim 12) | §2.2 | `policies/base.py:109` | **ok** |
| No post-deploy signal in feature vector | §2.2 | `policies/base.py:92–109` — only Context fields used | **ok** |
| **COST MATRIX** | | | |
| deploy\_failure=10, block\_bad=0.5 (20:1 ratio) | §2.3, Abstract | `rewards/cost_model.py:31–36` | **ok** |
| CostConfig frozen dataclass, all 7 fields | §2.3 | `rewards/cost_model.py:12–44` | **ok** |
| oracle\_cost() for regret computation | §2.4 | `rewards/cost_model.py:100–128` | **ok** |
| **DELAYED REWARD** | | | |
| PendingRewardBuffer — no update before maturity | §3.3 | `delayed/buffer.py::PendingRewardBuffer` | **ok** |
| Delay rule k\_t = max(1, ⌈duration/60⌉) | §2.1, §4.2 | `evaluation/online_replay.py:93–97` | **ok** |
| Censored rewards excluded from policy updates | §2.1 | `evaluation/online_replay.py:173–175`; `cost_sensitive_bandit.py:159–162` | **ok** |
| Buffer costs 1.1% vs. no-delay | §3.3, §5.3 | ablation results: no\_drift=1879.0, no\_delay=1857.5 → (1879−1857.5)/1879 = 1.14% | **ok** |
| **DRIFT** | | | |
| PageHinkley detector at λ\_PH=50 | §3.4 | `drift/detectors.py:72–76` (PageHinkleyConfig.lambda\_=50.0) | **ok** |
| 44 false alarms over 1,150 stationary steps | §3.4, §5.3 | `ablation_smoke/*/ablation_summary.json` — no `drift_resets` field stored | **missing** |
| Drift fires 44× → 27% more expensive than no-drift | §5.3 | full=2383, no\_drift=1879 → (2383−1879)/1879=26.8%≈27% ✓; but 44 count unverifiable | **partial** |
| Drift excluded from main claims | §3.4 | Stated in paper; `full` variant absent from main tables | **ok** |
| **EVALUATION** | | | |
| Online replay — counterfactual CI proxy | §4.2 | `evaluation/online_replay.py` (286 lines) | **ok** |
| All logged actions = DEPLOY (bias disclosed) | §4.2, §7.1 | `evaluation/online_replay.py` header; confirmed in results JSONs | **ok** |
| Bootstrap CI: 10,000 resamples, seed 42 | §4.3 | `experiments/run_robustness.py:38` (BOOTSTRAP\_SEED=42) ✓; but `evaluation/statistical.py::BootstrapConfig` default seed=0 | **partial** |
| Paired bootstrap p-value | §4.3 | `evaluation/statistical.py:52–75` | **ok** |
| Holm-Bonferroni correction | §4.3 | `evaluation/statistical.py:78–90` | **ok** |
| cumulative\_cost metric | §2.4 | `evaluation/metrics.py:32–39` | **ok** |
| cumulative\_regret metric | §2.4 | `evaluation/metrics.py:65–73` | **ok** |
| cost\_cdf metric | §2.4 | `evaluation/metrics.py:115–140` | **ok** |
| IPS/DR estimator (unbiased eval) | §4.2, §7.1 | `evaluation/replay_eval.py` — exists but NOT used for any reported number | **partial** |
| **NUMERICAL CLAIMS** | | | |
| 19% gain — high failure (df=20) | Abstract, §5.2 | robustness\_high\_failure/*/: static=2564, linucb=2079 → 485/2564=18.9%≈19% | **ok** |
| 27% gain — low block (bs=1) | Abstract, §5.2 | robustness\_low\_block/*/: static=1584, linucb=1163.5 → 420.5/1584=26.5%≈27% | **ok** |
| 31% degradation — binary reward | Abstract, §5.3 | ablation\_smoke/0/: no\_cost=2469, no\_drift=1879 → 590/1879=31.4% | **ok** |
| 3.8% worse — real GH-Actions | Abstract, §5.4 | real\_github\_actions/0/: linucb=669.5, static=644.5 → 25/644.5=3.88% | **ok** |
| 5.3% failure rate — psf/requests | §4.5, §5.4 | Confirmed by CSV: 16/300 = 5.33% | **ok** |
| 23.3% failure rate — pallets/flask | §4.5 | Confirmed by CSV: 70/300 = 23.3% | **ok** |
| Thompson synthetic: mean 1877 ± 58 | §5.1 | seeds 0–4: [1814.5,1970.5,1844.5,1937.5,1821.5] → mean=1877.7 ✓, pop\_std=63.9 / sample\_std=71.5. ±58 does **not** match either convention | **partial** |
| Thompson synthetic range 1814–1970 | §5.1 | min=1814.5, max=1970.5 ✓ | **ok** |
| Thompson real: mean 585 ± 99 | §5.4 | seeds 0–4: [445.5,520.0,649.5,631.5,680.0] → mean=585.3 ✓, sample\_std=98.8≈99 ✓ | **ok** |
| Thompson real seeds: 445.5/520.0/649.5/631.5/680.0 | §5.4 | result files confirmed exactly | **ok** |
| Main table — static=1878, linucb=1879, heuristic=2319 | §5.1 | online\_smoke/0/ confirmed | **ok** |
| Ablation: no\_delay=1857.5, full=2383 | §5.3 | ablation\_smoke/0/ confirmed | **ok** |
| Short delay: linucb=1849, saves 29 units | §5.2 | robustness\_short\_delay/0/: linucb=1849.5, static=1878 → diff=28.5≈29 | **ok** |
| Long delay: linucb=1897, costs 19 more | §5.2 | robustness\_long\_delay/0/: linucb=1896.5, static=1878 → diff=18.5≈19 | **ok** |
| Real data: static=644.5, heuristic=860, linucb=669.5 | §5.4 | real\_github\_actions/0/ confirmed | **ok** |
| Real data: 17–22 censored rewards per trajectory | §4.5, §5.4 | confirmed: cost\_sensitive\_bandit=17, static=21, heuristic=22 | **ok** |
| **DATASETS** | | | |
| smoke/alpha: 600 builds, 15% failure | §4.1 | data/raw/travistorrent\_smoke.csv (1150 data rows, 2 projects) | **needs review** |
| smoke/beta: 550 builds, 35% failure | §4.1 | Same file — project-level breakdown not verified from CSV | **needs review** |
| psf/requests: 300 runs | §4.5 | 300 rows confirmed in CSV | **ok** |
| pallets/flask: 300 runs | §4.5 | 300 rows confirmed in CSV | **ok** |
| data/raw/ gitignored | §4.1, Appendix B | `.gitignore:8` — confirmed gitignored; files present in worktree as local untracked files | **ok** |
| **EXPERIMENT CONFIGS** | | | |
| experiments/configs/online\_smoke.json | Appendix B | EXISTS; α=1.0, λ=1.0, delay=60s, default cost matrix | **ok** |
| experiments/configs/real\_github\_actions.json | Appendix B | EXISTS | **ok** |
| experiments/configs/robustness\_high\_failure.json | §4.3, Appendix B | EXISTS | **ok** |
| experiments/configs/robustness\_low\_block.json | §4.3, Appendix B | EXISTS | **ok** |
| experiments/configs/robustness\_short\_delay.json | §4.3, Appendix B | EXISTS | **ok** |
| experiments/configs/robustness\_long\_delay.json | §4.3, Appendix B | EXISTS | **ok** |
| Seeds 0–4 used for all experiments | §4.3, Appendix B | 5 seed subdirs per experiment confirmed in results/ | **ok** |
| **EXPERIMENT SCRIPTS** | | | |
| experiments/run\_bandits.py | Appendix B | EXISTS | **ok** |
| experiments/run\_robustness.py | Appendix B | EXISTS | **ok** |
| experiments/run\_ablations.py | Appendix B | EXISTS | **ok** |
| **TESTS** | | | |
| 180 passing tests | Appendix A | pytest: 180 passed, 4 skipped (confirmed 2026-05-06) | **ok** |
| tests/test\_environment.py — hidden-state property | CLAUDE.md | EXISTS | **ok** |
| tests/test\_policies.py | CLAUDE.md | EXISTS | **ok** |
| tests/test\_replay\_eval.py | CLAUDE.md | EXISTS | **ok** |
| tests/test\_delayed\_buffer.py | CLAUDE.md | EXISTS | **ok** |
| **DOCS** | | | |
| docs/problem-formulation.md | §2 | EXISTS ✓ | **ok** |
| docs/related-work.md | §1.2 | EXISTS; covers 8 sections | **ok** |
| docs/algorithm.md | §3 | EXISTS | **ok** |
| Appendix A validity table | Appendix A | EXISTS — in paper itself (not a separate file) | **ok** |
| paper/source-of-truth.md | CLAUDE.md §H8 | DOES NOT EXIST | **missing** |
| **REFERENCES** | | | |
| Agrawal & Goyal 2013 (Thompson Sampling) | §3.2, Refs | Cited in paper; NOT covered in docs/related-work.md | **partial** |
| Beller et al. 2017 (TravisTorrent) | §4.1, Refs | Cited in paper; NOT in docs/related-work.md | **partial** |
| Chu et al. 2011 | §1.3, Refs | In docs/related-work.md | **ok** |
| Gama et al. 2014 | §3.4, Refs | In docs/related-work.md | **ok** |
| Joulani et al. 2013 | §3.3, Refs | In docs/related-work.md (combined entry) | **ok** |
| Kamei et al. 2013 | §1.1, Refs | In docs/related-work.md | **ok** |
| Li et al. 2010 (LinUCB) | §3.1, Refs | In docs/related-work.md | **ok** |
| McIntosh & Kamei 2018 | §1.1, Refs | In docs/related-work.md | **ok** |
| Mouss et al. 2004 (Page-Hinkley) | §3.4, Refs | Cited in paper; NOT in docs/related-work.md | **partial** |
| Pike-Burke et al. 2018 | §5.2, Refs | In docs/related-work.md (combined entry) | **ok** |

---

## 2. Code Without Paper Claim (cleanup candidates)

| File | Reason not referenced |
|---|---|
| `policies/epsilon_greedy.py` | Not in §3.5 baselines table; paper's four-policy comparison uses static\_rules, heuristic\_score, linucb, thompson |
| `policies/offline_classifier.py` | Planned in CLAUDE.md Phase D6 but never run; not in paper |
| `policies/ablation_variants.py` | Used internally by `experiments/run_ablations.py` but not a public policy in the paper's taxonomy |
| `environment/synthetic.py` | Synthetic experiments use `data/raw/travistorrent_smoke.csv` via the loader, not the `environment/` abstraction |
| `environment/replay.py` | Real experiments use `evaluation/online_replay.py`, not this module |
| `environment/base.py` | Parent of the above; unused in paper experiments |
| `environment/delays.py` | Delay is computed inline in `online_replay.py:93–97`; this module is unused |
| `evaluation/replay_eval.py` (IPS) | Mentioned in §4.2 as the unbiased alternative but **no reported number uses it**; all numbers come from `online_replay.py` |
| `evaluation/plots.py` | All functions raise `NotImplementedError`; paper has no figures |
| `delayed/imputation.py` | Mentioned in CLAUDE.md Phase D2 but not referenced in paper |
| `drift/adapt.py` | Drift adaptation handled directly in `cost_sensitive_bandit.py:171–172`; this module not called |
| `knowledge_base/` (entire package: db.py, learning.py) | Not referenced in paper or experiments |
| `ingestion/github_client.py` | Data collected offline; not referenced in paper pipeline |
| `experiments/baseline.py` | Legacy script, predates pivot; not referenced in paper |
| `experiments/cost_analysis.py` | Legacy; not referenced in paper |
| `experiments/decision_eval.py` | Legacy; not referenced in paper |
| `experiments/evaluation.py` | Legacy; not referenced in paper |
| `experiments/feedback_loop_eval.py` | Legacy; not referenced in paper |
| `experiments/risk_eval.py` | Legacy; not referenced in paper |
| `experiments/sensitivity_analysis.py` | Legacy; not referenced in paper |
| `experiments/configs/first_real_result.json` | Not referenced in Appendix B or paper |
| `experiments/results/robustness/robustness_summary.json` | Different path from per-seed dirs; not cited in paper |
| `experiments/results/graphs/*.png` | Legacy graphs from old pipeline; not referenced in paper |

---

## 3. Paper Claim Without Code (submission blockers)

| Claim | Section | What Is Missing |
|---|---|---|
| "44 false alarms" for PageHinkley on 1,150 stationary steps | §3.4, §5.3 | `drift_resets` count is **not stored** in `ablation_smoke/*/ablation_summary.json`. The claim cannot be independently reproduced from the saved output. |
| Thompson synthetic std = ±58 | §5.1 | Actual population std = 63.9, sample std = 71.5 across seeds 0–4. Neither matches ±58. The synthetic table caption in §5.1 cites an incorrect std; the paper needs correction or an explanation of which convention produces 58. |
| Bootstrap CI: seed 42 | §4.3 | `evaluation/statistical.py::BootstrapConfig` defaults to `seed=0`. The seed=42 is defined in `run_robustness.py` (BOOTSTRAP\_SEED=42) but **not passed** to `BootstrapConfig` — `run_robustness.py:59,99` create `np.random.default_rng(BOOTSTRAP_SEED)` but use it directly, not via BootstrapConfig. Reproducibility of any bootstrap CI depends on which code path actually ran. Needs audit. |
| `paper/source-of-truth.md` | CLAUDE.md §H8 | File does not exist. No formal per-claim traceability document. |
| Ablation variant `no_drift` description in §3.4 and table | §5.3 | The `no_drift` variant has `reset_on_drift=False` in `run_ablations.py:77–81`. The paper describes it as "No drift reset" which is consistent. But the paper's ablation table column header "Drift Resets" shows "—" for `no_drift` and "44" for `full`. The 44 count is not verifiable from saved JSON. |
| Docs coverage: Agrawal & Goyal 2013, Beller 2017, Mouss 2004 | Refs | These three papers are cited in the paper but have no entries in `docs/related-work.md`. |
| Operating-envelope grid (§1.3 two-conditions claim) | §1.3 | §1.3 is the theoretical framing; the empirical characterization across failure-rate × feature-informativeness cells is only partially completed. The negative real-data finding (§5.4) covers one cell (low failure, sparse features). The positive synthetic finding covers one cell (high failure, informative features). No systematic grid exists yet. |
| Figures | — | `evaluation/plots.py` raises `NotImplementedError` for all functions. The paper currently has no figures, which is acceptable; but any revision adding figures has no generator. |

---

## 4. Statistical Rigor Status

| Headline Number | Seeds | CI present | p-value present | Multiple-comparison correction | Notes |
|---|---|---|---|---|---|
| 19% gain (high failure) | 5 (0–4) | No — CI width = 0 (deterministic policy) | No | No | Deterministic policies produce identical results across seeds; bootstrap CI is degenerate. Paper correctly labels as "zero CI width." |
| 27% gain (low block) | 5 (0–4) | No — CI width = 0 | No | No | Same as above. |
| 31% ablation degradation | 5 (0–4) | No — CI width = 0 | No | No | `no_cost` and `no_drift` are deterministic. |
| 3.8% worse (real data, LinUCB) | 5 (0–4) | No — CI width = 0 | No | No | LinUCB is deterministic; seed variation comes only from buffer RNG, which is fixed. |
| Thompson synthetic: 1877 ± 58 | 5 (0–4) | Reported as std, not CI | No | No | **Std value is wrong**: actual is ±63.9 (pop) or ±71.5 (sample). Must correct. |
| Thompson real: 585 ± 99 | 5 (0–4) | Reported as std, not CI | No | No | Std matches sample std (98.8≈99 ✓). |
| 44 drift false alarms | 1 (seed 0 only) | No | No | No | Count not stored in results. Single-seed claim. |

**Overall:** No pairwise p-values or multiple-comparison corrections are present anywhere in the reported results. The paper acknowledges this via its "Preliminary" validity labels in Appendix A. The bootstrap infrastructure (`evaluation/statistical.py`) is implemented and tested but not applied to any reported table.

---

## 5. Reproducibility Status

| Numerical Claim | Config | Seed | Result File | Traceable? |
|---|---|---|---|---|
| static=1878, linucb=1879 (main synthetic) | `experiments/configs/online_smoke.json` | 0 | `experiments/results/online_smoke/0/online_summary.json` | **yes** |
| Thompson synthetic 1877±58 | `experiments/configs/online_smoke.json` | 0–4 | `experiments/results/online_smoke/{0..4}/online_summary.json` | **yes** (but std value wrong) |
| linucb high-failure=2079, static=2564 | `experiments/configs/robustness_high_failure.json` | 0 | `experiments/results/robustness_high_failure/0/online_summary.json` | **yes** |
| linucb low-block=1163.5, static=1584 | `experiments/configs/robustness_low_block.json` | 0 | `experiments/results/robustness_low_block/0/online_summary.json` | **yes** |
| linucb short-delay=1849.5 | `experiments/configs/robustness_short_delay.json` | 0 | `experiments/results/robustness_short_delay/0/online_summary.json` | **yes** |
| linucb long-delay=1896.5 | `experiments/configs/robustness_long_delay.json` | 0 | `experiments/results/robustness_long_delay/0/online_summary.json` | **yes** |
| Ablation: no\_cost=2469, no\_drift=1879, no\_delay=1857.5, full=2383 | `ablation_smoke` (inferred from run\_ablations.py) | 0 | `experiments/results/ablation_smoke/0/ablation_summary.json` | **yes** |
| 44 drift false alarms | same ablation | 0 | **not in JSON** | **no** |
| Real data: static=644.5, linucb=669.5 | `experiments/configs/real_github_actions.json` | 0 | `experiments/results/real_github_actions/0/online_summary.json` | **yes** |
| Thompson real seeds: 445.5/520/649.5/631.5/680 | `experiments/configs/real_github_actions.json` | 0–4 | `experiments/results/real_github_actions/{0..4}/online_summary.json` | **yes** |
| 5.3% failure rate (psf/requests) | — | — | `data/raw/github_actions_real.csv` (gitignored; present locally) | **yes** (local only) |

**Note:** `paper/source-of-truth.md` does not exist. All traceability above was reconstructed manually during this audit. The absence of a traceability document means any future re-run or revision requires repeating this audit.

---

## 6. Recommended Next Steps

### Submission Blockers (must fix before submission)

1. **Fix Thompson synthetic std.** Paper §5.1 reports "1877 ± 58". Actual values are ±63.9 (population) or ±71.5 (sample). Determine which convention to use and correct the table. The real-data table uses sample std (98.8≈99); for consistency, use sample std throughout — which changes synthetic to ±72, not ±58.

2. **Add `drift_resets` to ablation output.** The claim "44 false alarms" is the primary evidence for §3.4 and the drift row in Table 2. It has no traceable source. Add `drift_resets` to `ablation_summary.json` output (one line in `run_ablations.py`) and regenerate.

3. **Create `paper/source-of-truth.md`.** Map every table cell to its `config + seed + result file`. This is required before any submission and takes ~1 hour to produce from the audit table above.

4. **Clarify bootstrap seed.** The paper says "seed 42" but `BootstrapConfig` defaults to seed=0. Verify which seed was actually used in run\_robustness.py and document explicitly. If seed 42 was used, update the BootstrapConfig default or document the override.

5. **Smoke dataset composition.** Paper claims smoke/alpha=600 builds at 15% failure, smoke/beta=550 at 35% failure. These counts were not verified from the CSV during this audit. Must confirm directly from `data/raw/travistorrent_smoke.csv` project-level row counts and actual failure rates.

### Rigor Gaps (should fix)

6. **Bootstrap CI widths are zero for deterministic policies.** All headline numbers for static\_rules, linucb, cost\_sensitive\_bandit have CI width = 0. The paper correctly labels these as "Preliminary — synthetic data. Deterministic policies have zero CI width." This is honest but weakens empirical credibility. No fix is possible without stochastic policies or a different evaluation protocol.

7. **No pairwise p-values.** `evaluation/statistical.py` implements `paired_bootstrap_pvalue` and `holm_bonferroni` but neither is called anywhere in the experiment scripts. Adding p-values between the key comparisons (static vs. linucb at high-failure; linucb vs. static on real data) would materially strengthen the results section.

8. **IPS is implemented but unused.** `evaluation/replay_eval.py` exists and is tested but is not used for any reported number. The paper acknowledges this ("IPS weight is identically 1.0, reducing the estimator to the direct method"). Acceptable as-is, but should be explicitly noted in §7 as a limitation, not just §4.2.

### Polish (nice to have)

9. **Add 3 missing references to docs/related-work.md.** Agrawal & Goyal 2013, Beller et al. 2017, Mouss et al. 2004 are cited in the paper but have no coverage entries.

10. **Clean up legacy experiment scripts.** `experiments/baseline.py`, `cost_analysis.py`, `decision_eval.py`, `evaluation.py`, `feedback_loop_eval.py`, `risk_eval.py`, `sensitivity_analysis.py` are unused by the paper pipeline and clutter the `experiments/` package.

11. **`evaluation/plots.py` is stub-only.** All functions raise `NotImplementedError`. Since the paper has no figures, this is not a blocker, but any future revision adding figures will need these implemented.

12. **Operating-envelope grid.** §1.3 promises empirical coverage of both conditions (high-failure/informative AND low-failure/sparse). Currently only two cells are populated. A fuller characterization (even on synthetic data) would substantially strengthen the paper's central claim.

---

## 7. Open Questions for Abi

1. **Thompson ±58**: Where does 58 come from? It matches neither population nor sample std across seeds 0–4. Was this computed from a different run, a different seed range, or an arithmetic error? Needs to be corrected before submission.

2. **44 drift false alarms**: Was this count read from a terminal output and manually entered into the paper, or computed from a saved result? If the former, re-run `experiments/run_ablations.py` with `drift_resets` logging added to the JSON output and re-verify.

3. **Bootstrap seed**: The paper says seed=42. `run_robustness.py` defines `BOOTSTRAP_SEED=42` but passes it to `np.random.default_rng()` directly, not to `BootstrapConfig`. Did the robustness CIs actually use seed=42? If so, `BootstrapConfig(seed=42)` should be passed explicitly.

4. **Smoke dataset breakdown**: Paper claims alpha=600 builds at 15% failure and beta=550 at 35% failure. The combined CSV has 1150 rows; the per-project split and actual failure rates should be verified from the raw file before the paper is finalized.

5. **Dataset scope decision** (from the re-alignment plan, Step 2): The paper currently uses 600 real GH-Actions runs across 2 repos. The Step 2 choice (case study vs. TravisTorrent generalization) determines whether §5.4 is retitled or replaced. This decision is pending Abi's input and affects everything from Step 3 onward.
