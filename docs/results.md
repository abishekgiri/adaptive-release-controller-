# Phase 23 (Partial): Real-World Sanity Check — GitHub Actions CI Data

> **REAL CI DATA — BIASED EVALUATION.**
> Costs use `compute_cost(policy_action, CI_outcome)` as counterfactual proxy.
> Feature space is sparse (no commit-level metadata). Results are highly preliminary.

## What This Tests

Phase 23 real-data sanity check loads real GitHub Actions workflow run history from two
public repositories (`psf/requests`, `pallets/flask`) via GitHub REST API and runs the
full online-replay experiment pipeline with relaxed dataset filters.

**This is a pipeline validation, not a policy performance claim.**

## Dataset

| Project | Runs | Failure rate | Span | Feature richness |
| --- | ---: | ---: | ---: | --- |
| `psf/requests` | 300 | 5.3% | 20 days | Sparse — no commit-level metadata |
| `pallets/flask` | 300 | 23.3% | 99 days | Sparse — no commit-level metadata |

- File: `data/raw/github_actions_real.csv` (gitignored — not committed)
- Config: `experiments/configs/real_github_actions.json`
- Filters: `min_builds=100`, `min_history_days=0` (relaxed from standard 500/365)
- Censored rewards: 17–22 per trajectory from cancelled CI runs (real-world censoring)

## Exact Commands

```bash
for seed in 0 1 2 3 4; do
    python -m experiments.run_bandits \
        --config experiments/configs/real_github_actions.json \
        --seed $seed
done
```

## Results (seeds 0–4)

| Policy | Steps | Censored | Cumul. Cost | Mean/Step | Deploy% | Canary% | Block% |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `static_rules` | 600 | 21 | **644.5** | **1.113** | 64.3% | 20.0% | 15.7% |
| `heuristic_score` | 600 | 22 | 860.0 | 1.488 | 100.0% | 0.0% | 0.0% |
| `linucb` | 600 | 17 | 669.5 | 1.148 | 53.0% | 4.8% | 42.2% |
| `cost_sensitive_bandit` | 600 | 17 | 669.5 | 1.148 | 53.0% | 4.8% | 42.2% |
| `thompson` (mean ± std, n=5) | 600 | 17–21 | **585 ± 99** | 1.008 ± 0.171 | varies | varies | varies |

Thompson seed breakdown: 445.5 / 520.0 / 649.5 / 631.5 / 680.0 (seeds 0–4)

## Expected Cost Analysis (static per-project optima)

| Project | E[deploy] | E[canary] | E[block] | Optimal |
| --- | ---: | ---: | ---: | --- |
| `psf/requests` (5.3% failure) | **0.53** | 1.16 | 1.92 | deploy |
| `pallets/flask` (23.3% failure) | 2.33 | 1.70 | **1.65** | block |

## Interpretation

**LinUCB over-blocks the low-failure project.** At 5.3% failure, deploying costs 0.53/step
while blocking costs 1.92/step. LinUCB converges to 42.2% block across both projects,
paying unnecessary block_safe costs on `psf/requests`. Static rules' 64% deploy rate is
closer to optimal, giving static rules a 3.8% cost advantage over LinUCB (644.5 vs 669.5).

**Thompson has high variance on short trajectories.** With only 300 steps per project,
Thompson's posterior has not converged. Seed 0 discovers the deploy-heavy strategy and
achieves 445.5 (best of all policies). Seed 4 gets stuck block-heavy and achieves 680.0
(worst bandit). The mean (585) beats static rules on average but not reliably.

**Real censoring observed.** 17–22 cancelled CI runs per trajectory produce genuine
censored rewards. Censoring rate differs by policy (different actions correlate with
CI run completion). This is a confounder our evaluation does not model.

**What this does NOT establish:**
- That static rules are generally better than bandits (this is a feature-sparse, short-trajectory experiment)
- That Thompson Sampling is generally better (high variance, 5 seeds)
- Any causal claim about real deployment cost

## Critical Limitations

1. **Feature space is sparse**: No commit-level metadata (files changed, lines added, author info).
   The bandit learns primarily from recent-failure-rate and bias term. This is not a fair test
   of the algorithm's full capability.
2. **Short trajectories**: 300 steps per project. Standard bandit theory requires O(d²) steps
   to learn d-dimensional linear model. With d=13, ~170 steps minimum — marginal here.
3. **Single run per project**: No cross-project variance, no bootstrap CI width for deterministic policies.
4. **Relaxed filters**: `min_builds=100`, `min_history_days=0` — not comparable to standard config.
5. **Data source**: GitHub Actions API (not TravisTorrent). Schema mapping is approximate.

---

# Phase 22: Ablation Study — Cost-Sensitive Bandit Components

> **SIMULATION — NOT CAUSAL INFERENCE.**
> All costs use `compute_cost(policy_action, logged_CI_outcome)` as a counterfactual proxy.

## What This Tests

Phase 22 isolates which components of the `CostSensitiveBandit` contribute to cost reduction
on the synthetic smoke dataset (1150 steps across 2 projects). Four variants are compared:

| Variant | Delay buffer | Reward signal | Drift reset | Implementation |
| --- | --- | --- | --- | --- |
| `full` | Enabled | −cost (asymmetric) | PageHinkley reset | `CostSensitiveBandit` + `PageHinkleyDetector` |
| `no_delay` | **Disabled** | −cost | — | `ImmediateLinUCB` (no buffer) |
| `no_cost` | Enabled | −1/0 (binary) | — | `BinaryRewardBandit` |
| `no_drift` | Enabled | −cost | **Disabled** | `CostSensitiveBandit(reset_on_drift=False)` |

## Exact Commands

```bash
# Seeds 0–4 (deterministic under smoke data)
for seed in 0 1 2 3 4; do
    python -m experiments.run_ablations --seed $seed
done
```

## Ablation Results (all seeds identical — deterministic delays)

| Variant | Steps | Updates | Censored | Cumul. Cost | Mean/Step | Deploy% | Canary% | Block% |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `full` | 1150 | 1150 | 0 | 2383.00 | 2.072 | 40.4% | 30.1% | 29.5% |
| `no_delay` | 1150 | 1150 | 0 | **1857.50** | **1.615** | 1.7% | 19.8% | 78.4% |
| `no_cost` | 1150 | 1150 | 0 | 2469.00 | 2.147 | 41.4% | 22.1% | 36.5% |
| `no_drift` | 1150 | 1150 | 0 | 1879.00 | 1.634 | 8.0% | 5.7% | 86.3% |

## Component Analysis

### Delay handling (full vs no_delay)

`no_delay` has the **lowest cost (1857.50)** and the highest block rate (78.4%). With immediate
feedback, the policy receives 1150 updates spread evenly across steps and converges to the
blocking strategy faster. The delayed variant (`no_drift`) receives the same 1150 updates but
they arrive after a lag of `max(1, ceil(build_duration_s / 60))` steps, so early steps are
guided by an uninformed prior.

**Preliminary finding (smoke data only):** Removing delay improves cost by ~21.5 points (1.1%)
relative to `no_drift` and by ~525.5 points relative to `full`. The delay component alone costs
nothing — the loss in `full` vs `no_delay` is primarily from drift false alarms (see §Drift).

### Cost weighting (no_drift vs no_cost)

`no_cost` uses binary reward (−1 on failure, 0 otherwise) instead of `r = −cost`. This is the
**worst variant (2469 cost)**. Without cost weights, the policy cannot distinguish:
- `deploy + failure` → cost 10 vs `block + safe` → cost 2 (factor 5×)

The binary signal treats all failures equally regardless of which action caused them. The
block arm's b-vector gets −1 updates for all failures (including failures it correctly
blocked), which is the same signal DEPLOY would get on a failure it caused — providing
misleading gradient information.

**Finding: Cost weighting is the single most important component** on the smoke dataset.
Removing it adds 590 cost relative to `no_drift` (+31.4%). Removing the delay buffer adds
only 22 cost relative to `no_drift` (the opposite direction, actually —  delay adds cost by
slowing convergence, so `no_delay` is faster).

### Drift adaptation (full vs no_drift)

`full` uses `PageHinkleyDetector(lambda_=50.0)` with `reset_on_drift=True`. Diagnostic:

| Project | Drift resets | Total updates |
| --- | ---: | ---: |
| `smoke/alpha` | 17 | 600 |
| `smoke/beta` | 27 | 550 |

**44 false-alarm resets on stationary data.** The smoke dataset has no true distribution
shift; every PageHinkley signal is a false alarm. Each reset discards all learned weights
and forces the model to rediscover the block-heavy strategy from scratch. This explains why
`full` has 40.4% deploy (uninformed post-reset priors) vs `no_drift` at 8.0%.

**Finding: On stationary data, drift detection hurts.** `no_drift` costs 1879 vs `full` 2383
(−504, −21.2%). This is expected behavior: drift detectors are designed to pay a false-alarm
cost in exchange for faster recovery after real shifts. The ablation shows the detector is too
sensitive for the smooth cost distribution in this dataset; `lambda_=50.0` needs tuning.

**On data with real concept drift (e.g. abrupt build failure rate change), `full` should
outperform `no_drift`.** This is the synthetic environment experiment (RQ4), not yet run.

## Reliability Assessment

| Finding | Status |
| --- | --- |
| Cost weighting is critical | **Preliminary** — smoke data only, 2 projects |
| Delay slows convergence but does not change conclusions | **Preliminary** |
| PageHinkley fires false alarms on stationary data | **Reliable** — this is expected detector behavior |
| Drift reset hurt on smooth stationary data | **Reliable** on this dataset |
| Full model outperforms no_drift on drifting data | **TODO** — requires synthetic drift environment (RQ4) |

---

# Phase 21: Robustness Analysis

> **SIMULATION — NOT CAUSAL INFERENCE.**
> All CIs are computed via percentile bootstrap over seeds; on the smoke dataset, all seeds
> produce identical results (deterministic delay model), so CI widths are zero.
> Bootstrap machinery is verified correct — variance will emerge with real stochastic data.

## Conditions

| Condition | Change from default | Deploy failure | Delay step (s) |
| --- | --- | ---: | ---: |
| `online_smoke` (default) | — | 10.0 | 60 |
| `robustness_high_failure` | deploy_failure↑ × 2, canary_failure↑ × 2 | 20.0 | 60 |
| `robustness_low_block` | block_safe↓ 2→1, block_unknown↓ 2→1 | 10.0 | 60 |
| `robustness_short_delay` | delay_step_seconds↑ 60→120 (halves step-count) | 10.0 | 120 |
| `robustness_long_delay` | delay_step_seconds↓ 60→30 (doubles step-count) | 10.0 | 30 |

## Exact Commands

```bash
python -m experiments.run_robustness \
    --configs experiments/configs/online_smoke.json \
              experiments/configs/robustness_high_failure.json \
              experiments/configs/robustness_low_block.json \
              experiments/configs/robustness_short_delay.json \
              experiments/configs/robustness_long_delay.json \
    --seeds 0 1 2 3 4 \
    --results-root experiments/results/robustness
```

## Results Table: Policy × Condition × Mean Cost ± Bootstrap CI

> CI shown as [lo, hi]; zero-width CIs confirm deterministic delay model, not a code bug.

### Default cost matrix (deploy_failure=10)

| Policy | Default | Short delay | Long delay |
| --- | ---: | ---: | ---: |
| `static_rules` | 1878 [1878, 1878] | 1878 [1878, 1878] | 1878 [1878, 1878] |
| `heuristic_score` | 2319 [2319, 2319] | 2319 [2319, 2319] | 2319 [2319, 2319] |
| `linucb` | 1879 [1879, 1879] | **1849** [1849, 1849] | 1897 [1897, 1897] |
| `cost_sensitive_bandit` | 1879 [1879, 1879] | **1849** [1849, 1849] | 1897 [1897, 1897] |

### Alternative cost matrices (default delay, deploy_failure varies)

| Policy | Default (df=10) | High failure (df=20) | Low block (bs=1) |
| --- | ---: | ---: | ---: |
| `static_rules` | 1878 | 2564 | 1584 |
| `heuristic_score` | 2319 | 4103 | 2319 |
| `linucb` | 1879 | **2079** | **1164** |
| `cost_sensitive_bandit` | 1879 | **2079** | **1164** |

## Interpretation

### Do conclusions survive cost-matrix changes?

**Yes, with one nuance.** Under the default cost matrix, `linucb` and `static_rules` are
essentially tied (1879 vs 1878 — a difference of 1 cost unit across 1150 steps). This tie
reflects that static rules' canary-heavy strategy is near-optimal at the smoke data's
average failure rate (~25%).

Under the **high failure cost** matrix (deploy_failure=20): bandits win by **485 cost (19%)**
because the learned block-heavy strategy (92.7% block) avoids expensive failures. Static rules
continues to canary 60.9% of commits, paying the doubled failure penalty.

Under the **low block penalty** matrix (block_safe=1): bandits win by **420 cost (27%)**
because blocking is now cheaper (block_safe=1 vs 2). The bandits shift further into blocking
(89.3% block), while static rules cannot adapt its fixed thresholds.

**Conclusion: bandit advantage increases as blocking becomes relatively cheaper or failures
become more expensive.** This is the correct qualitative behavior from a cost-minimizing learner.

### Do conclusions survive delay changes?

**Partially.** The bandit ranking changes at extreme delay settings:

- **Short delay** (`delay_step_seconds=120`): each build "completes" in half as many steps.
  Bandits receive feedback twice as fast in step-count terms, converge to a block-heavy strategy
  (69.1% block) faster, and beat static rules by 29 cost units.

- **Long delay** (`delay_step_seconds=30`): each build occupies twice as many steps. The bandit
  receives rewards further into the future and the initial uninformed-prior period is longer.
  It converges to a less decisive strategy (62.3% block vs 69.1% in short delay), and costs
  1897 — 19 more than static rules (1878).

**Finding: under heavy delay (many steps per build), the static rule marginally outperforms
a learning bandit.** The effect is small (1.0% more cost) and may not survive with real
multi-project data, but it is consistent with the expected delay-sensitivity of UCB algorithms.

### Is policy ranking stable?

| Ranking | Default | High failure | Low block | Short delay | Long delay |
| --- | --- | --- | --- | --- | --- |
| Best bandit vs static | Tied (Δ=1) | Bandit wins (Δ=485) | Bandit wins (Δ=420) | Bandit wins (Δ=29) | Static wins (Δ=19) |
| Heuristic score | Always worst | Always worst | Always worst | Always worst | Always worst |

**Heuristic score is consistently worst** across all conditions because it never blocks (0%
block rate) and deploys 37% of commits, absorbing full failure costs.

## Reliability Assessment

| Finding | Confidence | Caveat |
| --- | --- | --- |
| Bandit advantage grows with failure cost | **Preliminary** | 2 projects, synthetic data |
| Bandit advantage grows with lower block cost | **Preliminary** | Same caveat |
| Long delay hurts bandit convergence | **Preliminary** | Effect is small (1%) |
| Bootstrap CI machinery is correct | **Reliable** | Unit-tested |
| Zero-width CI is expected on this dataset | **Reliable** | Deterministic delay confirmed |
| All results need real multi-project replication | Required | — |

---

# Phase 20: Online Replay Experiment Results (Corrected)

> **SIMULATION — NOT CAUSAL INFERENCE.**
> Costs computed as `compute_cost(policy_action, logged_outcome)` using the CI
> outcome as a counterfactual proxy. Valid only under the assumption that CI outcome
> is independent of the deployment action chosen. Do not report as causal estimates
> of real-world operational cost.

---

## Experiment Setup

| Item | Value |
| --- | --- |
| Config file | `experiments/configs/online_smoke.json` |
| Evaluation mode | Online replay simulation (`evaluation/online_replay.py`) |
| Dataset | `data/raw/travistorrent_smoke.csv` — synthetic, TravisTorrent-format CSV |
| Dataset note | Real TravisTorrent CSV not yet present; smoke CSV was generated to match schema |
| Dataset rows | 1150 across 2 synthetic projects |
| Projects / trajectories | `smoke/alpha` (600 builds, 15% failure); `smoke/beta` (550 builds, 35% failure) |
| Project history span | Both span > 365 days (min_history_days filter passed) |
| Min builds filter | 500 (both projects pass) |
| Seeds | 0, 1, 2, 3, 4 (all produce identical results — see §Seed Variance) |
| Delay model | `delay_steps = max(1, ceil(build_duration_s / 60))` — deterministic from CSV |
| Cost matrix | Default `CostConfig` (deploy_failure=10, canary_failure=4, block_safe=2, block_bad=0.5) |
| LinUCB α | 1.0, λ=1.0 |
| CostSensitiveBandit α | 1.0, λ=1.0 (no drift detector in this run) |

### Exact Commands

```bash
# Run online replay for seeds 0–4
for seed in 0 1 2 3 4; do
    python -m experiments.run_bandits \
        --config experiments/configs/online_smoke.json \
        --seed $seed
done
```

Output written to `experiments/results/online_smoke/<seed>/` (gitignored).

---

## Aggregate Results (seeds 0–4, all identical — see §Seed Variance)

| Policy | Steps | Updates applied | Censored skipped | Cumul. cost | Mean cost/step | Deploy% | Canary% | Block% |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `static_rules` | 1150 | 1150 | 0 | **1878.0** | 1.633 | 2.6% | 60.9% | 36.5% |
| `heuristic_score` | 1150 | 1150 | 0 | 2319.0 | 2.017 | 37.3% | 62.7% | 0.0% |
| `linucb` (α=1.0) | 1150 | 1150 | 0 | 1879.0 | 1.634 | 8.0% | 5.7% | 86.3% |
| `cost_sensitive_bandit` (α=1.0) | 1150 | 1150 | 0 | 1879.0 | 1.634 | 8.0% | 5.7% | 86.3% |

All 1150 rewards matured and were applied (`flush_at_end=True`). Zero censored outcomes
in the smoke dataset (all rows have a definite pass/fail status).

---

## Per-Project Breakdown

| Project | Failure rate | Policy | Cumul. cost | Updates | Deploy | Canary | Block |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| `smoke/alpha` | 15% | `static_rules` | 911.5 | 600 | 21 | 473 | 106 |
| `smoke/alpha` | 15% | `linucb` | 982.0 | 600 | 66 | 50 | 484 |
| `smoke/alpha` | 15% | `cost_sensitive_bandit` | 982.0 | 600 | 66 | 50 | 484 |
| `smoke/beta` | 35% | `static_rules` | 966.5 | 550 | 9 | 227 | 314 |
| `smoke/beta` | 35% | `linucb` | 897.0 | 550 | 26 | 15 | 509 |
| `smoke/beta` | 35% | `cost_sensitive_bandit` | 897.0 | 550 | 26 | 15 | 509 |

---

## Evidence That Learning Is Occurring

The step trace below (first 20 steps of `smoke/alpha`, LinUCB, seed 0) confirms the policy
changes its action as delayed rewards arrive. It does not simply deploy everywhere.

```
step=  0  DEPLOY   outcome=success  cost=0.0   updates_applied=0  pending_before=0
step=  1  DEPLOY   outcome=failure  cost=10.0  updates_applied=0  pending_before=1
step=  2  CANARY   outcome=success  cost=1.0   updates_applied=1  pending_before=2
step=  3  CANARY   outcome=success  cost=1.0   updates_applied=0  pending_before=2
...
step= 10  BLOCK    outcome=failure  cost=0.5   updates_applied=1  pending_before=9
step= 11  BLOCK    outcome=success  cost=2.0   updates_applied=0  pending_before=9
step= 14  BLOCK    outcome=success  cost=2.0   updates_applied=1  pending_before=11
```

Key observations:
- Step 0: uninformed prior → DEPLOY (maximum UCB on fresh λI matrix)
- Step 1: DEPLOY again (step 0's reward still in buffer, not yet matured)
- Step 2: first reward matures (`updates_applied=1`) → policy shifts toward CANARY
  (the step 0 SUCCESS with cost=0 made deploy look good, but step 1 FAILURE with
  cost=10 created a strong negative b-vector update when it arrived at step ≥2)
- Steps 10+: policy converges to BLOCK-heavy strategy after observing failure costs

The `pending_count_before` column confirms the buffer is accumulating correctly:
delays vary per record (based on `build_duration_s`), and rewards arrive
asynchronously.

---

## Did LinUCB and Cost-Sensitive-Bandit Diverge?

**With identical hyperparameters (α=1.0, λ=1.0): NO — they are mathematically equivalent.**

Both implement the disjoint LinUCB update rule:
```
A_a  +=  x xᵀ
b_a  +=  r x    where r = −cost
UCB  =  θᵀx  +  α √(xᵀ A⁻¹ x)
```

With the same α, λ, and reward signal, the b-vectors are identical to machine
precision after the same update sequence. `‖b_linucb − b_bandit‖₂ = 0.000000`
for all three arms.

**With different α: they diverge significantly.**

| Policy | α | Cumul. cost (smoke/alpha) | Deploy | Canary | Block |
| --- | ---: | ---: | ---: | ---: | ---: |
| `linucb` | 0.0 (exploit) | 963.5 | 43 | 421 | 136 |
| `cost_sensitive_bandit` | 5.0 (explore) | 999.5 | 120 | 275 | 205 |

Actions differ on **309 of 600 steps (51.5%)** — a clear divergence driven by
exploration vs exploitation. This confirms the divergence mechanism works; the
default smoke config intentionally uses identical hyperparameters.

---

## Seed Variance

Results are **identical across all 5 seeds** (0–4). This is expected: the online replay
loop uses explicit delays derived from `build_duration_s` (deterministic from the CSV),
bypassing the buffer's RNG sampler entirely. The master `rng` seed is consumed only to
draw per-project seeds, which then feed the buffer's `PendingRewardBuffer(rng=...)`.
Since no stochastic delay sampling occurs, the seed has no effect on outcomes.

**Implication:** Variance across seeds will only appear when using stochastic delays
(`delay_p` parameter on `PendingRewardBuffer`) or when the policy itself uses RNG
(e.g. Thompson sampling). The current deterministic delay model produces zero
Monte Carlo variance — this is a feature for reproducibility but means bootstrap
CIs across seeds are uninformative (all collapsed to a single value).

---

## Interpretation

### What learned differently from what

`linucb` and `cost_sensitive_bandit` both converged to a **block-heavy strategy (86%
BLOCK)** after learning. This is the correct response to the cost matrix: with even
modest failure rates, blocking (`cost=0.5 on failure, 2.0 on success`) is cheaper than
deploying (`cost=10.0 on failure`) or canarying (`cost=4.0 on failure`).

On `smoke/alpha` (15% failure): `static_rules` is cheaper (911.5 vs 982.0) because its
canary-heavy strategy (79% canary) exploits the low failure rate — at 15% failure,
`canary_expected = 0.85×1.0 + 0.15×4.0 = 1.45/step` beats `block_expected =
0.85×2.0 + 0.15×0.5 = 1.775/step`. LinUCB over-blocks on the low-failure project.

On `smoke/beta` (35% failure): `linucb` is cheaper (897.0 vs 966.5) because its
block-heavy strategy pays off — at 35% failure, `block_expected = 0.65×2.0 +
0.35×0.5 = 1.475/step` beats `canary_expected = 0.65×1.0 + 0.35×4.0 = 2.05/step`.

**Aggregate totals are essentially tied** (1878 vs 1879) because the two projects
cancel out. This is a smoke-data artifact.

`heuristic_score` is the most expensive policy (2319.0, mean 2.017/step) because it
never BLOCKs (BLOCK% = 0%) — its score thresholds never trigger BLOCK on the
smoke contexts. It deploys 37% of commits, absorbing high failure costs.

### What this does NOT show

- Whether the bandit converges faster or slower than static rules
- Whether the bandit would outperform static rules on the real multi-project TravisTorrent dataset
- Any causal claim about real-world deployment outcomes

---

## Delayed Buffer Diagnostics

| Metric | Value |
| --- | --- |
| Total rewards queued | 1150 (one per step) |
| Total rewards applied (`flush_at_end=True`) | 1150 |
| Censored rewards skipped | 0 |
| Peak pending count (observed) | ~14 (smoke/alpha, early steps) |
| Delay range (smoke data) | 1–60 steps (30–3600s builds ÷ 60s/step) |
| Stochastic delay used | No (explicit delay from `build_duration_s`) |

All 1150 rewards were applied before result collection. The buffer correctly held
rewards until their scheduled step and released them in chronological order.

---

## Limitations

| Limitation | Impact | Mitigation needed |
| --- | --- | --- |
| Synthetic smoke dataset | Results not representative of real CI/CD projects | Replace with real TravisTorrent CSV |
| Two projects only | No cross-project uncertainty estimate | Need 10+ projects for meaningful CIs |
| Identical seeds produce identical results | Bootstrap CIs degenerate | Use stochastic delays (`delay_p`) for variance |
| LinUCB == CostSensitiveBandit at same α | Cannot distinguish policies without different hyperparams or drift detection | Enable drift detector or sweep α |
| Counterfactual proxy validity | BLOCK action cost is estimated from CI outcome the policy never ran | Acknowledge as assumption; use synthetic env for ground truth |
| No online training on real dataset | Policies reset per project; no warm-start cross-project | Expected; document as design choice |
| `heuristic_score` never BLOCKs | Score formula needs tuning for smoke context range | Check threshold calibration |

---

# Phase 18: Results Validation and Credibility Audit

> **Status:** Audit complete. No paper claims are currently valid for submission.
> See §What Is Reliable vs Not Yet Reliable for what the pipeline *does* establish.

---

## 1. IPS Validity

### Finding: IPS estimates are degenerate — treat as direct-method estimates

**Logged propensities are absent from TravisTorrent.** The dataset records CI build outcomes,
not the propensities of any deployment policy. The experiment pipeline sets
`logged_propensity = 1.0` for every step (see `ExperimentConfig.logged_propensity` in
`experiments/run_baselines.py`).

With constant propensity = 1.0, the IPS weight for every matched action is:

```
w_t = 1 / logged_propensity = 1 / 1.0 = 1.0
```

The IPS estimator reduces to the **direct method**:

```
IPS-cost ≈ (1/T) * Σ_{t: a_logged = a_candidate} cost_t
```

This is only unbiased if the logging policy was deterministic with propensity 1.0 — an assumption
that cannot be verified and is almost certainly false for a dataset of human-driven CI runs.

**Consequence:** The "IPS estimated policy value" column in all result tables is correctly
computed *given the assumption*, but the assumption is unverifiable. The estimator is biased
by an unknown amount in an unknown direction.

**Mitigation applied (Phase 18):** Added explicit documentation of this assumption in
`experiments/run_baselines.py`. Results tables must carry the footnote:

> *IPS propensity set to 1.0 (unknown logging policy). Estimates are equivalent to the
> direct method. Counterfactual correction is not applied.*

**What is needed before publication:** Either (a) collect data from a known stochastic logging
policy, or (b) switch the primary metric from IPS-estimated value to direct online cost on
the synthetic environment where the true logging propensity is known.

---

## 2. Replay Bias

### Finding: Every logged action is DEPLOY — CANARY and BLOCK have zero coverage

`data/loaders.py`, line 165:

```python
yield TravisTorrentRecord(
    context=context,
    action=Action.DEPLOY,   # ← hardcoded; not inferred from data
    outcome=row.outcome,
    ...
)
```

**TravisTorrent records CI build outcomes, not deployment decisions.** Each row represents a
commit that was built and tested. The loader maps every row to `Action.DEPLOY` because the act
of running CI implies "someone tried to deploy (or at least integrate) this commit."

This creates a fundamental coverage problem:

| Action | Logged count (smoke run, 600 rows) | Fraction |
| --- | ---: | ---: |
| DEPLOY | 600 | 100% |
| CANARY | 0 | 0% |
| BLOCK | 0 | 0% |

**Consequences:**

1. **Any policy that selects CANARY or BLOCK for a context contributes nothing to the cost
   estimate for that step** (IPS weight = 0; no match). The IPS estimate only uses steps where
   the candidate policy agrees to DEPLOY.

2. **Static rules appears cheaper than LinUCB/cost-sensitive-bandit not because its deploys
   are cheaper, but because it deploys less often:**

   | Policy | Matched steps | Cumulative cost | Cost per matched step |
   | --- | ---: | ---: | ---: |
   | `static_rules` | 397 / 600 | 600.0 | 1.511 |
   | `linucb` | 600 / 600 | 980.0 | 1.633 |
   | `cost_sensitive_bandit` | 600 / 600 | 980.0 | 1.633 |

   Static rules blocks/canaries ~34% of commits; those steps are simply excluded from its cost
   estimate. The resulting lower total is an artifact of **reduced coverage, not better decisions**.

3. **LinUCB and cost-sensitive bandit produce identical results** because neither has learned
   anything. In the replay, policies are evaluated from their initial model state without
   receiving any training updates. Both fresh bandit models default to `Action.DEPLOY`
   for every context (maximum UCB on an uninformed prior), so they match every logged DEPLOY
   step and receive the full trajectory cost.

**Conclusion:** The real-data replay as currently structured cannot distinguish between a
good deployment policy and a policy that blindly deploys everything. The comparison between
static rules and bandits is **not interpretable** as evidence for or against the research claim.

---

## 3. Effective Sample Size

### Finding: ESS is determined by action coverage, not reward quality

With `logged_propensity = 1.0` and binary action matching (weight = 1 if match, 0 if not):

```
ESS = (Σ w_t)² / Σ w_t² = (matched_actions)² / matched_actions = matched_actions
```

ESS collapses exactly to matched action count. The standard diagnostic interpretation ("ESS
< n/10 signals high variance") is invalid here because variance comes from action coverage,
not from extreme importance weights.

| Policy | Matched actions | ESS | ESS / total steps |
| --- | ---: | ---: | ---: |
| `static_rules` | 397 | 397.0 | 66.2% |
| `linucb` | 600 | 600.0 | 100.0% |
| `cost_sensitive_bandit` | 600 | 600.0 | 100.0% |

100% ESS for bandits looks excellent but is an artifact: the policies deploy on every step
because they haven't learned, not because they have low variance estimates.

**ESS is not a useful diagnostic for this evaluation setup.** It should only be reported
alongside matched-action count and interpreted as coverage, not weight quality.

---

## 4. Cost Sensitivity

### Finding: Only one cost matrix has been evaluated; sensitivity is unknown

The evaluation protocol (`docs/evaluation-protocol.md`) specifies four cost matrix
configurations for RQ2:

| Matrix ID | Key difference | Status |
| --- | --- | --- |
| `default` | `deploy_failure=10, block_safe=2` | ✓ Smoke run complete |
| `low-block-penalty` | `block_safe=1, block_unknown=1` | ✗ Not run |
| `high-failure-penalty` | `deploy_failure=20, canary_failure=8` | ✗ Not run |
| `high-canary-overhead` | `canary_success=3, canary_failure=6` | ✗ Not run |

**Current numbers cannot claim robustness.** A policy that "wins" under the default cost matrix
may lose under high-canary-overhead if it over-routes to CANARY.

**Two concrete cost sensitivity checks to run before publication:**

1. **Symmetric cost:** set `deploy_failure = block_safe = canary_success = 2.0` — if ordering
   changes, conclusions are cost-dependent.

2. **Zero block cost:** set `block_safe = 0, block_bad = 0` — a policy that always blocks has
   zero cost. If any policy approaches this, the comparison is trivially driven by blocking rate
   rather than decision quality.

---

## 5. Delay Sensitivity

### Finding: No delay sensitivity experiments have been run

The smoke run used `delay_steps = max(1, ceil(tr_duration_s / 60))` derived from TravisTorrent
build durations. No synthetic delay ablation has been run.

The evaluation protocol specifies three delay conditions (RQ3):

| Experiment | Delay model | Status |
| --- | --- | --- |
| `RQ3-delay-none` | Fixed delay = 1 | ✗ Not run |
| `RQ3-delay-moderate` | Geom(p=0.30), max 20 | ✗ Not run |
| `RQ3-delay-heavy` | Geom(p=0.10), max 50 | ✗ Not run |

`delayed/buffer.py` now supports `delay_p` for Geometric sampling (added Phase 18), so
the infrastructure is in place. The experiments themselves have not been run.

**Known risk:** With heavy geometric delay (E[k] ≈ 10 steps), a bandit that hasn't received
enough matured rewards may under-explore. Whether the cost-sensitive bandit degrades
gracefully is an empirical question that is currently unanswered.

---

## 6. Drift Behavior

### Update (Phase 22): PageHinkley is now implemented

`drift/detectors.py` `PageHinkleyDetector` is now fully implemented (Phase 22). The ablation
study (Phase 22) demonstrates that the detector fires 44 false alarms on 1150 stationary steps
with `lambda_=50.0`. Threshold tuning is needed before RQ4 experiments can be run.

ADWIN and DDM remain as stubs (`NotImplementedError`).

| Experiment | Drift type | Status |
| --- | --- | --- |
| `RQ4-drift-none` | Stationary | ✗ Not run (synthetic env needed) |
| `RQ4-drift-abrupt-reset` | Abrupt, policy reset on detection | ✗ Not run |
| `RQ4-drift-abrupt-no-reset` | Abrupt, no reset | ✗ Not run |

**Phase 22 ablation shows the drift mechanism works but is over-sensitive on stationary
data.** On data with real concept drift, the reset should help. This requires the synthetic
drift environment (RQ4), not yet run.

---

## 7. Sanity Checks

### Check 7.1 — LinUCB and cost-sensitive bandit produce identical results

Phase 17 smoke run: both produce cumulative cost 980.0, IPS value -1.6333, ESS 600.0 on
every seed. This is expected: neither has received any training updates before evaluation.

**Interpretation:** The replay currently evaluates policy priors, not learned policies. The
cost-sensitive bandit's contributions — weighted cost targeting, drift-triggered reset — are
invisible when the model has no learned parameters.

**Fix required:** The replay must apply delayed policy updates *during* trajectory traversal
(receive matured rewards → update model → evaluate subsequent steps). This is online replay,
not the current one-pass evaluation.

### Check 7.2 — Static rules cost < bandit cost is expected, not evidence of superiority

Static rules: cumulative cost 600 (397 matched steps) vs bandit 980 (600 matched steps).
The cost-per-matched-step comparison: static rules 1.511, bandit 1.633. The difference
is small and possibly explained by selection bias — static rules only matches the lowest-risk
DEPLOY decisions.

**No statistical test can be run on the current results because the comparison is
confounded by coverage** (see §2).

### Check 7.3 — Pre-Phase-14 results (evaluation-results.json) are invalid

`experiments/results/evaluation-results.json` and `cost-analysis.json` are Phase 6
pre-pivot artifacts using the MAPE-K risk-score system with circular validity
(outcome predicted from the same features that drive the decision). **Do not cite
these numbers.** They are retained for historical reference only.

---

## What Is Reliable vs Not Yet Reliable

### Reliable (infrastructure only — not paper claims)

| Component | Status | Evidence |
| --- | --- | --- |
| `PendingRewardBuffer` delay logic | Correct | 174 tests pass (Phase 22) |
| `CostConfig` cost matrix | Correct | Unit tests, code review |
| `LinUCBPolicy` parameter updates (A, b) | Correct | Matrix update tests |
| `PageHinkleyDetector` implementation | Correct | Phase 22: fires 44× on 1150 stationary steps |
| `evaluate_ips` cost accumulation | Correct | Unit tests with known propensities |
| `TravisTorrentLoader` context extraction | Correct | Schema and integration tests |
| `StaticRulesPolicy` determinism and thresholds | Correct | Decision-semantics test suite |
| Online replay with delayed updates | Runs correctly | Phase 20–22; all rewards matured |
| Robustness runner + bootstrap CI machinery | Correct | Phase 21; zero-width CIs on deterministic data |

### Not Yet Reliable (cannot support paper claims)

| Claim | Why it fails | Status | Required fix |
| --- | --- | --- | --- |
| Cost-sensitive bandit outperforms static rules | 2-project synthetic data; not generalizable | **Preliminary** (Phase 20) | Run on real multi-project TravisTorrent |
| LinUCB and cost-sensitive bandit differ | Identical at same α; differ only at different α | **Preliminary** (Phase 20) | Different α sweep, or add drift detector |
| Results robust to cost matrix | Smoke data only | **Preliminary** (Phase 21) | Run with real data |
| Results robust to delay | Smoke data only; effect size is small | **Preliminary** (Phase 21) | Run with stochastic geometric delays |
| Drift adaptation reduces post-drift cost | PageHinkley implemented but RQ4 not run | **TODO** | Run synthetic drift environment (RQ4) |
| IPS estimates are valid counterfactuals | Logging propensities unknown | **Not addressed** | Use synthetic env with known propensity |
| Cost weighting is critical (vs binary) | Smoke data only | **Preliminary** (Phase 22) | Replicate on real data |

---

## What Needs Fixing Before Publication

Ordered by blocking severity:

### P0 — Blocks all evidence claims

1. **Online replay:** The experiment loop must deliver matured rewards to policies during
   trajectory traversal, then evaluate on subsequent steps. The bandit's learned parameters
   must actually influence its decisions before evaluation. Without this, every cost number
   compares untrained policies.

2. **Real TravisTorrent data:** The Phase 17 smoke CSV is a synthetic stand-in. Results on
   real multi-project data may differ substantially (project heterogeneity, censoring, action
   coverage distribution).

### P1 — Required for primary claim

3. **Implement one drift detector** (Page-Hinkley is simplest; see `drift/detectors.py`).
   Run RQ4 on the synthetic environment to demonstrate drift recovery.

4. **Run all four cost-matrix configurations** (RQ2). The primary claim requires showing
   the bandit advantage is not cost-matrix-dependent.

### P2 — Required for completeness

5. **Run delay sensitivity experiments** (RQ3). At minimum, run fixed-delay and
   Geom(0.30) on the synthetic environment.

6. **Document action coverage problem** in all result tables. Any replay table must
   state that logged actions are all DEPLOY and that CANARY/BLOCK coverage is zero.

7. **Implement heuristic-score baseline** (done Phase 18) and **thompson sampling**
   to complete the five-policy comparison table from `docs/evaluation-protocol.md`.

### P3 — Quality / reproducibility

8. **Replace synthetic smoke CSV with real TravisTorrent dump** (the 3.7M build
   dataset from MSR 2017). Filter to projects with ≥500 builds and ≥365 day history.

9. **Add a random-action baseline** to validate that bandit policies beat random choice.
   Without this floor, it is unclear whether any learned policy adds value.

---

## Summary Verdict

| Audit item | Finding | Severity |
| --- | --- | --- |
| IPS validity | Propensity = 1.0 assumed; reduces to direct method | High — bias direction unknown |
| Replay bias | All logged actions = DEPLOY; CANARY/BLOCK unobservable | Critical — comparisons confounded |
| ESS | Equals matched-action count; not a quality diagnostic here | Medium — misleading as reported |
| Cost sensitivity | Only one matrix evaluated | High — robustness unproven |
| Delay sensitivity | No experiments run | High — core mechanism unvalidated |
| Drift behavior | No detector implemented; no experiments | High — primary contribution unvalidated |
| Bandit vs static comparison | Bandit untrained; coverage confound | Critical — main claim unsupported |
| Phase 6 results | Pre-pivot, circular validity, wrong methodology | Critical — must not be cited |

**The pipeline is correct and the infrastructure is sound. The evaluation design is
not yet valid for supporting the paper's primary claim.**
