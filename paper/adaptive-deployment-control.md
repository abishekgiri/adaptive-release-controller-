# Continuous Deployment as Cost-Sensitive Decision-Making: When Contextual Bandits Outperform Static Rules and When They Don't

**Status: Research draft — not submitted. See Appendix A for the validity classification of every claim.**

---

## Abstract

Continuous deployment pipelines require a decision at every commit: deploy immediately, route through a canary, or block. Current practice frames this as a classification problem — predict failure probability, apply a static threshold. We argue the correct framing is *sequential cost minimization*: the relevant quantity is the operational cost of each action under outcome uncertainty, not a failure label.

We apply disjoint LinUCB (Li et al., 2010) and Bayesian linear Thompson Sampling (Agrawal & Goyal, 2013) to this three-action problem using an asymmetric cost matrix (a production incident costs 20× a correctly blocked bad change) and a pending-reward buffer that enforces the delayed-feedback invariant.

Results are two-sided. **Positive:** on synthetic data, the bandit outperforms a static rule by 19% when failure cost doubles and by 27% when blocking becomes cheaper — regimes where the cost structure rewards adaptation. An ablation confirms cost weighting is the dominant component: binary reward degrades cumulative cost by 31%. **Negative:** on real GitHub Actions CI data (600 runs, two public repositories), LinUCB over-blocks a 5.3%-failure-rate project — where deploy is optimal — and costs 3.8% *more* than a static rule. Feature sparsity and short trajectories prevent convergence to the project-specific optimal policy.

The takeaway: the bandit framing helps when failure costs are high and context is informative. When features are sparse or failure rates are low, well-calibrated static rules remain competitive.

---

## 1. Introduction

### 1.1 The Deployment Decision Problem

Every passing CI build triggers a deployment decision: ship to production, route through a canary, or block? A failed production deployment means incidents, on-call pages, and rollbacks. A blocked safe change costs developer time. A canary offers partial protection at throughput cost. Getting this right matters, and at scale it happens thousands of times per day.

Current tooling treats this as a prediction problem. Just-in-time defect prediction models (Kamei et al., 2013) estimate failure probability and apply a static threshold. Three structural limitations follow:

1. **The threshold encodes an implicit cost assumption.** Setting it requires a judgment about the cost of blocking safe changes vs. deploying bad ones. This assumption is rarely explicit and never adapts.
2. **Costs are asymmetric and context-dependent.** A production incident at a payment service costs an order of magnitude more than one at an internal tool. A single fixed threshold cannot represent this.
3. **The model does not learn from outcomes.** JIT models are trained offline on defect labels. They receive no feedback when predictions lead to incidents or unnecessary blocks.

### 1.2 Our Framing: Sequential Cost Minimization

We model deployment control as a **contextual bandit**: at each decision step *t*, the controller observes a context vector *x_t* (commit metadata, CI signal, change type, author history) and selects an action *a_t* ∈ {deploy, canary, block}. A reward *r_{t+k_t} = −cost(a_t, outcome_{t+k_t})* arrives *k_t* steps later, where *k_t* reflects the time until the deployment outcome is observable.

This framing differs from classification in three ways:

- **The objective is cumulative cost minimization, not accuracy maximization.** The policy is evaluated on what it costs to act, not on whether it correctly predicts a label.
- **The three-action space makes the canary option a first-class decision.** We model the action directly rather than applying a threshold to a risk score.
- **The policy learns online from its own decisions.** Each deployment outcome updates the policy's belief about which actions are cost-effective in which contexts.

### 1.3 When Does the Bandit Framing Pay Off?

The bandit advantage is not unconditional. Two conditions must hold simultaneously for a contextual bandit to outperform a well-tuned static rule:

**Condition 1 — sufficient failure rate asymmetry.** When failure rates are very low (say, 5%), the deploy arm is nearly always optimal regardless of context. A static rule that deploys aggressively matches the oracle. A bandit with exploration budget α wastes decisions sampling suboptimal arms (canary, block) before its posterior converges. The exploration cost dominates the expected gain.

**Condition 2 — sufficient informative context.** LinUCB with a *d*-dimensional feature vector needs roughly *O(d²)* updates per arm before its parameter estimates are statistically meaningful. With *d* = 13 in our setup, that is approximately 169 updates per arm before the confidence interval shrinks to first-order accuracy. On a 300-step real-world trajectory, the bandit has barely enough data to form reliable per-arm estimates, let alone differentiate arms based on context. If the feature vector carries no commit-level signal (no files changed, no test counts), the bandit degenerates to learning from failure rate and a bias term — information a static rule can encode directly.

When both conditions fail — low failure rate, feature sparsity, short trajectory — static rules are competitive and bandits may be worse. This paper documents both regimes empirically.

### 1.4 Contribution Summary

LinUCB and Thompson Sampling are well-established algorithms. Our contributions are in problem framing and empirical characterization:

- **Reframing:** Deployment control recast as sequential cost minimization with a three-action space {deploy, canary, block} and an explicit asymmetric cost matrix. Cumulative cost replaces accuracy as the primary metric.
- **Evaluation framework:** Online-replay protocol with a pending-reward buffer enforcing the delayed-feedback invariant; explicit bias disclosure; cost as the sole headline metric.
- **Component ablation:** Cost weighting, delayed feedback, and drift adaptation isolated and quantified. Cost weighting is the dominant factor (+31% cumulative cost when removed).
- **Two-sided empirical characterization:** Bandits outperform static rules when failure costs are high and context is informative. In the low-failure regime with feature sparsity (5.3% failure rate, no commit-level features), UCB-based policies over-block and cost 3.8% more than a static rule — a negative result that defines the operating envelope of the framing.

---

## 2. Problem Formulation

### 2.1 Contextual Bandit with Delayed Rewards

Let *X* be the space of pre-action observable features, *A* = {deploy, canary, block}, and *O* = {success, failure, censored, blocked} the outcome space.

At step *t* the agent observes *x_t ∈ X*, selects *a_t ∈ A*, and receives a reward

```
r_{t+k_t} = −cost(a_t, outcome_{t+k_t})
```

after a delay of *k_t* steps. In our experiments, *k_t = max(1, ⌈duration_t / 60⌉)* steps, where *duration_t* is the build duration in seconds. The pending-reward buffer holds the reward until step *t + k_t*; the policy has no information about it before that step.

When the outcome is not yet resolved by step *t + max_delay*, the reward is **censored** and excluded from policy updates.

### 2.2 Context Features

The feature vector *x_t ∈ ℝ^13* encodes 12 pre-action observables (normalised) plus a bias term:

| Dimension | Feature | Normalisation |
| --- | --- | --- |
| 0 | files_changed | ÷ 50 |
| 1 | lines_added | ÷ 1000 |
| 2 | lines_deleted | ÷ 500 |
| 3 | src_churn | ÷ 1500 |
| 4 | is_pr | binary |
| 5 | tests_run | ÷ 200 |
| 6 | tests_added | ÷ 10 |
| 7 | build_duration_s | ÷ 180 |
| 8 | author_experience | ÷ 10 |
| 9 | recent_failure_rate | [0, 1] |
| 10 | has_dependency_change | binary |
| 11 | has_risky_path_change | binary |
| 12 | bias | 1.0 |

All features are available before the deployment action. No post-deploy signal or outcome appears in *x_t*.

### 2.3 Cost Matrix

The cost function *cost: A × O → ℝ≥0* encodes operational priorities:

| Action | Outcome | Cost | Interpretation |
| --- | --- | ---: | --- |
| deploy | success | 0.0 | Successful rollout — no cost |
| deploy | failure | 10.0 | Production incident; on-call, rollback |
| canary | success | 1.0 | Canary overhead + promotion latency |
| canary | failure | 4.0 | Partial-rollout incident, limited blast |
| block | would succeed | 2.0 | Safe change delayed; developer wait |
| block | would fail | 0.5 | Risky change correctly held back |
| block | unknown | 2.0 | Counterfactual unobserved (replay) |

Cost asymmetry is deliberate: deploy + failure (10) is 20× block + would_fail (0.5). Reward is *r = −cost*. The cost matrix is configurable; §4.3 sweeps it.

### 2.4 Objective

The policy *π: X → A* minimizes cumulative operational cost over *T* decisions:

```
J(π) = Σ_{t=1}^{T} cost(a_t, outcome_{t+k_t})
```

Cumulative operational cost is the sole primary metric throughout this paper. Action distributions (deploy%, canary%, block%) are reported as diagnostics to explain *why* a cost difference arose.

---

## 3. Method

### 3.1 CostSensitiveBandit (Disjoint LinUCB with Cost-Sensitive Reward)

`CostSensitiveBandit` applies disjoint LinUCB (Li et al., 2010) to the deployment decision, replacing click-through reward with negative operational cost and routing all updates through the delayed-feedback buffer.

**Per-arm model.** For each arm *a ∈ A*:

```
A_a ∈ ℝ^{d×d},  initialized to λI
b_a ∈ ℝ^d,       initialized to 0
θ_a = A_a^{-1} b_a
```

**Action selection:**

```
UCB(a) = θ_a^T x_t + α √(x_t^T A_a^{-1} x_t)
a_t    = argmax_{a ∈ A} UCB(a)
```

**Update rule** (on matured reward *(x_i, a_i, r_i)*):

```
A_{a_i} ← A_{a_i} + x_i x_i^T
b_{a_i} ← b_{a_i} + r_i x_i,  where  r_i = −cost_i
```

The negative cost signal means the policy learns to prefer arms with lower expected operational cost.

### 3.2 Thompson Sampling Baseline

Bayesian linear Thompson Sampling (Agrawal & Goyal, 2013) maintains a per-arm Gaussian posterior over weight vectors:

```
Prior:     θ_a ~ N(0, v₀ · I)
Posterior: Λ_a = (1/v₀)I + (1/σ²)Σ xᵢxᵢᵀ
           b_a = (1/σ²) Σ rᵢ xᵢ
           μ_a = Λ_a^{-1} b_a
```

Action selection samples *θ_a ~ N(μ_a, Λ_a^{-1})* via Cholesky decomposition and picks *argmax_a θ_aᵀ x_t*. Exploration is implicit; no α parameter is needed. Default hyperparameters: *v₀ = 1.0*, *σ² = 0.1*. Updates use the same negative-cost reward as `CostSensitiveBandit`.

Thompson's stochastic action selection produces nonzero seed variance: identical data yields different trajectories across seeds. This makes it the only policy for which bootstrap CIs are informative in our current experiment setup (all other policies are deterministic given the same delayed-feedback schedule).

### 3.3 Delayed Reward Buffer

The `PendingRewardBuffer` (`delayed/buffer.py`) holds *(context, action, cost, outcome, censored)* until `pop_available(t + k_t)` is called. No model update occurs before maturity.

**The buffer's value is correctness, not performance.** Removing it improves cost by 1.1% (§5.3) because immediate feedback accelerates convergence. The buffer exists to prevent a policy from using future information at decision time — a requirement for valid evaluation in any real deployment setting.

### 3.4 Drift Adaptation (Exploratory — Excluded from Main Claims)

`CostSensitiveBandit` optionally accepts a Page-Hinkley detector (Mouss et al., 2004) that triggers model reset on detected distribution shift. On stationary data, the detector at *λ_PH = 50* fires 44 false alarms over 1,150 steps, making the full model 27% more expensive than the no-drift variant. Drift results are excluded from main claims; the threshold requires calibration against non-stationary data not yet available.

### 3.5 Baselines

| Policy | Description | Learning |
| --- | --- | --- |
| `static_rules` | Deterministic threshold rule (files changed, failure rate, risky paths) | None |
| `heuristic_score` | Weighted risk score → thresholded action; fixed weights | None |
| `linucb` | Disjoint LinUCB (Li et al., 2010), *r = −cost* | UCB exploration |
| `thompson` | Bayesian linear TS (Agrawal & Goyal, 2013), *r = −cost* | Posterior sampling |

At identical α, λ, and reward signal, `linucb` and `cost_sensitive_bandit` are mathematically identical (‖b-vector difference‖₂ = 0 confirmed experimentally). They are reported together in the main table.

---

## 4. Experiments

### 4.1 Synthetic Dataset

The primary experiment uses a synthetic dataset (`data/raw/travistorrent_smoke.csv`) generated to match the TravisTorrent schema (Beller et al., MSR 2017):

| Project | Builds | Failure rate | History span |
| --- | ---: | ---: | ---: |
| `smoke/alpha` | 600 | 15% | >365 days |
| `smoke/beta` | 550 | 35% | >365 days |

**Limitations:** Two projects, fixed failure rates, deterministic delay model. Bootstrap CIs collapse to zero for all deterministic policies. Thompson Sampling is the only policy with nonzero seed variance. No cross-project generalization is possible from this dataset.

### 4.2 Online Replay Setup and Bias

We use online replay (`evaluation/online_replay.py`): policies learn during trajectory traversal. Costs are *cost(policy_action, logged_CI_outcome)*, using CI outcome as a counterfactual proxy for the deployment outcome.

**Online replay is a biased evaluation.** Every logged action is DEPLOY (the loader hardcodes this). Costs for CANARY and BLOCK are counterfactual: we assume CI outcome would be the same regardless of deployment action. This is approximately valid for TravisTorrent (CI runs before any deployment) but unverifiable in general. We use online replay exclusively for learning-dynamics verification, not causal cost estimation.

**Configuration (synthetic experiment):**

| Parameter | Value |
| --- | --- |
| α | 1.0 |
| λ | 1.0 |
| Delay model | *k_t = max(1, ⌈build_duration_s / 60⌉)* |
| Cost matrix | Default (§2.3) |
| Seeds | 0–4 |
| flush_at_end | True |

### 4.3 Robustness Conditions

Five conditions test whether conclusions depend on cost matrix or delay setting:

| Condition | Key change |
| --- | --- |
| Default | — |
| High failure | deploy_failure: 10→20, canary_failure: 4→8 |
| Low block | block_safe: 2→1, block_unknown: 2→1 |
| Short delay | delay_step_seconds: 60→120 |
| Long delay | delay_step_seconds: 60→30 |

Percentile bootstrap CI: 10,000 resamples, seed 42, over seeds 0–4.

### 4.4 Ablation Conditions

| Variant | What changes |
| --- | --- |
| `no_drift` (reference) | No drift reset |
| `no_delay` | Immediate updates (no buffer) |
| `no_cost` | Binary reward (−1/0 instead of −cost) |
| `full` | + PageHinkley drift reset |

### 4.5 Real-World Sanity Check Dataset

To test whether the synthetic findings are contradicted by real CI data, we collected GitHub Actions workflow run history from two public repositories via the GitHub REST API:

| Project | Runs | Failure rate | Span |
| --- | ---: | ---: | ---: |
| `psf/requests` | 300 | 5.3% | 20 days |
| `pallets/flask` | 300 | 23.3% | 99 days |

**Schema mapping:** `head_sha` → `git_trigger_commit`, `conclusion` → `tr_status`, timestamps → duration. No commit-level features are available without additional per-commit API calls; fields default to 0. The feature vector is dominated by the bias term and recent-failure-rate — 11 of 13 dimensions carry no project-specific signal.

**Filter relaxation:** Standard filters (`min_builds=500`, `min_history_days=365`) cannot be satisfied by a 20–99 day API sample. Relaxed to `min_builds=100`, `min_history_days=0`. Results are not directly comparable to the synthetic conditions.

**Censored rewards:** Cancelled CI runs (≈3%) produce genuine censored rewards — the first experiment confirming real-world censoring rather than theoretical.

---

## 5. Results

### 5.1 Main Results (Synthetic Data)

**[Preliminary — synthetic 2-project dataset. Deterministic policies have zero CI width.]**

**Table 1:** Cumulative cost, default cost matrix, synthetic dataset. All 1,150 rewards matured (0 censored). Thompson: mean ± std across seeds 0–4. *Takeaway: all three non-heuristic policies are effectively tied in aggregate; the per-project breakdown below reveals the structure the aggregate hides.*

| Policy | Steps | Cumul. Cost | Mean/Step | Deploy% | Canary% | Block% |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `static_rules` | 1150 | **1878.0** | **1.633** | 2.6% | 60.9% | 36.5% |
| `heuristic_score` | 1150 | 2319.0 | 2.017 | 37.3% | 62.7% | 0.0% |
| `linucb` / `cost_sensitive_bandit` | 1150 | 1879.0 | 1.634 | 8.0% | 5.7% | 86.3% |
| `thompson` (n=5 seeds) | 1150 | **1877 ± 58** | 1.633 ± 0.050 | 6.4% | 30.5% | 63.1% |

`linucb` and `cost_sensitive_bandit` are identical at the same α and λ — mathematically expected (‖b-vector difference‖₂ = 0).

The aggregate tie (1878 vs. 1879) is a dataset artifact. **Per-project, the bandit and static rule take opposite positions:**

| Project | Failure rate | `static_rules` | `linucb` | Optimal arm (expected cost) |
| --- | ---: | ---: | ---: | --- |
| `smoke/alpha` | 15% | **911.5** | 982.0 | canary (1.45/step) |
| `smoke/beta` | 35% | 966.5 | **897.0** | block (1.475/step) |

At 15% failure, canary is cheaper than block (1.45 vs. 1.775/step); the static rule's canary-heavy strategy is near-optimal and LinUCB over-blocks. At 35% failure, block is cheapest (1.475/step); LinUCB correctly converges to 86.3% block.

**Thompson Sampling** produces the same mean cost as LinUCB but with std = 58 across seeds (range 1814–1970). Its action distribution differs: 30.5% canary vs. 5.7% for LinUCB, reflecting posterior uncertainty rather than point-estimate UCB. This is the expected behavioral difference between posterior sampling and optimism-based exploration.

### 5.2 Robustness Results (Synthetic Data)

**[Preliminary — synthetic data. Bootstrap CIs are zero-width for deterministic policies.]**

**Cost matrix sweep:**

| Policy | Default (df=10) | High failure (df=20) | Low block (bs=1) |
| --- | ---: | ---: | ---: |
| `static_rules` | 1878 | 2564 | 1584 |
| `heuristic_score` | 2319 | 4103 | 2319 |
| `linucb` | 1879 | **2079** | **1164** |
| `cost_sensitive_bandit` | 1879 | **2079** | **1164** |

Under high failure cost (deploy_failure=20): the bandit outperforms static rules by **485 units (19%)** by shifting to 92.7% block. Static rules cannot adapt their fixed thresholds. Under low block penalty (block_safe=1): the bandit saves **420 units (27%)** by exploiting cheaper blocking.

**Delay sweep:**

| Policy | Default (60s) | Short delay (120s) | Long delay (30s) |
| --- | ---: | ---: | ---: |
| `static_rules` | 1878 | 1878 | 1878 |
| `linucb` | 1879 | **1849** | 1897 |
| `cost_sensitive_bandit` | 1879 | **1849** | 1897 |

Under long delay (doubled step-count delay), the bandit's uninformed-prior phase extends and costs 19 units more than static rules (1.0%), consistent with O(√(d_max · T)) delay-induced regret inflation (Pike-Burke et al., 2018). Under short delay, the bandit gains 29 units by converging faster.

### 5.3 Ablation Study (Synthetic Data)

**[Preliminary — synthetic data. All 5 seeds identical for deterministic variants.]**

**Table 2:** Component ablation, default cost matrix, seed 0. Reference = `no_drift`. *Takeaway: cost weighting is the dominant component (+31%); drift detection is destructive on stationary data (+27%); the buffer costs 1.1% for temporal validity.*

| Variant | Cumul. Cost | Mean/Step | Deploy% | Block% | Drift Resets |
| --- | ---: | ---: | ---: | ---: | ---: |
| `no_cost` (binary reward) | 2469.0 | 2.147 | 41.4% | 36.5% | — |
| `full` (+ PageHinkley drift) | 2383.0 | 2.072 | 40.4% | 29.5% | 44 |
| `no_drift` (reference) | 1879.0 | 1.634 | 8.0% | 86.3% | 0 |
| `no_delay` (immediate updates) | **1857.5** | **1.615** | 1.7% | 78.4% | — |

| Component | Δ vs. `no_drift` | Δ% | Interpretation |
| --- | ---: | ---: | --- |
| Cost weighting → binary (`no_cost`) | +590 | +31.4% | Destroys action–outcome asymmetry |
| Drift reset on stationary data (`full`) | +504 | +26.8% | 44 false alarms, each erasing learned weights |
| Delayed buffer → immediate (`no_delay`) | −21.5 | −1.1% | Faster convergence; buffer costs temporal validity |

**Cost weighting (+31%)** is the dominant component. Binary reward (−1/0) gives the block arm the same signal as deploy for failures it correctly avoided — wrong gradient. The 31% degradation quantifies the cost of ignoring the asymmetric cost matrix.

**Delay removal (−1.1%).** Removing the buffer improves performance slightly because immediate feedback accelerates convergence. The buffer's purpose is temporal validity, not performance. The 1.1% is the honest price of a correct evaluation.

**Drift on stationary data (+27%).** PageHinkley at λ_PH = 50 fires 44 false alarms, resetting learned weights each time and reverting to deploy-heavy uninformed priors (40.4% deploy rate post-reset vs. 8.0% converged). This is detector behavior on stationary data; it does not characterize the component's value on non-stationary data. Drift results are **excluded from main claims**.

### 5.4 Real-World Sanity Check

**[Highly preliminary — real GitHub Actions data, 2 projects, feature sparsity, relaxed filters, biased evaluation. Do not compare magnitudes against synthetic results.]**

**Table 3:** Online replay on real GitHub Actions data, default cost matrix, seeds 0–4. *Takeaway: in the low-failure regime with feature sparsity, LinUCB over-blocks and costs 3.8% more than a static rule; Thompson's mean is better but variance is high.*

| Policy | Steps | Censored | Cumul. Cost | Mean/Step | Deploy% | Canary% | Block% |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `static_rules` | 600 | 21 | **644.5** | **1.113** | 64.3% | 20.0% | 15.7% |
| `heuristic_score` | 600 | 22 | 860.0 | 1.488 | 100.0% | 0.0% | 0.0% |
| `linucb` / `cost_sensitive_bandit` | 600 | 17 | 669.5 | 1.148 | 53.0% | 4.8% | 42.2% |
| `thompson` (n=5 seeds) | 600 | 17–21 | **585** ± 99 | **1.008** ± 0.171 | varies | varies | varies |

Thompson per-seed: 445.5 / 520.0 / 649.5 / 631.5 / 680.0 (seeds 0–4).

**Per-project expected cost analysis:**

| Project | Failure rate | E[deploy] | E[canary] | E[block] | Optimal |
| --- | ---: | ---: | ---: | ---: | --- |
| `psf/requests` | 5.3% | **0.53** | 1.16 | 1.92 | deploy |
| `pallets/flask` | 23.3% | 2.33 | 1.70 | **1.65** | block |

**Finding R1 (negative result): LinUCB over-blocks in the low-failure regime and costs more than static rules (669.5 vs. 644.5).** At 5.3% failure, deploy is optimal (0.53/step); blocking costs 1.92/step. LinUCB converges to 42.2% block across both projects because feature sparsity prevents differentiation — both projects look similar with zero files_changed, zero tests_run, etc. The bandit learns primarily from the recent-failure-rate feature, which does not converge quickly enough to assign different strategies to the two projects within 300 steps. The static rule's explicit threshold on `files_changed` and `recent_failure_rate` happens to produce a 64.3% deploy rate that is close to optimal for this failure-rate distribution.

This finding directly illustrates §1.3 Conditions 1 and 2: the low-failure regime and feature sparsity together prevent the bandit from outperforming a rule that has those conditions hardcoded.

**Finding R2: Thompson is beneficial on average but high-variance.** Mean cost 585 (±99) beats both LinUCB (669.5) and static rules (644.5). But variance is large: seed 0 achieves 445.5 (33% better than static), seed 4 achieves 680.0 (5.5% worse). On 300-step trajectories, Thompson's posterior has not converged and action distributions swing widely across seeds.

**Finding R3: Real-world censoring is present.** Cancelled CI runs produce 17–22 censored rewards per trajectory (≈3%), varying by policy. This confirms reward censoring is not merely theoretical.

---

## 6. Key Findings

### 6.1 Reliable Findings (mechanism verification — hold regardless of dataset)

**F1: Cost weighting is the most important model component (+31% cost when removed).**
Binary reward destroys the signal asymmetry the cost matrix encodes. This is a mechanism result: it holds whenever the cost matrix is asymmetric (20× ratio between deploy+failure and block+would_fail). It does not require real-world validation to hold.

**F2: The delayed-feedback buffer enforces correctness at a 1.1% performance cost.**
Removing the buffer improves performance because immediate feedback accelerates convergence. The buffer's value is validity — preventing use of future information at decision time (Joulani et al., 2013). The 1.1% is the honest cost of temporal correctness.

**F3: Page-Hinkley at λ_PH = 50 fires 44 false alarms on 1,150 stationary steps.**
Expected behavior for a sensitive detector on data with no drift. The threshold requires calibration; we make no claim about its performance on non-stationary data.

**F4: Thompson Sampling produces nonzero seed variance (std = 58 synthetic, std = 99 real); LinUCB does not.**
Stochastic posterior sampling yields different per-trajectory behavior. On real short-trajectory data, this variance is large enough that any individual seed can be best or worst of all policies.

### 6.2 Preliminary Findings (synthetic data — cannot generalize)

**F5: Bandit advantage scales with failure cost severity (+19% at df=20, +27% at low block cost).**
When the cost matrix moves away from the default implicit assumptions of a fixed static rule, the adaptive policy exploits the new cost structure and the static rule cannot. The qualitative pattern is expected from first principles; the magnitudes are dataset-specific.

**F6: Under severe delay, the learning bandit costs 1% more than static rules.**
Consistent with delay-induced regret inflation. The effect is small and may not survive on real stochastic-delay data.

**F7: The aggregate bandit/static tie is a cancellation artifact.**
At the project level, the bandit wins at 35% failure and loses at 15% failure. The tie is specific to this two-project synthetic dataset design.

### 6.3 Real-Data Findings (highly preliminary — 2 real projects, feature sparsity)

**F8 (negative result): LinUCB over-blocks in the low-failure regime and costs 3.8% more than static rules.**
On psf/requests (5.3% failure, feature sparsity, 300 steps), the bandit cannot distinguish the project from pallets/flask using the available feature signal. It converges to a block-heavy strategy that is near-optimal for flask but expensive for requests. Static rules' explicit threshold happens to match the optimal strategy for this failure-rate mix. This is precisely the failure mode predicted by §1.3.

**F9: Thompson Sampling is 9% cheaper than LinUCB on average across seeds (real data).**
Posterior sampling's stochastic exploration sometimes discovers the project-specific optimal strategy (seed 0: 445.5) but with high variance (seed 4: 680.0). The mean improvement over static rules is 9%, but no individual seed reliably beats static rules with high probability.

**F10: Reward censoring is empirically confirmed in real CI data.**
Cancelled runs produce 3% censored rewards, with policy-dependent variation. This validates the buffer's censoring path as exercised on real data.

---

## 7. Threats to Validity and Limitations

### 7.1 Online Replay is Biased — All Results Are Simulations

Online replay computes costs as *cost(policy_action, logged_CI_outcome)*. Every logged action is DEPLOY. Costs for CANARY and BLOCK are counterfactual, resting on the assumption that CI outcome is independent of the deployment action taken. This assumption is unverifiable. The evaluation measures relative policy rankings within the simulation; it does not measure real-world operational cost.

No logging propensities exist. Inverse propensity scoring cannot be applied — the IPS weight is identically 1.0, reducing the estimator to the direct method. This is unbiased only if the true logging policy always deployed, which is false for human-operated pipelines.

The evaluation is internally consistent (the same counterfactual assumption applies to every policy equally), so relative rankings are meaningful. Absolute cost numbers are simulation artifacts.

### 7.2 Synthetic Dataset: Two Projects, Fixed Failure Rates, Zero CI Width

The primary experiment uses 1,150 rows across two synthetic projects at fixed failure rates (15% and 35%). The aggregate tie between LinUCB and static rules (1878 vs. 1879) is a design artifact — the two projects cancel. Deterministic delay model produces zero bootstrap CI width for all policies except Thompson. No claim generalizes beyond this dataset without real multi-project replication.

### 7.3 Real Sanity Check: Feature Sparsity, Short Trajectories, Relaxed Filters

The GitHub Actions experiment uses a feature vector in which 11 of 13 dimensions carry no commit-level signal. The bandit degenerates to learning from failure rate and bias term. With 300 steps per project and d=13 dimensions, the bandit is near its minimum convergence threshold (O(d²) = 169 updates per arm). Both conditions in §1.3 are violated: psf/requests is in the low-failure regime AND feature sparsity prevents informative context. The negative result (LinUCB loses to static rules) is expected under these conditions and should not be interpreted as evidence that bandits generally underperform static rules.

### 7.4 LinUCB and CostSensitiveBandit Are Identical at Same Hyperparameters

‖b-vector difference‖₂ = 0 at α=1.0, λ=1.0. The implementations diverge only when α differs, the drift detector is active, or future extensions are applied. The current experiments cannot distinguish them.

### 7.5 Drift Detection Is Miscalibrated for Stationary Data

Page-Hinkley at λ_PH = 50 fires 44 false alarms on 1,150 stationary steps, erasing learned weights each time. The component is implemented and mechanistically correct; it requires non-stationary data to evaluate its intended behavior. The `full` model is excluded from all performance claims.

### 7.6 Thompson Sampling Propensities Are Intractable

The true selection probability for Thompson Sampling requires integrating over the posterior — intractable in closed form. We report propensity = 1.0 throughout. IPS correction cannot be applied to Thompson results even in principle. Thompson's cost estimates carry the standard replay bias plus this additional limitation.

---

## 8. Conclusion

### What Works

**Cost-sensitive reward design is the most impactful engineering decision.** Replacing the asymmetric cost matrix with binary (−1/0) feedback degrades performance by 31%. This gap exists because binary reward cannot distinguish a deploy failure (cost 10) from a block that correctly prevented a failure (cost 0.5) — both receive the same −1 signal for failures. The cost matrix makes the asymmetry explicit and trainable.

**The bandit framing delivers when both conditions in §1.3 hold.** At deploy_failure=20 (doubled), the bandit outperforms static rules by 19% by learning a block-heavy strategy the static rule cannot reach. At low block cost, it outperforms by 27%. The framing's advantage is conditional, not universal — it requires sufficient failure rate and sufficient context signal to justify the exploration cost.

**Thompson Sampling's stochastic exploration is the right tool for short or uncertain trajectories.** Its mean performance on real data (585 vs. 644.5 for static rules) is better, and its variance across seeds (±99) makes it the only policy for which uncertainty quantification is meaningful in the current setup.

### What Doesn't Work (and Why)

**LinUCB over-blocks in the low-failure regime with feature sparsity.** On psf/requests (5.3% failure, zero commit-level features), the bandit cannot distinguish the project from pallets/flask and applies a one-size-fits-all block-heavy strategy. At 5.3% failure, deploy is optimal (0.53/step vs. block 1.92/step); LinUCB's 42.2% block rate costs 3.8% more than a static rule. This is not a flaw in LinUCB — it is a flaw in the application: the convergence condition (informative context, sufficient trajectory length) is violated.

**Drift detection is destructive on stationary data at default threshold.** PageHinkley at λ_PH = 50 fires 44 false alarms, each resetting learned weights, resulting in a model 27% more expensive than no-drift. The component needs calibration against non-stationary data before any performance claim can be made.

**The aggregate synthetic tie is misleading.** LinUCB and static rules tie at 1878 vs. 1879 only because the two synthetic projects are balanced to cancel. Project-level, the results diverge substantially (897 vs. 966 on the 35% project; 911 vs. 982 on the 15% project). Aggregates hide the structure.

### What This Changes About Deployment Decision Research

The classification framing of JIT defect prediction (Kamei et al., 2013) encodes a fixed, implicit cost assumption at the threshold. When that assumption is wrong — when failure costs are high, when blocking is cheap, when the failure rate is project-specific — the threshold cannot adapt. A contextual bandit with an explicit cost matrix can. The cost matrix itself is the interface through which operational priorities are expressed; it should be reported as an explicit hyperparameter in any deployment control system, not buried in a threshold.

The negative result is equally informative: bandits do not uniformly outperform static rules. They require sufficient context signal and sufficient trajectory length. In the low-failure regime with feature sparsity, the exploration cost dominates and a well-tuned static rule is competitive. This defines the operating envelope for the framing.

### Future Work

Three directions are tractable extensions of the current infrastructure:

1. **Feature-rich real data.** Fetch commit-level metadata (files changed, tests run, author history) alongside GitHub Actions run status. This populates all 12 non-bias dimensions of the feature vector and enables a fair comparison on real data.
2. **Non-stationary evaluation.** Design a synthetic environment with abrupt failure-rate shifts and evaluate whether PageHinkley (with calibrated λ_PH) reduces post-shift regret relative to the no-drift variant.
3. **Per-project exploration calibration.** Low-failure-rate projects benefit from lower α (less UCB exploration); high-failure projects benefit from higher α. A meta-policy that adapts α based on observed failure rate would avoid the over-blocking failure mode documented in §5.4.

---

## References

- Agrawal, S., Goyal, N. "Thompson Sampling for Contextual Bandits with Linear Payoffs." ICML 2013.
- Beller, M., et al. "TravisTorrent: Synthesizing Travis CI and GitHub for Full-Stack Research." MSR 2017.
- Chu, W., et al. "Contextual Bandits with Linear Payoff Functions." AISTATS 2011.
- Gama, J., et al. "A Survey on Concept Drift Adaptation." ACM CSUR 2014.
- Joulani, P., et al. "Online Learning under Delayed Feedback." ICML 2013.
- Kamei, Y., et al. "A Large-Scale Empirical Study of Just-in-Time Quality Assurance." TSE 2013.
- Li, L., et al. "A Contextual-Bandit Approach to Personalized News Article Recommendation." WWW 2010.
- McIntosh, S., Kamei, Y. "Are Fix-Inducing Changes a Moving Target?" EMSE 2018.
- Mouss, H., et al. "Test of Page-Hinckley, an Approach for Fault Detection in an Agro-Alimentary Production System." MED 2004.
- Pike-Burke, C., et al. "Bandits with Delayed, Aggregated Anonymous Feedback." ICML 2018.

---

## Appendix A: Validity Classification

| Claim | Status | Evidence | Condition for promotion |
| --- | --- | --- | --- |
| Cost weighting important (+31% without it) | **Preliminary** | Smoke dataset, 2 projects | Replicate on ≥10 real projects with full features |
| Bandit advantage grows with failure cost | **Preliminary** | Smoke dataset | Real data, multiple projects |
| Delay reduces learning speed by 1.1% | **Preliminary** | Smoke dataset | Stochastic delay experiment |
| Thompson variance > LinUCB variance | **Reliable** | Smoke + GitHub Actions, 5 seeds each | — |
| Buffer enforces delayed-feedback correctness | **Reliable** | Code + 180 passing tests | — |
| PageHinkley fires false alarms at λ_PH=50, stationary data | **Reliable** | Expected detector behavior | — |
| Infrastructure runs correctly on real GitHub Actions data | **Reliable** | 600 real runs, 2 projects executed | — |
| LinUCB over-blocks in low-failure regime with feature sparsity | **Preliminary** | GitHub Actions sanity check, 1 project | Feature-rich real data |
| Drift adaptation reduces cost on non-stationary data | **Not evaluated** | Experiment not run | Synthetic drift schedule |
| Bandit generally outperforms static rules | **Not established** | Static rules win on real low-failure data | Feature-rich multi-project real data |

---

## Appendix B: Experiment Reproducibility

```bash
# Main online replay (includes Thompson Sampling)
for seed in 0 1 2 3 4; do
    python -m experiments.run_bandits \
        --config experiments/configs/online_smoke.json \
        --seed $seed
done

# Real-data sanity check (requires data/raw/github_actions_real.csv)
for seed in 0 1 2 3 4; do
    python -m experiments.run_bandits \
        --config experiments/configs/real_github_actions.json \
        --seed $seed
done

# Robustness sweep
python -m experiments.run_robustness \
    --configs experiments/configs/online_smoke.json \
              experiments/configs/robustness_high_failure.json \
              experiments/configs/robustness_low_block.json \
              experiments/configs/robustness_short_delay.json \
              experiments/configs/robustness_long_delay.json \
    --seeds 0 1 2 3 4

# Ablation study
for seed in 0 1 2 3 4; do
    python -m experiments.run_ablations --seed $seed
done
```

Results are written to `experiments/results/` (gitignored). The smoke dataset is at `data/raw/travistorrent_smoke.csv` (gitignored). All policy implementations, evaluation runners, and config files are tracked in version control.
