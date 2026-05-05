# Adaptive Deployment Control via Cost-Sensitive Contextual Bandits with Delayed Feedback

**Status: Research draft — not submitted. See §Validity Classification for what claims the current evidence supports.**

---

## Abstract

Continuous deployment pipelines must decide, for every commit, whether to deploy immediately, route through a canary, or block. Current practice encodes this decision as a classification problem (predict whether the build will fail) and applies a static threshold. We argue this framing is wrong: the relevant quantity is not the probability of failure but the *operational cost* of each deployment action given uncertainty about the outcome. We formulate deployment control as a cost-sensitive contextual bandit with delayed rewards and a three-action decision space {deploy, canary, block}. Our contribution is CostSensitiveBandit: a disjoint LinUCB policy that optimizes negative operational cost, gates learning through a pending reward buffer until feedback is observable, and optionally resets learned state upon drift detection. Experiments on a synthetic TravisTorrent-format dataset (1,150 builds, 2 projects) show that cost weighting is the most important component of the model: replacing the asymmetric cost signal with binary feedback increases cumulative cost by 31%. Under high failure cost regimes (deploy_failure=20), the bandit outperforms a hand-tuned static rule by 19%. Under severe delay, the static rule marginally recovers. Drift detection results are excluded from the main claims due to false-alarm behavior on stationary data; this is noted as a known limitation requiring threshold calibration.

---

## 1. Introduction

### 1.1 The Deployment Decision Problem

Software engineering teams push hundreds to thousands of commits per day. Each commit triggers a CI pipeline, and when CI passes, a deployment decision must be made: send this change to production immediately, stage it through a canary deployment, or hold it for further review? This decision is consequential: a failed production deployment triggers incidents, on-call pages, and rollbacks. An unnecessarily blocked safe change slows developer velocity. A canary deployment offers partial protection at the cost of slower delivery.

Current tooling treats this as a prediction problem. Just-in-time defect prediction models (Kamei et al., 2013) estimate the probability that a commit introduces a bug and apply a static threshold to flag risky changes. Several limitations follow from this framing:

1. **The threshold is a decision, not a metric.** Setting it requires an implicit cost assumption about false positives (blocked safe changes) vs. false negatives (deployed bad changes). This assumption is typically unexamined and fixed.
2. **Costs are asymmetric and context-dependent.** A failed deploy to a payment service costs far more than one to an internal tool. A binary classifier with a single threshold cannot represent this.
3. **The model does not learn from deployment outcomes.** JIT models are trained offline on defect labels. They do not receive feedback when their predictions lead to incidents or unnecessary blocks.
4. **Distributions shift.** McIntosh and Kamei (2018) show that JIT models degrade as project characteristics evolve. Re-training is deferred to scheduled batch jobs rather than triggered by observed distributional change.

### 1.2 Our Framing: Sequential Cost Minimization

We model deployment control as a **contextual bandit**: at each decision step *t*, the controller observes a context vector *x_t* (commit metadata, CI signal, change type, author history) and selects an action *a_t* ∈ {deploy, canary, block}. A reward *r_{t+k_t} = −cost(a_t, outcome_{t+k_t})* arrives *k_t* steps later, where *k_t* reflects the time until the deployment outcome is observable.

This framing differs from classification in three ways:

- **The objective is cumulative cost minimization, not accuracy maximization.** The policy is evaluated on what it costs to act, not on whether it correctly predicts a label.
- **The three-action space makes the canary option a first-class decision.** A classifier outputs a risk score; the system operator decides what to do with it. We model the action directly.
- **The policy learns online from its own decisions.** Each deployment outcome updates the policy's belief about which actions are cost-effective in which contexts.

### 1.3 Contribution Summary

- **Problem formulation:** We specify deployment control as a cost-sensitive contextual bandit with delayed rewards (§2), with an explicit asymmetric cost matrix encoding operational priorities.
- **Algorithm:** CostSensitiveBandit (§3): disjoint LinUCB with a pending-reward buffer that enforces delayed-feedback correctness, and optional drift-triggered model reset.
- **Experiments:** Online replay on a TravisTorrent-format dataset (§4–5), with robustness sweeps across cost matrices and delay settings, and an ablation study isolating model components.
- **Findings:** Cost weighting is the most important component (+31% cost when removed). Bandit advantage over static rules scales with failure cost severity. Delay reduces learning speed but not the eventual action preference (§6).
- **Honest limitations:** No propensities → counterfactual evaluation is unidentified. Drift detection results are excluded from claims (§7).

---

## 2. Problem Formulation

### 2.1 Contextual Bandit with Delayed Rewards

Let *X* be the space of pre-action observable features, *A* = {deploy, canary, block}, and *O* = {success, failure, censored, blocked} the outcome space.

At step *t* the agent observes *x_t ∈ X*, selects *a_t ∈ A*, and receives a reward

```
r_{t+k_t} = −cost(a_t, outcome_{t+k_t})
```

after a delay of *k_t* steps. The delay in our experiments is deterministic: *k_t = max(1, ⌈duration_t / 60⌉)* steps, where *duration_t* is the build duration in seconds. The pending-reward buffer holds *r_{t+k_t}* until step *t + k_t*; the policy receives no information about the reward before that step.

When the outcome is not yet resolved by step *t + max_delay*, the reward is **censored**: it is excluded from policy updates. We report the count of censored rewards in every experiment.

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

All features are available before the deployment action is taken. No post-deploy signal, outcome, or risk score appears in *x_t*.

### 2.3 Cost Matrix

The cost function *cost: A × O → ℝ≥0* encodes operational priorities. The default configuration:

| Action | Outcome | Cost | Interpretation |
| --- | --- | ---: | --- |
| deploy | success | 0.0 | Successful rollout — no cost |
| deploy | failure | 10.0 | Production incident; on-call, rollback |
| canary | success | 1.0 | Canary overhead + promotion latency |
| canary | failure | 4.0 | Partial-rollout incident, limited blast |
| block | would succeed | 2.0 | Safe change delayed; developer wait |
| block | would fail | 0.5 | Risky change correctly held back |
| block | unknown | 2.0 | Counterfactual unobserved (replay) |

Cost asymmetry is deliberate: deploy + failure (cost 10) is 20× the cost of block + would_fail (cost 0.5). This encodes the operational reality that production incidents are far more expensive than unnecessary holds.

Reward is *r = −cost* (maximizing reward ≡ minimizing cost).

### 2.4 Objective

The policy *π: X → A* minimizes cumulative operational cost over a trajectory of *T* decisions:

```
J(π) = Σ_{t=1}^{T} cost(a_t, outcome_{t+k_t})
```

We evaluate policies by comparing *J(π)* across conditions. Cumulative regret against an oracle (which observes the outcome before acting) is defined but not yet computed due to the BLOCK + unknown counterfactual; this is a stated open problem.

### 2.5 Non-Stationarity

The conditional distribution *P(outcome | x)* is assumed piecewise-stationary. Distribution shift arises in practice from: codebase architecture changes, team composition, test suite evolution, and seasonal deployment patterns. We model this with a Page-Hinkley detector on the reward stream; detected shifts trigger optional model reset. Drift experiments on non-stationary data are a stated future direction (§7).

---

## 3. Method

### 3.1 CostSensitiveBandit

CostSensitiveBandit is a disjoint LinUCB policy (Li et al., 2010) with three extensions: (1) it optimizes negative operational cost rather than click-through reward; (2) all updates route through a pending-reward buffer that enforces the delayed-feedback invariant; (3) it accepts an optional drift detector that triggers model reset.

**Per-arm model.** For each arm *a ∈ A*:

```
A_a ∈ ℝ^{d×d},  initialized to λI
b_a ∈ ℝ^d,       initialized to 0
θ_a = A_a^{-1} b_a
```

**Action selection** (greedy UCB):

```
UCB(a) = θ_a^T x_t + α √(x_t^T A_a^{-1} x_t)
a_t    = argmax_{a ∈ A} UCB(a)
```

The exploration parameter *α* scales the confidence interval. *λ* is the ridge regularisation coefficient.

**Update rule.** When a matured reward *(x_i, a_i, r_i)* is released from the buffer:

```
A_{a_i} ← A_{a_i} + x_i x_i^T
b_{a_i} ← b_{a_i} + r_i x_i,  where  r_i = −cost_i
```

The negative cost reward means the policy learns to prefer arms with lower expected operational cost.

### 3.2 Delayed Reward Buffer

The `PendingRewardBuffer` (`delayed/buffer.py`) implements the temporal gating of rewards. At decision step *t*, the policy schedules a reward with a specific release step *t + k_t*. The buffer holds the *(context, action, cost, outcome, censored)* tuple until `pop_available(t + k_t)` is called. No model update occurs before maturity.

This enforces the key invariant: **policy learning cannot use future information.** Policies that update at decision time are implicitly assuming *k_t = 0*, which is invalid in deployment settings where outcomes are observed hours to days later.

In the online replay experiments, *k_t = max(1, ⌈build_duration_s / 60⌉)* steps, ranging from 1 to 60 steps depending on build duration. All rewards mature and are applied at end-of-trajectory (`flush_at_end=True`).

### 3.3 Drift Adaptation [Exploratory — Excluded from Main Claims]

CostSensitiveBandit optionally accepts a `DriftDetector`. On each matured reward, the detector updates on the cost observation:

```
if detector.update(cost_t):
    reset all A_a ← λI, b_a ← 0
    detector.reset()
```

We implement the Page-Hinkley test (Mouss et al., 2004). The detector tracks a cumulative sum *S_t* of deviations from a running mean, and signals drift when *S_t − min_{s≤t} S_s > λ_PH*.

**Important limitation:** On stationary data (our smoke dataset), the Page-Hinkley detector with *λ_PH = 50* fires 44 false alarms over 1,150 steps. Each false alarm discards all learned weights, causing the full model to perform worse than the no-drift variant (§5.3). This indicates the threshold requires calibration. Drift adaptation results on non-stationary data are not yet available and are therefore excluded from the paper's claims. We report the false-alarm behavior honestly as a mechanism characterization.

### 3.4 Baselines

| Policy | Description |
| --- | --- |
| `static_rules` | Deterministic threshold rule over context features (files changed, failure rate, risky paths) |
| `heuristic_score` | Weighted risk score → thresholded action; fixed weights, no learning |
| `linucb` | Standard LinUCB (Li et al., 2010); same model as CostSensitiveBandit at identical α, λ |

At identical hyperparameters (α=1.0, λ=1.0) and with the same reward signal (*r = −cost*), `linucb` and `cost_sensitive_bandit` are mathematically equivalent (‖b_linucb − b_bandit‖₂ = 0 to machine precision after the same update sequence). The two implementations diverge when hyperparameters differ or when `cost_sensitive_bandit` includes a drift detector. In the main experiment we report them together to confirm this equivalence and separately when ablating α.

---

## 4. Experiments

### 4.1 Dataset

**Real TravisTorrent data is not yet available.** All results use a synthetic dataset (`data/raw/travistorrent_smoke.csv`) generated to match the TravisTorrent schema (Beller et al., MSR 2017). The dataset contains 1,150 builds across 2 synthetic projects:

| Project | Builds | Failure rate | History span |
| --- | ---: | ---: | ---: |
| `smoke/alpha` | 600 | 15% | >365 days |
| `smoke/beta` | 550 | 35% | >365 days |

Both projects pass the dataset filters (min_builds=500, min_history_days=365).

**Consequence for generalizability:** All findings are preliminary. The two-project dataset has no cross-project variance; bootstrap CIs collapse to point estimates. Results on real TravisTorrent (3.7M builds, ~1,000 projects) may differ substantially due to project heterogeneity, censoring, and the true CI outcome distribution.

### 4.2 Online Replay Setup

We use **online replay** (`evaluation/online_replay.py`): policies learn during trajectory traversal. At step *t*, the buffer releases matured rewards from prior steps, the policy updates its model, selects an action for the current commit, and queues a new pending reward.

Online replay is **not** unbiased counterfactual evaluation. Costs are computed as *cost(policy_action, logged_CI_outcome)* using the CI outcome as a proxy counterfactual. This is valid under the assumption that CI outcome is independent of the deployment action — an assumption that holds approximately in TravisTorrent because CI runs before any deployment decision is made. We state this explicitly as a threat to validity.

All logged actions in TravisTorrent are DEPLOY (the loader hardcodes `action=Action.DEPLOY` at line 165 of `data/loaders.py`). This means the cost for CANARY and BLOCK is computed counterfactually from the CI outcome, not from an observed partial rollout or hold. This is the primary source of evaluation bias.

**Configuration:**

| Parameter | Value |
| --- | --- |
| α | 1.0 |
| λ | 1.0 |
| Delay model | *k_t = max(1, ⌈build_duration_s / 60⌉)* |
| Cost matrix | Default (Table 1) |
| Seeds | 0–4 (identical results — deterministic delay) |
| flush_at_end | True (all pending rewards applied) |

### 4.3 Robustness Conditions

To test whether conclusions depend on the cost matrix or delay setting, we run five conditions:

| Condition | Key parameter change | Config file |
| --- | --- | --- |
| Default | — | `online_smoke.json` |
| High failure | deploy_failure: 10→20, canary_failure: 4→8 | `robustness_high_failure.json` |
| Low block | block_safe: 2→1, block_unknown: 2→1 | `robustness_low_block.json` |
| Short delay | delay_step_seconds: 60→120 (halves step-count delay) | `robustness_short_delay.json` |
| Long delay | delay_step_seconds: 60→30 (doubles step-count delay) | `robustness_long_delay.json` |

Runner: `experiments/run_robustness.py` with percentile bootstrap CI (10,000 resamples, seed 42) over seeds 0–4.

### 4.4 Ablation Conditions

To isolate model components we evaluate four variants of the cost-sensitive bandit:

| Variant | What changes | Implementation |
| --- | --- | --- |
| `no_drift` | No drift detector / model reset | `CostSensitiveBandit(reset_on_drift=False)` |
| `no_delay` | Immediate reward application (no buffer) | `ImmediateLinUCB` |
| `no_cost` | Binary reward (−1 failure, 0 otherwise) | `BinaryRewardBandit` |
| `full` | Delayed + cost reward + PageHinkley reset | `CostSensitiveBandit` + `PageHinkleyDetector` |

`no_drift` is the cleanest analog to standard LinUCB with cost-weighted reward and delayed updates. We use it as the reference variant when computing component deltas.

---

## 5. Results

### 5.1 Main Online Replay Results

**[Preliminary — synthetic 2-project dataset, CI collapsed to point estimates]**

Table 1: Cumulative cost comparison, default cost matrix, seeds 0–4.

| Policy | Steps | Updates | Censored | Cumul. Cost | Mean Cost/Step | Deploy% | Canary% | Block% |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `static_rules` | 1150 | 1150 | 0 | **1878.0** | 1.633 | 2.6% | 60.9% | 36.5% |
| `heuristic_score` | 1150 | 1150 | 0 | 2319.0 | 2.017 | 37.3% | 62.7% | 0.0% |
| `linucb` (α=1.0) | 1150 | 1150 | 0 | 1879.0 | 1.634 | 8.0% | 5.7% | 86.3% |
| `cost_sensitive_bandit` (α=1.0) | 1150 | 1150 | 0 | 1879.0 | 1.634 | 8.0% | 5.7% | 86.3% |

All 1,150 rewards matured and were applied (0 censored). `linucb` and `cost_sensitive_bandit` are identical at the same α and λ — this is mathematically expected (see §3.4) and confirmed by ‖b-vector difference‖₂ = 0.

**Per-project breakdown:**

| Project | Failure rate | Policy | Cumul. Cost | Deploy | Canary | Block |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| `smoke/alpha` | 15% | `static_rules` | 911.5 | 21 | 473 | 106 |
| `smoke/alpha` | 15% | `linucb` | 982.0 | 66 | 50 | 484 |
| `smoke/beta` | 35% | `static_rules` | 966.5 | 9 | 227 | 314 |
| `smoke/beta` | 35% | `linucb` | 897.0 | 26 | 15 | 509 |

On the low-failure project (smoke/alpha, 15%), the static rule's canary-heavy strategy (79% canary) is near-optimal: expected canary cost = 0.85×1.0 + 0.15×4.0 = 1.45/step, vs. block = 0.85×2.0 + 0.15×0.5 = 1.775/step. LinUCB over-blocks on low-failure data and pays the block_safe cost unnecessarily.

On the high-failure project (smoke/beta, 35%), LinUCB's block-heavy strategy (92.5% block) is cheaper: block_expected = 0.65×2.0 + 0.35×0.5 = 1.475/step vs. canary_expected = 0.65×1.0 + 0.35×4.0 = 2.05/step.

The aggregate tie (1878 vs 1879) is a smoke-data artifact where the two projects cancel out.

**Evidence that learning occurs.** The following step trace (smoke/alpha, LinUCB, seed 0) confirms actions change as rewards arrive:

```
step  0: DEPLOY   outcome=success  cost=0.0   updates=0  (uninformed prior)
step  1: DEPLOY   outcome=failure  cost=10.0  updates=0  (reward queued)
step  2: CANARY   outcome=success  cost=1.0   updates=1  (step 0 matures)
step 10: BLOCK    outcome=failure  cost=0.5   updates=1  (learning BLOCK arm)
step 14: BLOCK    outcome=success  cost=2.0   updates=1  (BLOCK converging)
```

The policy transitions from DEPLOY → CANARY → BLOCK within 10 steps, consistent with the cost-matrix optimum for a 15–35% failure environment.

### 5.2 Robustness Results

**[Preliminary — synthetic data, zero-width bootstrap CIs reflect deterministic delay model]**

Table 2: Mean cumulative cost across conditions (seeds 0–4, all identical due to deterministic delays).

**Cost matrix sweep (default delay, deploy_failure varies):**

| Policy | Default (df=10) | High failure (df=20) | Low block (bs=1) |
| --- | ---: | ---: | ---: |
| `static_rules` | 1878 | 2564 | 1584 |
| `heuristic_score` | 2319 | 4103 | 2319 |
| `linucb` | 1879 | **2079** | **1164** |
| `cost_sensitive_bandit` | 1879 | **2079** | **1164** |

**Delay sweep (default cost matrix):**

| Policy | Default (60s) | Short delay (120s) | Long delay (30s) |
| --- | ---: | ---: | ---: |
| `static_rules` | 1878 | 1878 | 1878 |
| `linucb` | 1879 | **1849** | 1897 |
| `cost_sensitive_bandit` | 1879 | **1849** | 1897 |

**Key observations:**

*Cost matrix.* Under high failure cost (deploy_failure=20), the bandit outperforms the static rule by **485 cost units (19%)** by shifting to 92.7% block and avoiding the 20-cost failure events. The static rule continues its fixed canary-heavy strategy (60.9% canary) and absorbs the doubled failure penalty. Under low block penalty (block_safe=1), the bandit further exploits cheaper blocking (89.3% block) and reduces cost by **420 units (27%)** relative to static. The static rule cannot adapt its thresholds.

*Delay.* Under short delay (delay_step_seconds=120, i.e., halved step-count delay), the bandit receives feedback twice as fast and converges to a block-heavy strategy (69.1% block), beating the static rule by 29 units. Under long delay (delay_step_seconds=30, doubled step-count), the bandit's uninformed-prior phase extends, it settles on a less decisive strategy (62.3% block), and costs 1,897 — 19 units more than the static rule (1.0% difference). This is consistent with theoretical results that delay inflates regret by O(√(d_max · T)) (Pike-Burke et al., 2018).

*Ranking stability.* Heuristic score is worst in all 5 conditions (never BLOCKs, absorbs full failure costs). The bandit/LinUCB ordering relative to static rules changes only at extreme delay (long delay), where static marginally wins by 1%.

### 5.3 Ablation Study

**[Preliminary — synthetic data]**

Table 3: Component ablation, default cost matrix and delay, seed 0 (all seeds identical).

| Variant | Cumul. Cost | Mean/Step | Deploy% | Canary% | Block% | Drift Resets |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `full` (delayed + cost + drift) | 2383.0 | 2.072 | 40.4% | 30.1% | 29.5% | 44 |
| `no_delay` (immediate updates) | **1857.5** | **1.615** | 1.7% | 19.8% | 78.4% | — |
| `no_cost` (binary reward) | 2469.0 | 2.147 | 41.4% | 22.1% | 36.5% | — |
| `no_drift` (cost + delayed, no reset) | 1879.0 | 1.634 | 8.0% | 5.7% | 86.3% | 0 |

We take `no_drift` as the reference variant (standard cost-weighted delayed bandit without drift reset) and compute deltas:

| Component removed | Δ Cumul. Cost vs. no_drift | Δ% |
| --- | ---: | ---: |
| Cost weighting → binary (no_cost) | +590.0 | +31.4% |
| Delayed buffer → immediate (no_delay) | −21.5 | −1.1% |
| No drift reset → drift reset (full) | +504.0 | +26.8% |

**Cost weighting (+31%).** Binary reward (−1/0) loses the asymmetry between action–outcome pairs. The DEPLOY arm receives −1 updates for failures whether or not the policy caused the failure, and 0 for successes regardless of the opportunity cost of blocking. The block arm receives −1 updates for failures it *correctly* avoided, producing misleading gradient information. The 31% cost increase is the strongest evidence that cost-sensitive reward design matters.

**Delay removal (−1.1%).** Immediate updates allow the policy to converge to the block-heavy strategy faster, reducing cost by 21.5 units. The negative sign means *removing* the delay *reduces* cost — delay is a structural cost the policy must pay in deployment settings. The effect is small (1.1%) because the delay model in this dataset averages ~10 steps per build, and 1,150 total updates are enough for the policy to converge despite the lag.

**Drift reset on stationary data (+27%).** With PageHinkley threshold *λ_PH = 50*, the detector fires 44 false alarms across 1,150 stationary steps: 17 resets on smoke/alpha and 27 on smoke/beta. Each reset discards all learned weights; the post-reset policy reverts to DEPLOY-heavy behavior (maximum UCB on uninformed priors) until it re-learns the block-heavy strategy. This explains the 40.4% deploy rate in `full` vs. 8.0% in `no_drift`. The detector is too sensitive for this dataset's smooth cost distribution. **We exclude the `full` model from the main claims. The drift component requires calibration against non-stationary data (RQ4), which is not yet available.**

---

## 6. Key Findings

We separate findings by evidential quality.

### 6.1 Reliable Findings (mechanism verification, not generalization claims)

**F1: Cost weighting is the most important model component (+31% cost when removed).**
Replacing the asymmetric cost signal with binary (−1/0) feedback substantially degrades performance on both projects. This is a mechanism finding that holds because the cost matrix is deliberately asymmetric (factor 20× between deploy+failure and block+would_fail). Any setting where failure is significantly more expensive than unnecessary blocking will exhibit this effect.

**F2: The delayed-feedback buffer enforces correctness, not performance.**
The buffer prevents the policy from using future reward information at decision time (the fundamental invariant of delayed-feedback bandits, formalized in Joulani et al., 2013). The ablation confirms that removing the buffer improves performance slightly (1.1%), as expected — immediate feedback is faster. The buffer's value is validity, not accuracy.

**F3: Page-Hinkley fires false alarms on stationary data at λ_PH = 50.**
44 false alarms over 1,150 steps on data with no true distribution shift. This is expected behavior for drift detectors: they trade false-alarm rate for detection speed. The current threshold is too sensitive and requires calibration.

### 6.2 Preliminary Findings (synthetic 2-project data, cannot generalize)

**F4: Bandit advantage scales with failure cost severity.**
At deploy_failure=20 (double the default), the bandit outperforms the static rule by 19% by learning a block-heavy strategy. At deploy_failure=10, they tie. This qualitative pattern — learner adapts, static rule cannot — is expected from first principles and confirmed empirically on smoke data.

**F5: Under severe delay, static rules marginally outperform a learning bandit.**
At delay_step_seconds=30 (doubled step-count delay), the bandit costs 1% more than static rules. The effect is small but directionally consistent with O(√(d_max · T)) delay-induced regret inflation. Whether this survives on real data with stochastic delays and more projects is unknown.

**F6: On the high-failure project (35% failure rate), the learning bandit outperforms the static rule.**
On smoke/beta, LinUCB achieves 897.0 vs. 966.5 for static rules by shifting to a block-heavy strategy. On smoke/alpha (15% failure), static rules win because the canary-heavy strategy is near-optimal at low failure rates. This suggests the bandit's advantage is most pronounced in high-failure-rate environments — a reasonable hypothesis that requires real data to confirm.

---

## 7. Limitations

### 7.1 No Logging Propensities — No Unbiased Counterfactual Evaluation

TravisTorrent records CI build outcomes, not deployment decisions. Every record is assigned `action = DEPLOY` by the data loader. There is no logging policy and no associated propensity scores.

**Consequence:** Inverse propensity scoring (IPS) cannot be applied. With `logged_propensity = 1.0` for all steps, the IPS weight is identically 1.0, reducing the estimator to the direct method. This estimator is unbiased only if the true logging policy selected DEPLOY with probability 1.0 on every commit — an assumption that is unverifiable and almost certainly false for human-driven CI pipelines.

**What the experiments do and do not measure:** The online replay measures *how costly the policy's decisions are given the observed CI outcomes as counterfactuals.* It does not measure *whether those decisions would produce the same outcomes in a real deployment.* The two quantities are equal only if CI outcome is a perfect proxy for deployment outcome — another unverifiable assumption.

**Required fix:** Either (a) collect data under a known stochastic logging policy (e.g. randomized deployment decisions), or (b) evaluate on a synthetic environment where the true propensities and outcomes are generated from the controlled data-generating process.

### 7.2 Online Replay is Biased

Online replay uses `cost(policy_action, logged_CI_outcome)` to evaluate CANARY and BLOCK decisions against an outcome that was generated under a DEPLOY decision. This is a counterfactual evaluation with an untested proxy:

- **CANARY:** The cost is computed as if the CI outcome predicts the partial-rollout outcome. In reality, a canary might fail on a different traffic slice than CI tests.
- **BLOCK:** The cost is computed as if CI outcome tells us what *would have* happened had the commit been deployed. This is the fundamental counterfactual — it is unobservable.

The evaluation is internally consistent (the same assumption applies to all policies equally) but is not an unbiased estimate of real-world deployment cost. We report it as a learning-dynamics verification, not a causal cost estimate.

### 7.3 Drift Detection Results Excluded from Claims

As documented in §5.3, the `full` model (with PageHinkley drift reset) performs **worse** than `no_drift` on stationary data due to 44 false-alarm resets. The drift adaptation mechanism is implemented and mechanistically correct — it correctly resets the model when triggered — but the trigger threshold (*λ_PH = 50*) is miscalibrated for this dataset.

**We do not claim that drift adaptation reduces operational cost.** The claim that cost-sensitive bandits with drift adaptation outperform non-adaptive policies on non-stationary trajectories is a hypothesis that requires the RQ4 synthetic drift experiment (abrupt-shift schedule on the synthetic environment). That experiment has not been run.

### 7.4 Synthetic 2-Project Dataset

All results are from a hand-crafted 1,150-row synthetic dataset with two projects at fixed failure rates (15% and 35%). Real TravisTorrent contains ~3.7M builds across ~1,000 projects with heterogeneous failure rates, project lifecycles, and team sizes. The smoke dataset produces zero bootstrap CI width (deterministic delay model) and therefore provides no statistical uncertainty estimates.

Claims cannot be generalized beyond the smoke dataset until replicated on real TravisTorrent data.

### 7.5 LinUCB and CostSensitiveBandit are Identical at Same Hyperparameters

At identical α, λ, and reward signal, the two policies are mathematically the same algorithm. The distinction between them is only observable when α differs, when the drift detector is active, or when future enhancements (Thompson sampling, windowed replay) are applied. The current experiments do not distinguish the two implementations.

### 7.6 Missing Baselines

The evaluation protocol (`docs/evaluation-protocol.md`) specifies five policies: static rules, heuristic score, LinUCB, cost-sensitive bandit, and Thompson sampling. Thompson sampling is not yet implemented. Results therefore cannot claim to show optimality within the specified comparison set.

---

## 8. Conclusion

We formulated software deployment control as a cost-sensitive contextual bandit with delayed rewards, replacing the classification framing of just-in-time defect prediction with a sequential cost-minimization objective. The key insight is that the relevant question is not "will this commit fail?" but "what is the cost of each deployment action given our uncertainty?"

Our CostSensitiveBandit implements disjoint LinUCB with a pending-reward buffer that enforces the delayed-feedback invariant and an asymmetric cost signal that rewards the policy for correctly weighting the consequences of different action–outcome pairs.

The ablation study shows that cost weighting is the most important model component: replacing it with binary feedback increases cumulative cost by 31%. Robustness experiments show that the bandit's advantage over a fixed static rule grows as failure costs increase or as blocking becomes cheaper — both conditions where an adaptive policy can exploit the cost structure that a static threshold cannot.

**The decision-making framing matters.** Static rules achieve competitive performance in the default cost setting because their fixed thresholds happen to match the average failure rate of the synthetic dataset. As the cost matrix shifts away from that implicit assumption, the bandit's adaptivity produces measurable gains. This supports the core claim: deployment control is better modeled as sequential decision-making than as classification, even when the underlying model class (linear regression) is the same.

**What remains.** The two most important open items before any submission claims can be made are: (1) results on real multi-project TravisTorrent data with stochastic delays and bootstrap CIs of nonzero width, and (2) RQ4 drift experiments showing that the drift adaptation mechanism reduces post-shift regret. Both require infrastructure that is implemented but not yet exercised on sufficient data.

---

## References

- Beller, M., et al. "TravisTorrent: Synthesizing Travis CI and GitHub for Full-Stack Research." MSR 2017.
- Chu, W., et al. "Contextual Bandits with Linear Payoff Functions." AISTATS 2011.
- Gama, J., et al. "A Survey on Concept Drift Adaptation." ACM CSUR 2014.
- Joachims, T., et al. "Deep Learning with Logged Bandit Feedback." ICLR 2018.
- Joulani, P., et al. "Online Learning under Delayed Feedback." ICML 2013.
- Kamei, Y., et al. "A Large-Scale Empirical Study of Just-in-Time Quality Assurance." TSE 2013.
- Li, L., et al. "A Contextual-Bandit Approach to Personalized News Article Recommendation." WWW 2010.
- McIntosh, S., Kamei, Y. "Are Fix-Inducing Changes a Moving Target?" EMSE 2018.
- Mouss, H., et al. "Test of Page-Hinckley, an Approach for Fault Detection in an Agro-Alimentary Production System." MED 2004.
- Pike-Burke, C., et al. "Bandits with Delayed, Aggregated Anonymous Feedback." ICML 2018.
- Weyns, D., et al. SEAMS literature on MAPE-K and learning-based adaptation (multiple years).

---

## Appendix A: Validity Classification

| Claim | Status | Evidence | Condition for promotion |
| --- | --- | --- | --- |
| Cost weighting is important (+31% without it) | **Preliminary** | Smoke dataset, 2 projects | Replicate on ≥10 real projects |
| Bandit advantage grows with failure cost | **Preliminary** | Smoke dataset | Replicate on real data |
| Delay reduces learning speed | **Preliminary** | Smoke dataset, small effect | Stochastic delay experiment, real data |
| Buffer enforces delayed-feedback correctness | **Reliable** | Code + 174 passing tests | — |
| PageHinkley fires false alarms at λ=50 on stationary data | **Reliable** | Expected detector behavior | — |
| Drift adaptation reduces cost on non-stationary data | **Not yet evaluated** | RQ4 not run | Implement synthetic drift experiment |
| Bandit outperforms static rules in general | **Not established** | Tied on default cost, smoke data | Real data, multiple cost configs |
| Heuristic score is consistently worst | **Preliminary** | All 5 conditions, smoke data | Real data |

---

## Appendix B: Experiment Reproducibility

All experiment code is in the repository. To reproduce:

```bash
# Phase 20: main online replay
for seed in 0 1 2 3 4; do
    python -m experiments.run_bandits \
        --config experiments/configs/online_smoke.json \
        --seed $seed
done

# Phase 21: robustness
python -m experiments.run_robustness \
    --configs experiments/configs/online_smoke.json \
              experiments/configs/robustness_high_failure.json \
              experiments/configs/robustness_low_block.json \
              experiments/configs/robustness_short_delay.json \
              experiments/configs/robustness_long_delay.json \
    --seeds 0 1 2 3 4

# Phase 22: ablation
for seed in 0 1 2 3 4; do
    python -m experiments.run_ablations --seed $seed
done
```

Results are written to `experiments/results/` (gitignored). The smoke dataset is at `data/raw/travistorrent_smoke.csv` (gitignored). All policy implementations, evaluation runners, and config files are tracked in version control.
