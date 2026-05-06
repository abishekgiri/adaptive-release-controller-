# Problem Formulation: Cost-Sensitive Delayed Contextual Bandits for Continuous Deployment

**Status: LOCKED. All five open questions resolved. Do not edit without updating docs/algorithm.md.**

---

## 1. Setting

At each discrete step `t`, a passing CI build triggers a deployment decision. The outcome arrives `k_t` steps later and may be censored if the observation window closes first. The goal is to minimise cumulative operational cost over `T` decisions.

---

## 2. Bandit Tuple

| Symbol | Definition |
|--------|-----------|
| `X` | `Context` (frozen dataclass) — pre-action observables only; no outcome, risk score, or post-deploy signal. Extracted by `features.extractor.extract_context`. |
| `A` | `Action` enum — `{deploy, canary, block}` (§3). |
| `R` | Reward `r_{t+k_t} = −cost(a_t, outcome_{t+k_t})`; observed after delay `k_t`; may be censored. |
| `π` | Policy `π : X → Δ(A)` — abstract class in `policies/`. |
| `T` | Horizon — number of decisions in a trajectory. |

Extends the disjoint LinUCB formulation of Li et al. (2010) with asymmetric costs (§4) and delayed feedback (§5).

---

## 3. Action Space

| Action | `Action` enum value | Semantics |
|--------|---------------------|-|
| Full rollout | `Action.DEPLOY` | Immediate production promotion. |
| Partial rollout | `Action.CANARY` | 5–10% of traffic; outcome observed after canary window. Reduces blast radius at throughput cost. |
| Hold | `Action.BLOCK` | No rollout; change queued for re-evaluation. |

---

## 4. Cost Matrix

Reward `r = −cost`. Costs are **project-wide** for all headline results; per-project cost matrix is a sensitivity ablation in §8 only (a per-project matrix introduces ≈1 free parameter per project, undermining credibility of any positive result).

| Action | Outcome | `CostConfig` field | Default |
|--------|---------|-------------------|---------|
| `deploy` | `success` | `deploy_success` | 0.0 |
| `deploy` | `failure` | `deploy_failure` | 10.0 |
| `canary` | `success` | `canary_success` | 1.0 |
| `canary` | `failure` | `canary_failure` | 4.0 |
| `block` | `success` (would have succeeded) | `block_safe` | 2.0 |
| `block` | `failure` (would have failed) | `block_bad` | 0.5 |
| `block` | `blocked` (counterfactual unknown) | `block_unknown` | 2.0 |

All values configurable via `rewards.cost_model.CostConfig` (frozen dataclass). Asymmetry ratio: `deploy_failure / block_bad = 20×`. Cost is `NaN` when outcome is `CENSORED` (§5).

---

## 5. Delayed and Censored Feedback

Reward for action `a_t` is not available at time `t`. Two delay models are used:

**Synthetic environment** (`environment/synthetic.py`): delay `k_t ~ Geom(p)`, `p = 0.3`. Choice follows Vernade, Cappé & Perchet (2017); geometric delay is the simplest model under which IPS/DR unbiasedness is analytically verifiable (`tests/test_replay_eval.py`).

**Real data replay** (`evaluation/replay_eval.py`): empirical hazard-rate model fit on TravisTorrent build durations. The fit is documented in `data/README.md`.

In both cases, if the outcome is not observed by step `t + max_delay`, the reward is marked **censored** (`Outcome.CENSORED`) and excluded from policy updates. All update paths go through `delayed/buffer.py`; no policy reads reward at decision time (Joulani et al., 2013).

---

## 6. Non-Stationarity

The mapping `P(outcome | context, hidden_state)` is non-stationary. Three drift modes are evaluated (Phase G):

| Mode | Description | Role |
|------|-------------|------|
| `none` | Stationary hidden state | Baseline; IPS/DR unbiasedness check. |
| `abrupt` | Single discontinuity at trajectory midpoint | Primary drift result. |
| `gradual` | Linear interpolation of hidden-state parameters across trajectory | Stress test. |

All three modes run with ≥ 30 seeds. If the method handles `abrupt` but not `gradual`, that is a reportable result (§11 Limitations), not a hidden weakness. Drift detection uses `drift/detectors.py` (ADWIN, Page-Hinkley); adaptation via `drift/adapt.py` (Gama et al., 2014).

---

## 7. Regret

**Oracle policy** `π*`: at each step `t`, selects the action minimising expected cost given full knowledge of the hidden-state random variable `H_t`. Let `O_t` denote the outcome random variable induced by `H_t` and the deployment process:

```
a*_t = argmin_{a ∈ A} E[cost(a, O_t) | H_t]
```

Under the default cost matrix, the oracle never selects `canary` — `canary` is dominated when the hidden state is known (`deploy_success=0 < canary_success=1`; `block_bad=0.5 < canary_failure=4`). The value of `canary` for a bandit comes from uncertainty reduction, which the oracle does not need (Lattimore & Szepesvári, 2020, §1).

**Cumulative regret** (primary metric):

```
Regret(T) = Σ_{t=1}^{T} [ cost(a_t, O_t) − cost(a*_t, O_t) ]
```

The realized empirical regret uses the observed sample of `O_t` once the delayed reward is available. Reported with bootstrap 95% CIs over ≥ 30 seeds; pairwise comparisons use paired bootstrap with Holm–Bonferroni correction (Evaluation Metrics, CLAUDE.md).

**Best-in-hindsight regret** (diagnostic): `Regret_bih(T) = Σ_t cost(a_t, O_t) − min_{a} Σ_t cost(a, O_t)`.

---

## 8. Threats to Validity

**Proxy-label assumption (primary threat):** TravisTorrent records CI build failure, not production deployment failure. We treat build failure as a proxy for deployment failure. This assumption is unverifiable without post-deploy telemetry and must be stated explicitly in §10. **Feature-outcome separation:** `HiddenState` and `Context` are disjoint frozen dataclasses; `tests/test_environment.py` asserts `set(HiddenState.__annotations__) ∩ set(Context.__annotations__) == ∅` — enforcing that the policy cannot infer the outcome from the context by construction. **Offline evaluation bias:** all logged actions are `deploy`; CANARY and BLOCK costs are counterfactual. IPS weights are clipped at 20× to bound variance. Absolute cost numbers are simulation artefacts; only relative rankings are reportable. **Sensitivity to cost matrix:** `run_cost_sweep.py` sweeps the cost ratio; conclusions must hold across a reasonable range. **Generalization:** all claims are scoped to the TravisTorrent project mix; no claim extends to projects outside the dataset without replication.
