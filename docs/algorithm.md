# Algorithm

## Name

Cost-Sensitive Delayed LinUCB for Continuous Deployment

## Problem Setting

At each deployment decision step `t`, the controller observes a context `x_t` containing only pre-action signals such as churn, test counts, build duration, dependency changes, risky paths, author experience, and recent failure rate.

The controller selects one action:

```text
A = {deploy, canary, block}
```

The reward is not observed immediately. A delayed reward arrives after `k_t` steps:

```text
r_{t+k_t} = -cost(a_t, outcome_{t+k_t})
```

The objective is to minimize cumulative operational cost, not maximize prediction accuracy.

## Current Implementation

| Concept | File |
| --- | --- |
| Context/action/reward schema | `data/schemas.py` |
| Feature encoder | `policies/base.py::FeatureEncoder` |
| Cost matrix | `rewards/cost_model.py::CostConfig` |
| Delayed reward gate | `delayed/buffer.py::PendingRewardBuffer` |
| Baseline LinUCB | `policies/linucb.py` |
| Contribution | `policies/cost_sensitive_bandit.py` |

## Model

The policy keeps one disjoint linear model per action.

For each action `a`:

```text
A_a = lambda * I
b_a = 0
theta_a = A_a^{-1} b_a
```

Action score:

```text
score(a) = theta_a^T x + alpha * sqrt(x^T A_a^{-1} x)
```

The action with the highest optimistic reward is selected. Because reward is negative cost, the model learns to prefer actions with lower expected operational cost.

## Delayed Update Rule

At action time, the policy may schedule a pending reward:

```text
buffer.add(action_id, context, action, outcome, current_step, cost, delay)
```

No model update occurs when the action is taken. Updates occur only when the buffer releases a matured reward:

```text
for pending in buffer.pop_available(step):
    if not pending.reward.censored:
        update(pending.context, pending.action, pending.reward)
```

The update is:

```text
x = encode(context)
r = -reward.cost
A_action = A_action + x x^T
b_action = b_action + r x
```

This preserves the delayed-feedback invariant: policy learning cannot use future reward information before it becomes observable.

## Drift Adaptation

The policy optionally accepts a drift detector implementing `drift/detectors.py::DriftDetector`.

On each matured reward:

```text
if detector.update(cost):
    reset model state
    detector.reset()
```

The current adaptation is intentionally simple:

| Mode | Behavior |
| --- | --- |
| `reset_on_drift = true` | Reinitialize all per-arm matrices |
| `reset_on_drift = false` | Continue learning without model reset |

Windowed replay or ensemble adaptation is out of scope for Phase 16.

## Pseudocode

```text
Input:
  action set A = {deploy, canary, block}
  feature encoder phi(x)
  exploration alpha
  ridge regularization lambda
  delayed reward buffer B
  optional drift detector D

Initialize:
  for each action a in A:
      A_a <- lambda * I
      b_a <- 0

For each decision step t:
  release all matured rewards from B

  for each matured reward (x_i, a_i, r_i):
      if reward is not censored:
          z <- phi(x_i)
          cost_i <- observed operational cost
          reward_i <- -cost_i
          A_{a_i} <- A_{a_i} + z z^T
          b_{a_i} <- b_{a_i} + reward_i z

          if drift detector exists and D.update(cost_i) is true:
              reset all A_a, b_a
              D.reset()

  observe context x_t

  for each action a in A:
      z <- phi(x_t)
      theta_a <- inverse(A_a) b_a
      ucb_a <- theta_a^T z + alpha * sqrt(z^T inverse(A_a) z)

  choose action a_t <- argmax_a ucb_a
  execute a_t
  schedule delayed reward in B
```

## Tracked Quantities

The contribution exposes runtime counters through `CostSensitiveBandit.stats`.

| Quantity | Meaning |
| --- | --- |
| `cumulative_cost` | Sum of observed finite costs |
| `cumulative_regret` | Placeholder for oracle-regret integration |
| `action_counts` | Counts for deploy/canary/block |
| `delayed_updates_applied` | Number of matured rewards used for learning |
| `drift_resets` | Number of model resets triggered by drift |
| `pending_rewards` | Rewards still waiting in the buffer |

## What This Algorithm Is Not

This phase does not introduce:

- neural bandits
- causal inference
- doubly robust learning
- full reinforcement learning
- multi-step deployment planning
- production rollback automation

The contribution is deliberately narrow: cost-sensitive contextual bandit learning with delayed reward gating and simple drift reset.

## Expected Advantages

| Challenge | Why the algorithm should help |
| --- | --- |
| Asymmetric deployment costs | Learns from `-cost`, so failed deploys are penalized more than blocked safe changes |
| Delayed feedback | Prevents future reward leakage through the pending reward buffer |
| Canary tradeoff | Can learn when partial rollout has lower expected cost than deploy/block |
| Non-stationarity | Drift reset refreshes stale action-value estimates |
| Static CI/CD limitations | Learns from observed outcomes instead of fixed thresholds |

## Required Evidence

The algorithm is supported only if results show:

1. Lower cumulative operational cost than static rules.
2. Lower or comparable cumulative operational cost than LinUCB.
3. Stable behavior across at least the default and high-failure cost matrices.
4. Correct delayed-update behavior under stochastic delay.
5. Better post-drift recovery with reset adaptation than without adaptation.
