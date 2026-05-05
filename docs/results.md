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

### Finding: All drift detectors are unimplemented; no drift evidence exists

`drift/detectors.py` contains three stub classes (ADWIN, Page-Hinkley, DDM); all raise
`NotImplementedError`. No drift experiment (RQ4) has been run.

The evaluation protocol specifies:

| Experiment | Drift type | Status |
| --- | --- | --- |
| `RQ4-drift-none` | Stationary | ✗ Not run |
| `RQ4-drift-abrupt-reset` | Abrupt, policy reset on detection | ✗ Not run |
| `RQ4-drift-abrupt-no-reset` | Abrupt, no reset | ✗ Not run |

**Cannot claim anything about drift recovery.** This is one of the three primary contributions
stated in `docs/algorithm.md`. The evidence for it does not yet exist.

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
| `PendingRewardBuffer` delay logic | Correct | 146 tests pass |
| `CostConfig` cost matrix | Correct | Unit tests, code review |
| `LinUCBPolicy` parameter updates (A, b) | Correct | Matrix update tests |
| `evaluate_ips` cost accumulation | Correct | Unit tests with known propensities |
| `TravisTorrentLoader` context extraction | Correct | Schema and integration tests |
| `StaticRulesPolicy` determinism and thresholds | Correct | Decision-semantics test suite |
| Pipeline end-to-end (load → evaluate → write JSON) | Runs without error | Phase 17 smoke run |

### Not Yet Reliable (cannot support paper claims)

| Claim | Why it fails | Required fix |
| --- | --- | --- |
| Cost-sensitive bandit outperforms static rules | Bandit not trained; coverage confound | Online replay with policy updates |
| LinUCB and cost-sensitive bandit differ | Identical outputs from fresh priors | Same as above |
| Results robust to cost matrix | Only one matrix evaluated | Run RQ2 configs |
| Results robust to delay | No delay ablation | Run RQ3 with synthetic env |
| Drift adaptation reduces post-drift cost | No detector implemented; no drift run | Implement 1 detector; run RQ4 |
| IPS estimates are valid counterfactuals | Logging propensities unknown | Use synthetic env with known propensity |

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
