<!--
  paper/adaptive-deployment-control.md
  ------------------------------------------------------------------
  This is a SCAFFOLD, not a finished paper.

  Every section below contains:
    - a short prose description of what belongs in that section,
    - explicit dependencies (what code / experiment / dataset must
      exist before the section can be honestly written),
    - placeholder tables and figures referenced by filename in
      experiments/results/.

  Do NOT fabricate numbers. Every TODO must be filled by an actual
  experiment whose script lives in experiments/ and whose output
  lives in experiments/results/<config>/<seed>/. If a number cannot
  be traced to a config + seed, it does not go in the paper.

  Target venue class: SEAMS, ICSE-SEIP, MSR, ASE, or an ML venue
  if the algorithmic contribution is strong enough.
-->

# Adaptive Deployment Control as a Cost-Sensitive Delayed Contextual Bandit

**Authors:** TODO
**Affiliation:** TODO
**Status:** Draft — DO NOT CIRCULATE

---

## Abstract

> *Write last. ~200 words. Four moves: (1) one sentence on continuous
> deployment as a sequential decision problem; (2) one sentence on the
> gap in prior work (classification framings, immediate labels, no cost
> asymmetry, no drift handling); (3) one sentence on the contribution
> (cost-sensitive contextual bandit with delayed-reward buffer and drift
> adaptation); (4) two-to-three sentences on the empirical result —
> dataset, baselines beaten, magnitude of cost reduction, statistical
> rigor (e.g., "30 seeds, paired bootstrap, p<0.01"). End with the
> reproducibility statement.*

TODO — write after Results section is locked.

---

## 1. Introduction

> *Three to four paragraphs. Should answer: why does this problem matter,
> why is the standard framing wrong, what do we contribute, what do we
> show?*

### 1.1 Motivation

TODO — describe the gap between how continuous deployment decisions are
*made* in industry (static gates, post-hoc rollback) and how they should
be *modeled* (sequential decisions under uncertainty with delayed
feedback). Cite one or two real outage post-mortems if available.

### 1.2 The Mismatch with Prior Work

TODO — summarize, in two paragraphs, why prior just-in-time defect
prediction work (Kamei 2013 and successors) does not address this
problem despite appearances:

1. It treats the problem as classification, not control.
2. It assumes immediate, fully-observed labels.
3. It optimizes accuracy/F1, not operational cost.
4. It assumes stationary distributions.

### 1.3 Contributions

TODO — once results are in, list 3–4 contributions. Tentative form:

1. A formal contextual-bandit formulation of continuous deployment with
   the action space {deploy, canary, block}, an explicit cost matrix,
   and delayed possibly-censored rewards. (See §3.)
2. A cost-sensitive contextual bandit algorithm with a delayed-reward
   buffer and drift-aware adaptation. (See §4.)
3. An empirical comparison against static-rule, heuristic, and
   offline-classifier baselines on TODO-DATASET, with bootstrap CIs
   over ≥30 seeds. (See §6.)
4. A drift sensitivity analysis showing TODO. (See §7.)

### 1.4 Roadmap

TODO — one short paragraph mapping the rest of the paper to sections.

---

## 2. Related Work

> *Two pages. Organized by theme, not by paper. Every paragraph should
> end with what the cited work did NOT solve, so the contribution is
> clearly carved out.*

### 2.1 Just-In-Time Defect Prediction

TODO — Kamei et al. 2013 (foundational JIT defect prediction);
McIntosh & Kamei 2018 (limits and replication). Frame both as
classification on stationary data; note absence of action/cost framing.

### 2.2 Contextual Bandits

TODO — Li et al. 2010 (LinUCB); Chu et al. 2011 (linear contextual
bandits with regret bounds); Agarwal et al. 2014 (Taming the Monster).
Frame the gap: applied work is news/ads, not deployment.

### 2.3 Delayed-Feedback Bandits

TODO — Joulani, Györfi & György 2013 (online learning under delay);
Vernade et al. 2017 (stochastic bandits with delayed feedback);
Pike-Burke et al. 2018 (bandits with delayed, aggregated anonymous
feedback). Frame the gap: not applied to deployment, no cost asymmetry.

### 2.4 Off-Policy / Counterfactual Evaluation

TODO — Joachims, Swaminathan & de Rijke 2018 (deep learning with logged
bandit feedback); Dudik et al. 2014 (doubly robust). Frame the gap:
methodology only; no deployment-domain application.

### 2.5 Concept Drift

TODO — Gama et al. 2014 (drift survey); Bifet & Gavaldà 2007 (ADWIN);
Page-Hinkley as cited in the survey. Frame: detection is mature; drift-
aware policy adaptation in CD is unexplored.

### 2.6 Self-Adaptive Software / SEAMS

TODO — recent SEAMS proceedings (last 5 years) on self-adaptation in
DevOps / CI-CD pipelines. Frame: most are MAPE-K architectures without
formal decision-theoretic grounding; this work supplies that grounding.

### 2.7 Positioning

TODO — one paragraph that says: this work is the intersection of (2.2),
(2.3), (2.5), and the deployment domain. No prior work occupies all
four corners simultaneously.

---

## 3. Problem Formulation

> *This section must match `docs/problem-formulation.md` in the repo
> word-for-word on definitions. The doc is the source of truth; this
> section is the paper-facing version of it.*

### 3.1 Setting

TODO — describe the continuous-deployment decision loop. At each
discrete time `t`:

1. A change `c_t` arrives.
2. The agent observes a context `x_t ∈ X`.
3. The agent selects an action `a_t ∈ A = {deploy, canary, block}`.
4. After delay `k_t`, a reward `r_{t+k_t} ∈ R` is observed (possibly
   censored).
5. The agent updates its policy.

### 3.2 Bandit Tuple `(X, A, R, π, T)`

TODO — define each element formally:

- `X`: the context space — list every feature in the canonical
  feature_vector here. No outcomes, no decisions, no leakage.
- `A`: `{deploy, canary, block}`.
- `R`: a real-valued cost (lower is better) drawn from a hidden state
  conditional on `(a_t, hidden_state_t)`. Not a function of `x_t`
  directly. (Cite §3.4 cost matrix.)
- `π`: a policy mapping `X` to a distribution over `A`.
- `T`: time horizon for evaluation, in number of decisions.

### 3.3 Delayed and Censored Feedback

TODO — formalize delay distribution `k_t ~ D` (e.g., geometric with
parameter `p` or hazard-rate model derived from CI/CD post-deploy
windows). Define censoring: a reward is censored if it has not arrived
by the end of the trajectory.

### 3.4 Cost Matrix

TODO — reproduce the cost matrix from `rewards/cost_model.py`. Default
values:

| outcome \ action     | deploy | canary | block |
|----------------------|--------|--------|-------|
| underlying-safe      |   0    |   1    |   2   |
| underlying-unsafe    |  10    |   4    |   0   |

TODO — defend each cell value with one sentence (industry-cost
intuition or sweep-justification).

### 3.5 Non-Stationarity

TODO — formalize the drift model. Either piecewise-stationary segments
or a smooth drift on the hidden-state distribution. State the assumption
the algorithm makes (e.g., bounded drift rate).

### 3.6 Regret Definition

TODO — primary regret is cumulative cost difference vs. an oracle
policy that knows the hidden-state random variable `H_t`. Let `O_t`
denote the outcome random variable induced by `H_t` and the deployment
process. Diagnostic regret is vs. best-in-hindsight constant policy.
State formally:

`a*_t = argmin_{a in A} E[cost(a, O_t) | H_t]`

`Regret(T) = sum_{t=1..T} cost(a_t, O_t) - sum_{t=1..T} cost(a*_t, O_t)`

### 3.7 Threats to Validity (forward reference)

TODO — single paragraph pointing to §10 for the full treatment, but
flag the two structural threats up front: (a) hidden-state /
observable-context contamination in synthetic environments;
(b) propensity estimation error in offline replay.

---

## 4. Algorithm

> *This is the contribution section. It must be precise enough that
> someone outside our group can reimplement the algorithm from the
> paper alone.*

### 4.1 Overview

TODO — three to four sentences: cost-sensitive contextual bandit, with
a pending-reward buffer for delayed credit assignment, and a drift
detector that triggers a policy reset / windowed retrain.

### 4.2 Pseudocode

TODO — full pseudocode block. Mandatory. Should reference:

- `policies/cost_sensitive_bandit.py`
- `delayed/buffer.py`
- `drift/detectors.py`
- `drift/adapt.py`

```
Algorithm 1: Cost-Sensitive Delayed Contextual Bandit (CS-DCB)

Input: cost matrix C, delay distribution D, drift detector Δ
Initialize: posterior parameters θ, pending-reward buffer B, drift state s

for t = 1, 2, ..., T do
    Observe context x_t
    Sample / select action a_t = π(x_t; θ)        # bandit choice
    Append (t, x_t, a_t) to buffer B
    Receive any rewards (s, r_s) due at time t
        for each (s, r_s) received:
            Update θ with (x_s, a_s, r_s) under cost-sensitive loss
            Feed cost(a_s, outcome_s) to drift detector Δ
    if Δ.detected_drift() then
        Apply adaptation strategy (reset / windowed retrain)
end for
```

### 4.3 Cost-Sensitive Update Rule

TODO — derive or state the update rule. If it is a known one (e.g.,
LinUCB with cost-weighted residuals), cite. If it is novel, prove the
relevant property (regret bound, calibration, or whatever the
contribution claims).

### 4.4 Delayed-Reward Handling

TODO — describe the buffer: keyed by `action_id`, holds `(context,
action, propensity, decided_at)`, releases on reward arrival. Censoring
strategy: `delayed/imputation.py` describes options; the chosen
strategy goes here.

### 4.5 Drift Adaptation

TODO — describe which detector (ADWIN / Page-Hinkley / DDM) and which
adaptation (full reset, sliding window, exponential forgetting). State
hyperparameters and how they were chosen (sensitivity analysis lives
in §7).

### 4.6 Computational Complexity

TODO — per-step time and memory complexity in terms of context
dimension `d`, action set size `|A| = 3`, and buffer size.

---

## 5. Experimental Setup

### 5.1 Datasets

TODO — describe the chosen real dataset (TravisTorrent / Apachejit /
GH-Actions export). Include:

- Source URL.
- Version / hash.
- Number of trajectories, decisions per trajectory.
- Failure rate / class balance.
- Project breakdown.
- License.

Cross-reference `data/README.md`.

### 5.2 Synthetic Stress-Test Environment

TODO — describe `environment/synthetic.py`. Critical: explicitly state
that hidden state and observable context share no fields by
construction, and that the `tests/test_environment.py` property test
verifies this. This sentence must appear somewhere in the paper.

### 5.3 Baselines

TODO — list baselines in order of strength:

1. `static-rules` — ports `experiments/baseline.py` policy.
2. `heuristic-score` — fixed-threshold port of the retired risk score.
3. `offline-classifier` — logistic regression / gradient boosting
   trained on the first N% of trajectories.
4. `linucb` — LinUCB (Li et al. 2010).
5. `thompson` — Thompson Sampling, Bayesian linear model.
6. `epsilon-greedy` — sanity-check baseline.
7. `cs-dcb (ours)` — the contribution.

### 5.4 Evaluation Protocol

TODO — must match `docs/evaluation-protocol.md` exactly. State:

- ≥ 30 seeds per configuration.
- Bootstrap 95% CIs on all headline numbers.
- Pairwise paired-bootstrap tests with Holm–Bonferroni correction.
- Off-policy estimator (IPS, then DR) for evaluation on logged data.
- Train/eval split policy.

### 5.5 Hyperparameters

TODO — full hyperparameter table for every method, with the search
range and the chosen value. Reproducibility blocker if missing.

### 5.6 Reproducibility

TODO — state: every reported number is produced by a single config
file in `experiments/configs/` and a fixed seed list. Provide the
commit hash, dataset hash, and Python environment lock file location.

---

## 6. Results

> *Headline section. No prose without a table or figure to back it.*

### 6.1 Cumulative Operational Cost

TODO — Table 1: cumulative cost per policy, mean ± bootstrap 95% CI,
over ≥30 seeds. Highlight winner. Include p-values vs. CS-DCB.

| Policy             | Cum. Cost (mean) | 95% CI         | p vs. ours |
|--------------------|------------------|----------------|------------|
| static-rules       | TODO             | [TODO, TODO]   | TODO       |
| heuristic-score    | TODO             | [TODO, TODO]   | TODO       |
| offline-classifier | TODO             | [TODO, TODO]   | TODO       |
| linucb             | TODO             | [TODO, TODO]   | TODO       |
| thompson           | TODO             | [TODO, TODO]   | TODO       |
| epsilon-greedy     | TODO             | [TODO, TODO]   | TODO       |
| **cs-dcb (ours)**  | **TODO**         | [TODO, TODO]   |    —       |

Source: `experiments/results/<config>/<seed>/` aggregated by
`evaluation/metrics.py`.

### 6.2 Cumulative Regret vs. Oracle

TODO — Figure 1: cumulative regret curves over time for each policy,
shaded with 95% CI bands. Filename:
`experiments/results/figures/regret_vs_oracle.pdf`.

### 6.3 Cost CDF

TODO — Figure 2: per-decision cost CDF per policy. Highlights tail
behavior. Filename: `experiments/results/figures/cost_cdf.pdf`.

### 6.4 Action Distribution

TODO — Table 2: fraction of decisions assigned to deploy / canary /
block, per policy. Diagnostic for whether canary is being used as
designed.

### 6.5 Statistical Significance

TODO — paired-bootstrap p-values for every (policy_i, policy_j) pair,
Holm–Bonferroni corrected. State alpha level. State number of
comparisons.

### 6.6 Interpretation

TODO — three to four paragraphs. What did we expect? What happened?
Where did the expected pattern hold and where did it break? Be candid
about surprises.

---

## 7. Drift Analysis

### 7.1 Drift Schedule

TODO — describe the drift events injected into the synthetic
environment and (if applicable) the drift segments identified in the
real dataset.

### 7.2 Time-to-Recover

TODO — Table 3 / Figure 3: time-to-recover after each drift event,
per policy. Defined: number of decisions until instantaneous regret
returns within ε of pre-drift level.

### 7.3 Per-Segment Cumulative Regret

TODO — Figure 4: cumulative regret broken down by drift segment.
Demonstrates whether CS-DCB's drift adaptation actually helps, segment
by segment.

### 7.4 Detector Sensitivity

TODO — sensitivity sweep over drift-detector hyperparameters
(ADWIN delta, etc.). Show that the result is not knife-edge.

---

## 8. Cost-Sensitivity Analysis

### 8.1 Cost Matrix Sweep

TODO — vary the cost matrix from §3.4 over a defensible range. Plot
cumulative cost per policy as a function of the
`failure_cost / block_cost` ratio. Show that CS-DCB dominates over the
range, or honestly report where it does not.

Filename: `experiments/results/figures/cost_sweep.pdf`. Source:
`experiments/run_cost_sweep.py`.

### 8.2 Action-Cost Ablation

TODO — what happens to each policy if `canary` is removed from the
action space? If `canary` is the cheap action, removing it should hurt
CS-DCB more than static baselines. This is a strong sanity check.

### 8.3 Robustness to Cost Misspecification

TODO — train CS-DCB under one cost matrix, evaluate under a different
one. Quantify degradation. Defend the algorithm's robustness or honestly
mark this as a limitation.

---

## 9. Discussion

> *Three to four paragraphs. Not a results restatement.*

### 9.1 Why the Bandit Framing Wins

TODO — one paragraph explaining, in plain language, why a sequential
framing beats a classification framing in this domain. Tie to a
specific row in Table 1.

### 9.2 When Bandits Underperform

TODO — be candid. Identify a regime where a static rule or an offline
classifier matches or beats CS-DCB. (E.g., very low failure rates,
very short trajectories, perfectly stationary streams.) This paragraph
is what makes reviewers trust the rest of the paper.

### 9.3 Practical Deployment Considerations

TODO — what would it take to deploy CS-DCB in a real CI/CD pipeline?
Latency budget, propensity logging, cold-start handling, human-override
interface.

---

## 10. Threats to Validity

> *One full page. Reviewers will read this carefully. Do not understate.*

### 10.1 Construct Validity

TODO — does cumulative cost actually measure what we claim it measures?
Defend the cost matrix. Acknowledge that any cost matrix is a modeling
choice; cite §8 sweep as the mitigation.

### 10.2 Internal Validity

TODO — three subthreats:

1. Hidden-state / observable-context contamination in the synthetic
   environment. Mitigation: `tests/test_environment.py` property test.
2. Propensity estimation error in offline replay. Mitigation: doubly
   robust estimator + sensitivity analysis.
3. Seed variance hiding effect-size noise. Mitigation: ≥30 seeds and
   paired bootstrap.

### 10.3 External Validity

TODO — generalization beyond the chosen dataset. Honest statement:
results are claimed for the dataset evaluated; cross-project
generalization is future work (§12).

### 10.4 Conclusion Validity

TODO — multiple-comparison risk. Mitigation: Holm–Bonferroni.

---

## 11. Limitations

> *Distinct from threats to validity. Limitations are what the paper
> does NOT cover, framed honestly.*

TODO — at minimum:

1. Single dataset family. Cross-project transfer not evaluated.
2. Three-action space. Continuous-percentage canary rollout not modeled.
3. Cost matrix is project-uniform. Per-team incident-cost variation not
   modeled.
4. No deep learning. Neural bandits not evaluated. (Future work.)
5. Replay evaluation inherits the propensity assumptions of the logging
   policy.

---

## 12. Future Work

TODO — five to seven concrete extensions. Each should be a single
defensible sentence:

1. Cross-project transfer learning for cold-start risk in new
   repositories.
2. Causal estimation of deployment-induced failures vs. correlated
   failures, using IV / propensity-score matching.
3. Continuous-action canary policies (% rollout as a real-valued
   action).
4. Neural contextual bandits with proper exploration bounds.
5. Active human-in-the-loop deployment with cost-aware deferral.
6. Federated bandits across organizations with privacy guarantees.
7. Longitudinal study of policy drift in a live CD pipeline.

---

## 13. Conclusion

> *One short paragraph. Restate the contribution, the headline result,
> and the broader implication. Do not introduce new claims.*

TODO — write last, after Abstract is locked.

---

## Reproducibility Statement

TODO — single paragraph stating:

- All code is at `<URL>` under commit `<hash>`.
- All experiment configs are in `experiments/configs/`.
- All seeds are in `experiments/configs/seeds.txt`.
- Dataset versions and hashes are in `data/README.md`.
- Python environment is locked in `pyproject.toml` / `requirements.txt`.

---

## References

> *Use ACM or IEEE format depending on target venue. The list below
> is the minimum required reading from CLAUDE.md; expand as the
> related-work read progresses.*

1. Kamei, Y., Shihab, E., Adams, B., Hassan, A. E., Mockus, A.,
   Sinha, A., & Ubayashi, N. (2013). A Large-Scale Empirical Study of
   Just-in-Time Quality Assurance. *IEEE TSE*, 39(6), 757–773.

2. McIntosh, S., & Kamei, Y. (2018). Are Fix-Inducing Changes a Moving
   Target? A Longitudinal Case Study of Just-in-Time Defect Prediction.
   *IEEE TSE*, 44(5), 412–428.

3. Li, L., Chu, W., Langford, J., & Schapire, R. E. (2010). A
   Contextual-Bandit Approach to Personalized News Article
   Recommendation. *Proc. WWW '10*, 661–670.

4. Chu, W., Li, L., Reyzin, L., & Schapire, R. E. (2011). Contextual
   Bandits with Linear Payoff Functions. *Proc. AISTATS '11*, 208–214.

5. Joachims, T., Swaminathan, A., & de Rijke, M. (2018). Deep Learning
   with Logged Bandit Feedback. *Proc. ICLR '18*.

6. Gama, J., Žliobaitė, I., Bifet, A., Pechenizkiy, M., & Bouchachia, A.
   (2014). A Survey on Concept Drift Adaptation. *ACM Computing
   Surveys*, 46(4), 1–37.

7. Bifet, A., & Gavaldà, R. (2007). Learning from Time-Changing Data
   with Adaptive Windowing. *Proc. SDM '07*, 443–448.

8. Vernade, C., Cappé, O., & Perchet, V. (2017). Stochastic Bandit
   Models for Delayed Conversions. *Proc. UAI '17*.

TODO — add as related-work read progresses. Mandatory additions:
recent SEAMS proceedings, Joulani et al. 2013 on online learning under
delay, Dudik et al. 2014 on doubly-robust off-policy evaluation,
Agarwal et al. 2014 on contextual bandit oracles.

---

<!--
  END OF SCAFFOLD.

  Workflow for filling this in:

    1. Lock docs/problem-formulation.md first. Then §3 of this paper
       is a paraphrase of that doc.
    2. Complete the related-work read. Then §2 writes itself.
    3. Implement environment + cost model + LinUCB + Thompson +
       static baselines. Run experiments/run_baselines.py and
       experiments/run_bandits.py with ≥30 seeds.
    4. Generate Tables 1, 2 and Figures 1, 2 from
       evaluation/plots.py and evaluation/metrics.py.
    5. Write §6 Results with the actual numbers.
    6. Implement drift adaptation. Run experiments/run_drift_eval.py.
       Generate §7.
    7. Run experiments/run_cost_sweep.py. Generate §8.
    8. Now §1, §9, §10, §11, §12, §13 can be written.
    9. Write Abstract last.

  Any deviation from this order risks writing prose that has to be
  rewritten when the experiments contradict it. Do not skip ahead.
-->
