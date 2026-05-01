# Related Work

## JIT Defect Prediction

**Kamei et al. 2013 — "A Large-Scale Empirical Study of Just-in-Time Quality Assurance"**  
Introduces just-in-time (JIT) defect prediction: given a commit's static features (churn, complexity, author experience, diffusion), train a binary classifier to predict whether the commit is fix-inducing. The goal is to flag risky commits before they integrate. This is a classification problem evaluated on accuracy, AUC, and recall — not on the downstream cost of acting on the prediction. Our setting differs in three ways: (1) we model the deployment *decision* as the output, not a defect label; (2) cost is asymmetric and explicit in our action–outcome cost matrix; (3) we treat the problem as sequential — the policy must improve over time from its own deployment outcomes, not from pre-labeled training data.

**McIntosh & Kamei 2018 — "Are Fix-Inducing Changes a Moving Target?"**  
Demonstrates that JIT models degrade over time as project characteristics shift — feature distributions change, team composition changes, and the very definition of "risky change" evolves. The paper argues for periodic retraining but does not provide an online adaptation mechanism. The framing remains classification with accuracy metrics. Our work takes the non-stationarity finding as a first-class constraint rather than a limitation: drift detection and policy adaptation (via `drift/detectors.py` and `drift/adapt.py`) are built into the control loop, and we measure time-to-recover after a drift event rather than post-hoc accuracy degradation.

---

## Contextual Bandits

**Li et al. 2010 — "A Contextual-Bandit Approach to Personalized News Article Recommendation" (LinUCB)**  
Proposes LinUCB: a contextual bandit algorithm that maintains a linear reward model per arm and selects actions using an upper confidence bound. Achieves sublinear regret under linear realizability. Reward in this setting is click/no-click — observed immediately after the action. Our setting violates the immediate-reward assumption: deployment outcomes arrive `k_t ~ Geom(p)` steps after the action, and some are censored. LinUCB is our first real bandit baseline precisely because it is theoretically grounded and well-understood; we extend it with a delayed-reward buffer and measure the regret penalty incurred by delay.

**Chu et al. 2011 — "Contextual Bandits with Linear Payoff Functions"**  
Provides tight regret bounds for the stochastic linear contextual bandit: `O(√(dT log T))` cumulative regret where `d` is context dimension and `T` is the horizon. The analysis assumes i.i.d. contexts, immediate rewards, and a stationary reward distribution. All three assumptions are violated in our setting: contexts are temporally correlated CI/CD trajectories, rewards are delayed, and `P(outcome | context)` is non-stationary. The Chu et al. bounds establish the best-case floor; our empirical contribution is to show how much regret the delay and drift penalties add on top of that floor in a realistic deployment setting.

---

## Counterfactual and Off-Policy Evaluation

**Joachims et al. 2018 — "Deep Learning with Logged Bandit Feedback"**  
Addresses the problem of learning a new policy from historical logs collected under a different (logging) policy. The core technique is inverse propensity scoring (IPS): weight each logged reward by the inverse probability that the logging policy would have taken the same action, correcting for selection bias. Doubly-robust (DR) estimation combines IPS with a direct reward model for lower variance. These methods assume known logging propensities and sufficient coverage of the action space. Our offline evaluation in `evaluation/replay_eval.py` applies IPS and DR over real CI/CD logs, with propensity clipping at 20× to control variance — directly implementing the estimators from this line of work. The insufficiency is coverage: if the logging policy rarely blocks changes, the IPS variance for estimating a cautious bandit policy is high; this is a stated threat to validity.

---

## Concept Drift

**Gama et al. 2014 — "A Survey on Concept Drift Adaptation"**  
Comprehensively surveys drift detection methods (ADWIN, DDM, Page-Hinkley, EDDM) and adaptation strategies (retraining, ensemble weighting, sliding windows). The survey treats drift detection as a monitoring problem and adaptation as a supervised learning problem. In our setting, the adaptation target is a *bandit policy*, not a classifier — resetting a bandit's reward model is not the same as retraining a classifier, because the exploration–exploitation tradeoff resets too. We use ADWIN or Page-Hinkley on the reward stream (as surveyed by Gama et al.) but couple detection directly to policy reset and windowed update in `drift/adapt.py`, measuring regret recovery rather than accuracy recovery.

---

## Delayed Feedback in Bandits

**Pike-Burke et al. 2018 / Joulani et al. 2013 — Bandits with Delayed Feedback**  
A line of work (Joulani et al. 2013 *"Online Learning under Delayed Feedback"*; Pike-Burke et al. 2018 *"Bandits with Delayed, Aggregated Anonymous Feedback"*) establishes that stochastic delay with maximum delay `d` inflates regret by an additive `O(√(dT))` term. Censored rewards (where the delay is infinite for some fraction of outcomes) further complicate updates. These results justify why every policy update in our system must route through `delayed/buffer.py` — policies that consume reward at decision time are implicitly assuming `d = 0` and will be invalidated by the evaluation protocol. The key gap these papers leave is the interaction between delay and concept drift: regret bounds assume a stationary reward distribution, whereas our trajectories are non-stationary. Quantifying this interaction empirically on CI/CD data is part of our contribution.

---

## Self-Adaptive Systems (SEAMS)

**Weyns et al. / SEAMS Literature — MAPE-K and Learning-Based Adaptation**  
The SEAMS community formalises self-adaptive systems around the MAPE-K loop (Monitor, Analyse, Plan, Execute over a shared Knowledge base), establishing a common vocabulary for feedback-driven control in software systems. Recent SEAMS work incorporates reinforcement learning into the Plan phase, but typically assumes hand-crafted reward functions, stationary environments, and discrete state spaces derived from the managed system's internal model. Our work differs by: (1) treating the deployment decision as the *primary* research artifact rather than a systems-engineering concern; (2) replacing the static Analysis phase with an online bandit policy that learns cost-sensitive action preferences from trajectory data; and (3) evaluating on real CI/CD logs with proper off-policy correction rather than simulation. The SEAMS framing motivates the problem domain; our contribution is the algorithmic and evaluation machinery that the community currently lacks for deployment-quality decisions under delayed, cost-asymmetric feedback.
