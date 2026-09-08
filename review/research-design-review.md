# Research design review of the corrected release controller

Reviewed 6–7 September 2026. This is a design review, not a new experimental result or manuscript revision. No policies were trained, no additional seeds were run, and no headline artifacts were regenerated. The accompanying [descriptive audit script](research_design_data_audit.py) reads the existing CSVs and corrected loader; its [machine-readable evidence](evidence/research-design-data-audit.json) records dataset hashes, missingness, cardinalities, timestamp checks, and project counts.

**Decision:** the main CSV replay is a delayed, partially observed-label, full-information cost-sensitive prediction problem. It is not naturally a partial-feedback contextual-bandit problem. The current data is inadequate to test meaningful change context or generalize beyond the two exports. Retain LinUCB and Thompson as comparators; do not organize the research around a presumed need for bandits. The separate action-dependent synthetic simulator has a different feedback model and must remain a separate study.

## 1. Correct problem formulation

### The decision and target must be fixed first

The corrected replay decides at **CI workflow start**, not after a passing CI build and not at a recorded production release. For observation t, define:

- s_t: workflow-start decision time;
- o_t: time its CI conclusion actually becomes available;
- X_t: verified information available by s_t, including history H_t of earlier observable events;
- Y_t in {0,1}: CI success/failure, with 1 meaning failure;
- M_t: indicator that a binary label is observed by the declared evaluation cutoff;
- a_t in {deploy, canary, block}: a **simulated action label**;
- C_t(a,y): known, prespecified hypothetical cost matrix, possibly changing by a declared scenario.

The information set F_t contains current verified metadata, earlier metadata available by s_t, matured labels, previously declared cost matrices, and previous decisions. It excludes the current CI result, current final duration, unfinished-run outcomes, later repository snapshots, and evaluation-set statistics. The decision is measurable with respect to F_t. Seeing a label later does not make it permissible at decision time.

For each resolved observation the simulator defines the entire loss vector:

    L_t = (C_t(deploy,Y_t), C_t(canary,Y_t), C_t(block,Y_t)).

This vector becomes reconstructible at o_t. The objective currently measured is a complete-case proxy cost, sum_t M_t C_t(a_t,Y_t), with the same M_t for every policy. Dividing by sum_t M_t gives mean resolved-row cost. This is not total operational cost and is not automatically representative of unobserved labels. Delay and missing labels remain important, but neither creates action-selective feedback in this replay.

The appropriate primary framework is **delayed online binary probability estimation followed by minimum expected-cost action selection**. It can also be described as full-information online cost-sensitive decision-making, or cost-sensitive classification with three decisions and two outcome states. It need not be cast as three-class outcome prediction. Bayesian risk minimization is a decision principle compatible with either a Bayesian or a frequentist probability estimator; it is not itself an additional competing feedback model.

Let p_t = P(Y_t=1 | F_t). Then

    Q_t(a) = (1-p_t) C_t(a,0) + p_t C_t(a,1),
    a_t* = argmin_a Q_t(a).

This is the standard probability-to-expected-cost decision principle ([Elkan, 2001](https://cseweb.ucsd.edu/~elkan/rescale.pdf)). Applied to this repository's default matrix, the three expected costs are 10p, 1+3p, and 2-1.5p. Deploy minimizes cost below p=1/7, canary between 1/7 and 2/9, and block above 2/9; adjacent actions tie at each boundary. Other configured matrices require recalculating their lower envelope. The rule is not simply p times failure cost: success, canary overhead, and blocking penalties also matter.

**Structural implication derived from the code:** the loss vector is u+vY, where u contains success costs and v contains failure-minus-success costs. All three expected action values depend on one conditional probability. Fitting three unrelated action-value models discards this known relationship. A scalar *output* p(X,H) can still depend on rich context; a scalar probability output must not be confused with a context-free failure-rate model.

For Bayesian risk-neutral decisions, posterior expected cost uses the posterior predictive mean probability. Choosing an uncertain action does not improve the environment's label stream when that stream, its timing, and future contexts are independent of action. Conditional on those assumptions, the future-information term in a dynamic decision objective is the same for every current action. Consequently exploration has no information-acquisition value here. This is a statement about the supplied feedback model, not a theorem that stochastic algorithms can never have useful numerical behavior or that deployment never requires exploration.

Delays are a separate property of online learning; delayed feedback is studied across online-learning settings, not only bandits ([Joulani et al., 2013](https://proceedings.mlr.press/v28/joulani13.html)). A change in failure probability over time can motivate forgetting or online adaptation without motivating bandit feedback.

If the intended decision is instead **after CI has passed**, the CI label is already known. This dataset then lacks the subsequent deployment-incident target needed for the claimed decision. Moving the decision time to after CI while retaining CI failure as the unknown reward would be inconsistent. A sequential canary-then-promote process, with new telemetry and later actions, may require a sequential decision/partial-observation model rather than a one-step bandit.

## 2. Feedback-model analysis

### Code trace and the eight formulation questions

| Question | Finding grounded in the implementation |
|---|---|
| 1. What is available at decision time? | `data/loaders.py::iter_records` creates current metadata plus matured history. `evaluation/online_replay.py:193` releases pending feedback before calling `select_action`. `reveal_steps` maps completion times to the next decision. The context object itself does not prove that its raw fields were measured before the decision; Section 4 checks that separately. |
| 2. What feedback follows each action? | A resolved `Reward` contains the **CI outcome itself**, chosen-action cost, action identifier, and delay. The original context and chosen action are also supplied to `update`. Canceled/missing observations are skipped uniformly. |
| 3. Is feedback action-dependent? | In the CSV replay, **no**: `_effective_outcome` at `evaluation/online_replay.py:85` returns the logged outcome unchanged for every action. Completion times and subsequent CSV rows are fixed independently of policy actions. The selected numerical cost changes; availability and the binary label do not. |
| 4. Can every action cost be computed? | Yes for a resolved label: `rewards/cost_model.py:62` deterministically maps each action and that label to cost. This is model-defined counterfactual reconstruction, not observation of actual deployment counterfactuals. No loss vector can be reconstructed from a missing binary label without additional assumptions. |
| 5. Why use contextual bandits anyway? | They can be benchmark comparators or deliberately restricted learners. There is no demonstrated environmental need for their partial-feedback formulation. `LinUCBPolicy.update` and `ThompsonSamplingPolicy.update` update only the selected arm despite receiving a label that could supervise every arm. |
| 6. Which framework fits better? | Delayed supervised probability learning plus expected cost; alternatively full-information cost-sensitive learning or full-information online loss regression. Beta-Bernoulli rates are simple instances, not a substitute for testing contextual probability models. |
| 7. What would make it a true bandit? | Only selected-action reward/feedback is observable and the other action losses are not recoverable from it under known structure. A real action/outcome and feedback contract is necessary; merely routing a label through a method named `Reward` is insufficient. |
| 8. What claims fail? | Necessity of exploration, inherent partial feedback in the CI replay, extra information obtained by canarying, and deployment-specific bandit benefits. Selected-arm algorithms' observed costs remain valid descriptions of those implementations on the proxy task; their scientific interpretation changes. |

Standard contextual-bandit feedback reveals the chosen action's reward rather than the full loss vector ([Vowpal Wabbit documentation](https://vowpalwabbit.org/docs/vowpal_wabbit/python/latest/tutorials/python_Contextual_bandits_and_Vowpal_Wabbit.html)). A fully labeled dataset can legitimately be used to *simulate* restricted feedback if that restriction is explicit; benchmark work does this ([Bietti, Agarwal and Langford, 2021](https://jmlr.org/papers/v22/18-863.html)). That does not establish that restricted feedback describes the application.

**Even hiding `reward.outcome` would not repair this formulation.** In the default known matrix each action has distinct success and failure costs: deploy 0/10, canary 1/4, block 2/0.5. Exact selected-action cost therefore reveals Y, which reveals the entire vector. A restriction must be substantive and non-invertible, not just an API change. Action-dependent outcomes are one way to obtain partial feedback, but are not logically required: an unknown, non-reconstructible loss vector with only one observed component also suffices.

### Conditions for a separate genuine partial-feedback study

A defensible bandit model would require defined real decisions; measurable chosen-action consequences; non-reconstructible unchosen outcomes; and a feedback process specifying what block and canary actually reveal. Blocking generally does not reveal whether that change would have caused a production incident. Canary telemetry may provide side observations, which should be modeled rather than silently discarded. A CI label may remain a common auxiliary signal without identifying production losses.

For one-step contextual-bandit analysis, immediate action consequences must be the relevant objective and persistent effects on future opportunities/state must either be negligible or explicitly addressed. If actions alter queues, later releases, incidents, or remediation, a richer sequential model may be necessary. Delays, censoring, label attribution, observation windows, and available actions must be explicit. Observational deployment logs additionally need a justified identification/evaluation design; recorded propensities, overlap, and confounding assumptions are not optional details that this CI export supplies. A prospective randomized design is an alternative, subject to the actual operational constraints; none is performed here.

### The synthetic drift experiment is different

`environment/synthetic.py::_ACTION_FAILURE_MULTIPLIER` sets failure probability to p for deploy/block and 0.4p for canary. `step` samples and returns the selected action's outcome. Observing a canary success does not reveal the realized deploy outcome. Thus that simulator has action-dependent partial feedback, unlike the CSV replay. Known multipliers can transfer information about the underlying p, but do not reveal every realized loss.

This is a structured synthetic bandit setting with a strong, artificial assumption that blocked changes also produce would-have-failed labels. `experiments/run_drift_eval.py::MomentRatePolicy` is aware of the multiplier. Its estimator is not a standard Beta-Bernoulli posterior for a common outcome probability. The current drift history mixes outcomes from actions with different probabilities, so histories can differ between policies even under matched environment random draws. Do not pool its results with the full-information replay to support one feedback-model claim. It may remain an explicitly separate simulation sensitivity, after its own design is justified.

### Claims requiring withdrawal or further narrowing

| Claim or implication | Consequence of the formulation review |
|---|---|
| CI replay is inherently a contextual-bandit deployment problem | Incorrect under the implemented feedback model. |
| Exploration is needed to discover unchosen action costs | Incorrect for resolved CI labels and the known matrix. |
| Canary's benefit here is information acquisition | Incorrect. The replay produces the same label and timing after every action. Its possible benefit is the assumed expected-cost tradeoff. |
| Delay and drift justify a bandit formulation | Unsupported; these also occur in full-information prediction. |
| A selected-arm method beating a static rule establishes value of bandit learning | Unsupported; cost sensitivity, prediction, and inefficient feedback use are confounded. |
| Twelve context fields demonstrate use of rich real change information | Incorrect for this real export: eight are constant missing-data encodings. |
| Losing to scalar rules demonstrates that context is unnecessary | Too strong. Real change features are absent and strong contextual probability models have not been evaluated. |
| The corrected results demonstrate savings from real deploy/canary/block choices | Unsupported; the losses are hypothetical CI-label functions. |
| Numerical rankings of the existing implementations on the frozen data | Remain descriptive facts, subject to their stated cohort and parameters. They do not answer the redesigned research question. |

The current corrected manuscript already disclaims causal savings and formal bandit guarantees. It still needs a future explicit formulation correction and a narrower interpretation of its context results. Legacy `docs/problem-formulation.md` remains especially inconsistent: it starts decisions after a passing build and describes canary information benefits. `data/README.md` also discusses a large external corpus rather than the actual local fixture and contains unsupported IPS language. Their “LOCKED” or historical status does not make them evidence. They have been identified here, not rewritten during this review.

## 3. Data adequacy assessment

### Real export actually available

| Quantity | pallets/flask | psf/requests |
|---|---:|---:|
| Workflow rows | 300 | 300 |
| Success / failure / canceled | 225 / 70 / 5 | 267 / 16 / 17 |
| Failure fraction of all rows | 23.33% | 5.33% |
| Failure fraction of resolved rows | 23.73% | 5.65% |
| Distinct SHAs | 103 | 53 |
| Exact duplicate rows beyond first copy | 1 | 24 |
| Distinct recorded author names | 44 | 10 |
| Observed start-date span | 25 Jan–5 May 2026; 99.89 days | 14 Apr–5 May 2026; 20.86 days |
| Unique start timestamps | 210 | 121 |
| Corrected encoded feature rank, including intercept | 5 | 5 |
| Decisions with an already-observed resolved label for the same SHA | 104 | 69 |
| SHAs with differing statuses across their rows | 45 | 8 |
| SHAs with differing recorded author names | 1 | 9 |
| SHAs with differing PR flags | 1 | 18 |

There are **two repository clusters**, not 600 independent projects, commits, or deployments. Even their statistical independence is an assumption: related ecosystem, tooling, and time effects may correlate them. The 156 project/SHA pairs are not automatically independent either. The 173 decisions with earlier same-SHA feedback show why the prediction unit matters. Such feedback can be legitimate for predicting a later workflow; it is not evidence of predicting a never-before-seen change. Differing statuses for the same SHA could reflect workflows, attempts, events, or flakiness, not necessarily erroneous labels. Missing IDs prevent resolving those alternatives.

The two failure rates and duration distributions establish heterogeneity within these exports. They do not characterize a population of projects. In Requests there are only 16 failed workflow rows before splitting; distinct failed changes and effective independent failures may be fewer. Flexible models, calibration, subgroup analysis, equivalence, and drift claims would be particularly unstable. The two projects also cover very different durations, and we do not have the sampling frame needed to explain why these 300 rows were selected.

The export lacks run IDs, workflow IDs, attempt numbers, branch, event type, parent/diff references, collection commands and time, and authentic deployment outcomes. It has 25 exact duplicate rows. All timestamps parse with finish >= start, but 17 Requests rows have finish equal to start; all are canceled. Timestamp parseability is not proof of original event semantics. Whether “finished” came from a completion event or a mutable API update is unverified.

**Adequacy verdict:** sufficient for debugging, describing these two CI exports, and a preliminary feasibility case study. Insufficient for a general claim about context, bandits, CI systems, or deployment benefit. More policy seeds cannot repair missing fields, workflow identity, rare failures, or the two-project sample. With two clusters, a project bootstrap cannot create credible population evidence; leave-one-project-out would yield only two highly unstable transfer cases.

### Other local data does not supply missing real replication

The frozen smoke CSV has 1,150 generated rows in two artificial projects, 97/600 and 193/550 failures, and no overlapping starts/completions. Its feature matrices have rank 11 including bias; dependency and risky-path flags are still constant zero. The original generator is missing. Variation alone does not establish a known context-to-risk relationship or realistic joint feature distribution. Use it for implementation regression and historical reproducibility, not to validate the absence or presence of context effects.

The ignored `data/raw/travistorrent.csv` contains 600 rows labeled `phase17/smoke-repo`, with 98 failures and rank 10. No verified independent real-project provenance accompanies it; it cannot be counted as an additional real sample. The headline real CSV is the same export as the frozen real fixture, not another cohort. Mentions of large external datasets in the README are not locally available evidence.

## 4. Feature-validity audit

Three different standards are kept separate: **present in raw data**, **mechanically restricted to earlier observations**, and **proven available under the real operational decision contract**. Only the second can be verified fully from the corrected replay code for history fields. Nonzero variance is necessary for a feature to distinguish these observations, but does not prove predictive usefulness.

| Feature or group | Exists and varies in the real export? | Availability and leakage assessment | Risk rationale and decision |
|---|---|---|---|
| Files changed | Column exists; 600/600 blank; encoded 0 everywhere | A frozen commit diff could be known before workflow start, but is absent here | Change breadth is a plausible risk signal; **unusable in current real data**. |
| Lines added | 600/600 blank; encoded 0 | Requires the exact predecision diff; none supplied | Change size could matter; no current evidence. |
| Lines deleted | 600/600 blank; encoded 0 | Same requirement | Deletions could affect risk; no current evidence. |
| Source churn | 600/600 blank; encoded 0 | Requires a documented source-file definition and diff snapshot | Not evidence that churn is zero or irrelevant. Avoid redundant/redefined churn features without a contract. |
| Changed paths / dependency changes | No changed-path column; derived dependency flag is false for every row | Existing helper recognizes file patterns, but no paths were exported | Plausible signal; **missing**, not confirmed absence of dependency changes. |
| Risky path changes | No paths; false everywhere | Same issue | Operational/configuration paths could matter; currently untestable. |
| Current tests executed | `gh_tests_run` blank in all rows; corrected prior-run version also always 0 | Current execution count would normally be post-start; the loader now takes the latest completed run's count | Test history could help, but this export contains no counts or identities. |
| Tests added | `gh_tests_added` blank in all rows; 0 everywhere | Could be a static diff measure if explicitly implemented; could instead be post-run metadata. No provenance | Coverage/change relationship plausible; currently unavailable. |
| PR status | Present: Flask 125 true/175 false; Requests 129/171; two values per project | Event-associated PR status can exist at trigger time. The export lacks event/payload snapshots, and the flag varies within 19 SHAs | Candidate **event context**, not validated change complexity. Conditional use only after provenance verification. |
| Current final duration | Raw `tr_duration` varies | **Forbidden as a current feature** at CI-start decision time; final duration is known afterward | Can reflect workflow type, load, early failure, or timeout. A strong association could be leakage. |
| Prior completed-run duration | Corrected context has 54 unique values in Flask (0–86,402 s), 65 in Requests (0–423 s); medians 18/68 s | Loader filters finish <= current start. Mechanically earlier under recorded timestamps. No completion-event provenance or workflow identity; equal-time order needs a declared convention | Plausible history/load signal, but mixes unrelated workflows. Flask maximum is about 24 hours, far outside the encoder's 180 s denominator; no clipping occurs. Validate meaning and train-only transforms. |
| Author/project experience | Derived; 51 values in Flask (0–50), 39 in Requests (0–38) | Author emails and commit timestamps are absent. Counts use raw names and distinct SHAs seen earlier in row/start order. Names vary within 10 SHAs; role could be actor rather than commit author. Left-truncated and tied-start ordering is not complete author history | Plausible familiarity signal, but **not yet validated as author experience**. Resolve immutable identity and actor/author roles; define strict event ordering. |
| Recent failure rate | Derived from resolved completions in preceding seven days; 73 values in Flask (0–0.5), 63 in Requests (0–0.12) | Corrected code uses only prior rows completed by the decision and excludes missing labels. Verified against recorded availability. Same-SHA and cross-workflow history can enter, subject to task definition | Strongly plausible historical baseline, not genuine current-change context. Counts, age, and workflow grouping are needed to interpret reliability. |
| No-history indicator / observation count | Not encoded. Four real decisions have an empty seven-day resolved window | Current code represents no history as rate 0, conflating no evidence with evidence of no failures | Derive from the same matured history, expose equally to comparators, and use explicit cold-start handling. |
| Raw author identity | Names present (52 distinct across export); emails absent; raw identity not encoded | Availability/meaning unverified, and names are not stable IDs | Could encode familiarity or selection effects; do not treat high cardinality as useful predictive evidence. |
| Project identity | Two slugs; not in the numerical vector | Known at decision time. Constant inside each project-reset trajectory | Cannot explain a within-project contextual gain. A project-identity ablation is identifiable only in a separately defined pooled/transfer protocol. |
| Workflow structure and configuration | Workflow/run IDs, jobs, matrices, attempts and workflow YAML absent | A versioned configuration is potentially predecision information; completed job behavior is not | Important candidate confounder and context; currently unavailable. |
| Repository metadata | Only slug supplied; language/activity/age/dependency metadata absent | Must use a historical snapshot, not today's repository metadata joined backward | Potential transfer features, presently untestable. |
| SHA and step index | Present but not numerical model features | Identifier/time index known; may reveal repetition or time trend | Do not use arbitrary SHA encodings as risk features. SHA is needed for grouping; time trends must be separately labeled and evaluated. |

Eight of the twelve numerical context fields are constant zero in both real projects. The remaining four are PR status and three history measures. Including the intercept gives rank five, not evidence for a rich 13-dimensional change-risk problem. Each varying field is a candidate, not a demonstrated predictor: this review deliberately does not fit models or compute new performance results.

Additional code-contract issue: `features/extractor.py::extract_context` still takes a CI payload and calls `ci_run_duration_seconds` without enforcing that this payload belongs to a **prior** completed run. The corrected CSV loader is safer; the generic ingestion path cannot be assumed safe merely because it returns a `Context`. `GitHubClient.collect_deployment_inputs` selects the latest run per SHA rather than reconstructing an as-of event snapshot. These paths need a shared, executable availability contract before collecting a new evaluation corpus. No implementation change is made in this review.

## 5. Revised experimental design

### Primary question and competing explanations

**Question:** on precisely defined CI prediction units, does current change context improve delayed probability estimation and the resulting hypothetical cost decisions beyond historical signals, and do selected-arm algorithms add anything beyond a contextual probability model under the same available feedback?

Use this decomposition, without implying the components are universally additive:

    fixed or historical probability estimate -> expected-cost decision
    contextual probability estimate         -> expected-cost decision
    direct action-value learner             -> decision

| Explanation | Required matched contrast | What it can establish |
|---|---|---|
| A. Cost sensitivity explains gain | Freeze the SAME probability predictions and compare a prespecified cost-insensitive three-tier mapping g_ref(p) against argmin expected cost. Both have the same action set and are scored under the same matrix | Effect of changing the decision mapping; unlike bandit-versus-static, this does not simultaneously change predictions. |
| B. Historical estimation explains gain | Same expected-cost map with a frozen development-only probability p0 versus empirical/Bayesian/windowed history estimates | Value of online history adaptation relative to a no-update control; distinguish global and project-specific priors. |
| C. Context adds value | Same supervised family and training protocol using H alone versus H plus verified current-change features G | Incremental predictive and decision value of G, beyond history and model family. A rate-only comparator is necessary but insufficient to isolate G. |
| D. Bandit-specific learning adds value | Matched H+G information for probability-plus-cost, direct all-action loss learning, and selected-arm LinUCB/Thompson; add matched sampling/greedy controls | Performance of learning/decision architectures; in full information, a win alone cannot establish an exploration information benefit. |

Fix g_ref before test evaluation, including its two thresholds. Do not choose it to be deliberately bad on the final costs. Retain the original static rule as a legacy comparator, but its difference from an unrelated probability model is not a clean decomposition. Report interactions across features, cost matrices, and learners rather than assigning one universal percentage to each explanation.

### Required comparators

| Method | Input and update contract | Role |
|---|---|---|
| 1. Original static rule | Same allowed packet; fixed existing thresholds; no fitted state | Legacy reference only, not the sole baseline. |
| 2. Empirical rate + expected cost | Failures/resolved observations from matured project history; explicit development-chosen fallback before first label | Learning a base rate without smoothing. |
| 3. Bayesian rate + expected cost | Beta prior selected on development data; update once per matured binary unit, regardless of simulated action | Smoothed rate estimation. Stationary/exchangeability approximation must be stated. |
| 4. Rolling rate + expected cost | Time/count windows selected only on development folds; explicit count, empty-window fallback, and availability convention | Historical adaptation. Current fixed seven-day rule is one comparator, not privileged tuning. |
| 5. Contextual supervised probability + expected cost | Primary regularized logistic model with online updates or scheduled refits using matured labels; H-only and H+G variants; a small nonlinear probability model as a development-selected robustness check | Test context without disjoint action models. Fit/calibrate only on permissible past data. |
| 6. LinUCB | Identical current features, cost matrix, and feedback packet; standard selected-arm update recorded explicitly | Existing selected-arm action-value architecture. |
| 7. Thompson Sampling | Same packet and features; independently selected prior/noise settings within the same tuning budget | Existing Gaussian selected-arm architecture. |

Also include always-deploy/canary/block, a fixed-p0 expected-cost rule, and a **full-information direct-loss regression** control that updates every action model when Y arrives. The last is not standard LinUCB: label it accurately. It separates action-value parameterization from throwing away available supervision. To examine action randomization, compare posterior-predictive-mean versus sampled decisions with the same contextual probability model and common label updates. All-action loss observations from a single Y are correlated; do not count them as three independent labels or multiply an inappropriate independent Bayesian likelihood.

In the main full-information experiment, every learner is permitted the same delayed packet `(original features, Y, cost vector, observation time)`. Standard LinUCB/Thompson may use only the chosen component as an explicit algorithmic restriction, not as the environment's contract. The all-action and shared-probability controls are needed to interpret this restriction. Giving supervised learners all labels while presenting labels as unavailable to bandits would be an unfair formulation comparison.

A deliberately restricted-feedback experiment is a separate question. All methods there must obey the same genuinely non-reconstructible observation model. A probability baseline cannot receive hidden labels for blocked actions. Where outcomes depend on action, it needs a justified action-conditional model; a naive shared Beta posterior is not appropriate. Simply hiding label fields in the existing deterministic cost simulation is insufficient because costs reveal labels.

### Unit, splits, and online protocol

1. **Choose the unit before fitting.** Recommended first task: predict the first eligible CI run for a new change under a prespecified workflow/event definition, before any outcome for that change is known. Alternatively study every workflow attempt, but name that different target, retain IDs, cluster repeated SHAs, and report first-seen versus repeated-change performance. Do not reconstruct releases by arbitrarily taking the first CSV row. Workflow-group outcomes require a fixed required-workflow set, an explicit failure aggregation rule, and a label time no earlier than its definition permits.
2. **Treat both existing projects as development data.** Their labels and comparative results have already informed many decisions. They are not fresh confirmatory holdouts. New collection should have a documented sampling frame and locked eligibility rules not based on observed model gains.
3. **Use a primary within-project prequential test on untouched projects.** Hyperparameters and preprocessing rules are chosen on separate development projects using chronological rolling-origin validation. During each held-out project stream, all methods start with the same declared prior/pretraining allowance and learn only from matured past labels. Score every declared test decision, including a separately reported cold-start period. Do not let arbitrary project order transfer model state.
4. **Keep transfer as a distinct secondary protocol.** If models are pretrained across projects, grant the same training history to every comparable method and distinguish previously seen from unseen repositories. Raw project-ID one-hot effects cannot estimate an unseen project's behavior; use a declared unknown category or independently justified metadata/hierarchical prior. Never claim the per-project-reset study tested project identity.
5. **Group repeated units across splits.** Shared SHAs, runs and reruns cannot straddle independent training/test groups for the new-change target. Never random-split workflow rows. For temporal holdouts, exclude not-yet-observed training labels at the split boundary; do not pretrain with labels merely because the originating commit is old.
6. **Reconstruct one event stream.** All label arrivals at each instant are processed consistently before the next decision, with a documented tie convention; if order is unknown, conservatively batch simultaneous decisions before ambiguous equal-time feedback. Feature histories and model updates use this same clock. Use actual availability timestamps, not duration divided by arbitrary minutes.
7. **Fix the horizon and follow-up.** Predict T eligible decisions, then allow the same declared maturation window for scoring their outcomes. Tail labels cannot train a model and then improve an earlier scored prediction. Report unresolved labels separately, with common masks and coverage. If missingness is informative, limit claims to the resolved cohort and perform a prespecified missingness sensitivity; do not call dropped observations zero operational cost.

### Feature ablations and fair tuning

Use nested, explicitly named groups: bias/prior only; scalar historical failure rate with count/age; richer history H (author familiarity, prior workflow outcomes/durations and exposure); event metadata; verified current change features G; H+G; and project metadata/identity only in the pooled protocol. The key context contrast is H versus H+G in the same supervised family. Also compare H-only supervised learning against a scalar rule, so richer history is not mislabeled as current-change value. Every feature must pass an as-of availability check before inclusion.

Use identical outer folds, chronological inner folds, decision horizons, cost scenarios, and current information. Give each tunable family the same prespecified maximum number of candidate configurations and the same validation schedule; publish both trial counts and compute time. A fixed rule need not waste trials to match a flexible model. Choose priors, windows, regularization, calibration, feature transforms, LinUCB exploration, and Thompson noise/prior scale only within development data. All feature transforms and missingness handling are fitted there or updated from permissible history; no whole-export scaling. Use the same prespecified validation objective, such as mean proxy cost across the declared cost scenarios, for model selection; retain unweighted proper probability scores as separate diagnostics. If cost-specific tuning is allowed, allow it to all methods and report its extra budget.

Freeze probability predictions before comparing cost matrices when measuring the value of cost mapping. When retraining direct cost learners per matrix, make that a separately labeled comparison and calibrate reward-scale hyperparameters through the same tuning protocol. Do not treat the deploy-failure/block-bad ratio as the entire problem: publish all matrix entries and vary components in interpretable scenarios.

Class imbalance should be represented through sampling diverse projects and adequately long observation windows, not balancing or downsampling the test stream. If resampling or class weighting is used during training, calibration must target the original prevalence. Do not manufacture drift by shuffling rows or changing labels in a real-data test. Report natural time segments as temporal heterogeneity unless a change is independently defined. Controlled drift/class-imbalance/delay mechanisms can be studied in a separately specified synthetic full-information generator with zero-context and useful-context cases, avoiding an oracle-like risk field. That generator is proposed, not implemented or run here.

### Outcomes, uncertainty, and stopping criteria

Primary comparisons should report per-project mean resolved-unit proxy cost, paired cost differences, and a macro-average across projects; total/micro-averaged cost is secondary because it weights prolific projects more heavily. Report Brier score and log loss for probability models, calibration diagnostics with adequate failure counts, action distributions, resolution coverage, and cost-relevant threshold crossings. Better ranking/AUC alone does not establish better probabilities or better decisions. Context can improve prediction without changing decisions if risks remain on one side of a cost threshold.

Do not report disjoint action scores as calibrated failure probabilities without a justified transformation. A true conditional expected-cost oracle is available only in a synthetic model with known p and is an expected-cost **lower bound**, not a practical baseline. A hindsight realized-label oracle is a different, clairvoyant comparator and must not be called the Bayes oracle or standard bandit regret.

For new real data, project is the primary replication cluster; preserve temporal/run/commit dependence within it. Use paired project effects and, only with enough independent projects, cluster-level uncertainty. Within-project block/bootstrap sensitivities do not substitute for project replication. Stochastic-seed variation is nested within the fixed project/task and summarizes algorithm randomness, not population uncertainty. Do not count deterministic seed repeats as independent observations or tune on favorable seeds. Existing seed outputs need not be expanded during design review.

Prespecify the main contrast family, a multiplicity procedure for confirmatory comparisons, and a smallest practically meaningful cost improvement/equivalence margin in the declared proxy units. Hypothetical costs cannot acquire practitioner meaning through a p-value. Determine sample-size/precision targets from development-only variance and plausible effect sizes before collecting or unlocking test data. Non-significance does not establish equivalence. If intervals include material advantages in both directions, choose “inconclusive.”

## 6. Minimum additional data required

**There is no defensible universal minimum number of projects or rows.** Required size depends on independent project variation, event frequency, the smallest useful effect, model complexity, and the intended generalization. The necessary minimum is first a complete observation contract and actual usable features, not another batch of policy seeds.

For a CI-proxy study, obtain at least:

- Immutable repository, commit and parent/diff references; run/workflow/attempt IDs; event, branch, trigger and required-workflow definitions; and an explicit sampling manifest. Resolve exact duplicates and distinguish reruns from separate workflows.
- Decision/start, actual completion, ingestion/availability, and evaluation-cutoff timestamps; retain all terminal statuses, including cancel/timeout/skip/neutral distinctions. Define which statuses constitute prediction failure versus missing/competing outcomes before examining costs.
- Predecision diff size/churn and changed paths; documented dependency/test-change features; versioned workflow configuration; stable actor and commit-author identities kept distinct; and history that precedes the sampling window or an explicit left-truncation indicator.
- Historical feature snapshots and missingness indicators. Zero must not silently mean unavailable, and historical APIs must not supply information added after the decision.
- New, independently selected projects covering differing failure prevalence, workflow structures and change distributions, with enough chronological history for cold-start, training, calibration and evaluation. Both current projects are development cases.

**Planning target, not a statistical guarantee:** scope an initial collection around 5 development repositories plus at least 10 untouched evaluation repositories, with approximately 1,000 distinct eligible change decisions per repository over multiple months and longer coverage where drift is an objective. At a 5% failure rate, 1,000 decisions produce only about 50 failures on average, before missingness or any split; Requests currently has only 16 failed rows. This illustration explains why event counts and independent projects matter. Do not exclude low-failure projects merely to meet a quota: extend observation or accept wider uncertainty. Confirm or increase these planning numbers using a prospective precision/power plan; ten test projects do not establish broad generality by themselves.

Recovering missing diffs for the existing SHAs could improve a pilot after immutable-source verification, but would not create an untouched test set, repair the workflow IDs, or add projects. The frozen synthetic fixture needs a documented generation process if used for controlled research claims; its byte-for-byte reproducibility alone is insufficient.

A **production deployment** study additionally needs actual release decisions and eligible alternatives, CI state at that decision, rollout exposure, canary telemetry, incidents/rollback linkage, detection windows, remediation and delay costs, and the action-dependent observation process. Unchosen production outcomes will normally remain unknown. Those data and identification assumptions are a different research undertaking; adding them is not a necessary precondition for an honestly labeled CI simulation paper.

## 7. Defensible claims under each possible result

| Outcome | Required evidence | Defensible wording | Wording not justified |
|---|---|---|---|
| A. Context adds clear value | H+G beats matched H in untouched projects with meaningful paired effects; predictive and decision results distinguished | “Verified change features improved [probability score/proxy cost] over historical information on the evaluated CI cohorts.” | “Contextual bandits improve production deployment.” |
| B. Context helps under specific conditions | Prespecified cost/risk/workflow strata with enough independent replication and controlled selection | “Context improved proxy decisions in the specified conditions; other conditions did not show that benefit.” | A general operating boundary inferred from a post-hoc winning cell. |
| C. Bandits add no benefit beyond contextual probability models | Tuned comparisons with precision sufficient to exclude a meaningful benefit, or clear inferiority, under the declared full-information model | “Selected-arm LinUCB/Thompson offered no additional practically meaningful benefit over the contextual probability approach in these conditions.” | “Bandit algorithms are generally unnecessary.” |
| D. Simple failure-rate models suffice within evaluated conditions | Meaningful, verified context was actually available; strong supervised alternatives tested; equivalence/noninferiority or clear superiority supports the scoped statement | “Historical rate models matched or outperformed the tested alternatives within these datasets, costs and feedback assumptions.” | Sufficiency concluded from missing features, two exports, or p>0.05. |
| E. Evidence is inconclusive | Missing context/provenance, too few independent projects/failures, unstable intervals or mixed results | “The study cannot yet determine the incremental value of context or learning architecture.” | Interpreting uncertainty as proof of no effect. |

These are not mutually exclusive: A or B can coexist with C. The current evidence supports descriptive superiority of certain simple rules over the tested implementations on the frozen fixtures, and **E for meaningful change-context value and research generalization**. There is no contextual supervised benchmark yet, so C has not been tested. If a selected-arm algorithm wins a future full-information comparison, report that scoped implementation result, then test model capacity, regularization and feedback-use explanations; a win would not turn the environment into a bandit problem.

## 8. Should this remain a bandit paper?

**Not as its primary formulation.** The most coherent direction is a study of delayed CI risk prediction and cost-sensitive decisions, with selected-arm bandit algorithms among the comparators. First establish why these hypothetical decisions are a useful scientific abstraction. Do not manufacture label restrictions solely to retain the bandit label.

A bandit-centered paper would require a genuinely partial-feedback application contract or an explicitly synthetic benchmark addressing a distinct methodological question. The action-dependent drift simulator could support the latter in principle, but its assumed canary multiplier and blocked labels do not establish relevance to production. A separate bandit benchmark would need its own fair feedback-aware baselines and independent contribution.

The negative result itself is not automatically novel. Correcting errors in one's own implementation is necessary research hygiene. A stronger contribution would establish a reproducible, independently replicated explanation of when apparent complexity gains arise from cost mapping, history, genuine context, model choice, or information discarded by the learner. This review does not assume those broader results will emerge.

## 9. Is the current evidence sufficient for a research paper?

**NO for the proposed general empirical research claims.** The code and audit are a useful artifact and could support a tightly scoped technical case report. They do not yet support a submission-strength conclusion about the value of meaningful change context, contextual bandit learning, or production deployment.

The next work should be gated in this order: settle decision unit and target; establish the feedback/availability contract; repair collection provenance and obtain usable context plus independent projects; lock the matched design and inferential criteria; only then run the new study. Manuscript rewriting should follow the results. If the data gates cannot be met, report the two-export case study and its limits rather than expanding seeds or claiming that simple policies are generally sufficient.
