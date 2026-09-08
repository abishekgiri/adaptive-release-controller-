# Selective labels and the initial scientific estimand

**Proposal only; no model or policy evaluation performed.** The initial defensible task is delayed supervised prediction under the existing approval process, with explicit selection and observation limits. The [primary target](outcome-definition.md) is a protected release job's execution result, not an upstream CI conclusion or production incident.

## Formal observation model

Index immutable admission requests by i. Let X_i be context genuinely received and sealed while the gate is held; U_i denotes information used by reviewers but unrecorded for research. Let A_i=1 mean approval and A_i=0 mean denial. Pending/unresolved requests are distinct states, not coded as denial. Let T_i denote approval time and E_i the relevant environment state. Define Y_i(1,t,e) as the execution result **if that job executes after admission at time t in environment e**. This notation describes missing counterfactuals; it is not a claim of causal identification. If execution would not occur, the execution outcome is undefined unless the target is explicitly reformulated.

Let S_i indicate that the approved job starts; R_i indicate that its terminal success/failure/timeout label is observed and linked correctly; Q_i indicate that a valid pre-admission snapshot was captured. For the binary endpoint, a usable labeled example requires O_i = Q_i A_i S_i R_i = 1. Cancellation, gate timeout and supersession remain separate, and may cause R_i=0 for this endpoint. We observe Y_i=Y_i(1,T_i,E_i) only when the natural execution label is available. An approval record alone does not ensure O_i=1.

The complete-case predictor initially estimates:

`p_obs(x) = P(Y_exec=1 | X=x, Q=1, A=1, S=1, R=1; existing approval process)`.

It does **not** generally estimate the risk if every admission request were approved. A reviewer may reject changes with risks summarized in U_i; waiting may change E_i; capture Q_i can favor long waits; missing results R_i may correlate with severe disruptions. These are separate selection mechanisms. Restricting to approved releases does not remove them—it defines a narrower study population. Report capture and follow-up coverage before using the shorthand “approved-release prediction.”

The selective-label literature establishes why decisions that suppress labels complicate comparison with a replacement policy. Our release-specific model and restrictions here are deductions from this project's observation contract, not an application of a published correction theorem. [Lakkaraju et al., KDD 2017](https://www.cs.cornell.edu/home/kleinber/kdd17-selective.pdf).

## What each formulation permits

| Formulation | Valid starting use | Conditions and forbidden inference |
|---|---|---|
| Prediction among historically approved releases | Descriptive rates and, with authentic historical feature cutoffs, temporal prediction within approved, observed execution cases | Disclose Q/S/R selection, policy changes, missingness and correlated attempts. Cannot claim accuracy on denied releases or reconstruct snapshots from current mutable metadata |
| Prospective shadow prediction | Capture X before unchanged approvals, hide scores from reviewers, later score observed job labels | Desired initial empirical route; only if context is ready without delaying existing decisions. Generalization remains limited to observed selected populations |
| Observational decision-support research | Describe reviewer workflow, available information, risk communication requirements and hypothetical recommendations | Showing recommendations can affect A and timing and needs separate authorization. Observational scores do not prove improved decisions; interviews measure perceptions, not operational benefit |
| Selective-label learning | Model/diagnose selection, report support and sensitivity or partial bounds | Additional assumptions or data are required for correction. Reviewer variation alone is not random assignment; no automatic instrument from reviewer identity |
| Off-policy evaluation | Possible future evaluation when actual chosen-action rewards, action support and an identification strategy exist | Requires consistency, appropriate no-unmeasured-confounding/randomization assumptions, positivity for target actions and handling of delayed/missing rewards. Existing traces do not establish these conditions |
| Positive-unlabeled learning | Not the default | Approved executions contain observed successes **and** failures; denied cases are missing counterfactual executions, not a sample with only labeled positives. Rebranding does not create the usual labeling-mechanism assumptions |
| Survival/censoring methods | Potentially appropriate for duration or time-to-health-event after an actually executed release, under a separately specified endpoint | Denial is not merely a long right-censored execution of the same workload. Cancel/supersede may be informative competing events. Independent censoring cannot be assumed |
| Full-information cost-sensitive replay | Can score all actions for a stipulated cost matrix on the *observed approved subset* if losses are defined as functions of its observed label | This is hypothetical accounting in a selected population, not actual counterfactual denial utility or full-population policy value |

Bandits are not required to ask whether X predicts Y. They would become candidates only after real actions and chosen-action rewards are defined and legitimately observable. A denial's opportunity cost may itself be unobserved, so even a standard bandit reward contract need not hold. “Action-dependent feedback” alone does not establish that a bandit algorithm is appropriate.

## Identification limits and useful diagnostics

For illustration, temporarily posit a well-defined binary execution outcome Y(1) for every request under a fixed admission regime. Let q be the fraction of the full request frame with an observed such label and p_obs its failure fraction. Without assumptions about unobserved outcomes:

`q * p_obs <= P(Y(1)=1) <= q * p_obs + (1-q)`.

These worst-case bounds follow by assigning all missing binary outcomes first zero, then one. They do not fix the more basic problem of requests that cannot execute under the proposed regime. If q=0.8 and p_obs=0.02, the bounds are 1.6% to 21.6%. **This is an illustration, not measured project data.** Even a low rejection fraction can conceal most failures; a rejection percentage alone cannot grade selection severity.

With later authorized data, tabulate request/approval/start/capture/follow-up rates by project, time, gate version and predecision risk strata; compare X distributions for approved/denied/missed requests; inspect deterministic acceptance regions; and bound sensitivity to unknown labels. Never generate denied labels with the model being evaluated. A later successful retry is a different time/environment/possibly artifact, not the original denied counterfactual. An independent sandbox execution may produce a proxy label, not the real protected-stage outcome under approval.

To identify broader admission risk by weighting, one would need a justified condition such as `Y(1) independent of A given X`, nonzero approval probability in every relevant context, consistent versions/timing, and an additional valid model for Q/S/R. These conditions are presently **unestablished**. A fitted propensity score is not evidence of ignorability or overlap, and extreme weights cannot recover contexts that are never approved. Doubly robust estimators address specific nuisance-model errors under their assumptions; they do not repair missing support, an undefined reward or unrecorded decision drivers. [Dudík, Langford and Li, ICML 2011](https://icml.cc/2011/papers/554_icmlpaper.pdf).

## Probability prediction versus cost-sensitive decisions

For a hypothetical, specified one-step loss matrix, let approving cost c_AF on failure and c_AS on success; let denial cost c_D regardless of the unexecuted outcome. Then:

`L_approve(p) = p*c_AF + (1-p)*c_AS`, and `L_deny = c_D`.

If c_AF>c_AS, denial minimizes this *stipulated* expected loss when `p > (c_D-c_AS)/(c_AF-c_AS)`; thresholds outside [0,1] imply a constant action. This derivation needs no bandit. If denial cost depends on whether the release would succeed, actual lost benefit and the counterfactual failure risk are required instead; they are not observed merely because a job was denied. Real delay, emergency releases, downstream dependencies and future retries may also invalidate the one-step loss model.

For initial decision-support feasibility, ask owners to identify actual failure costs, false-denial harms, acceptable alert/review burden and situations in which a warning could change a decision. Keep cost elicitation separate from claimed measured utility. Do not import the old CI cost matrix. A predictor that mostly forecasts random infrastructure errors may have little value for denial even with measurable discrimination; waiting/retry support might be a different later task, not a silently added action.

A conservative hypothetical rule that only denies a subset of historically approved cases can have its flagged-case failure rate measured on those observed labels. This still does not establish the real effects of suppressing those releases or justify approving historically denied ones. Publishing a shadow recommendation log is not off-policy value identification.

## Claim ladder and recommendation

1. **Measurement first:** demonstrate a meaningful protected operation, pre-admission context and a reliably linked execution label.
2. **Prediction next, only after ratification:** quantify added contextual value beyond scalar stage/project history in observed approved executions; report uncertainty and scoped null results.
3. **Decision support later:** evaluate whether authorized reviewers interpret/use the information and how decisions change; separate these outcomes from release health.
4. **Enforcement/value last:** separately authorize actions and establish reward/identification requirements. No automatic inference from predictive success to saved costs or production benefit.

Selection is structurally serious for replacing admission decisions, but need not prevent a useful bounded approved-case prediction study. Its empirical severity is **UNKNOWN** because no complete gate request frame or owner data is available. If complete-case selection cannot be quantified or the approved population has too little consequential variation, stop rather than claim that a correction algorithm solved it.
