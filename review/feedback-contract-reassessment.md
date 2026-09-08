# Feedback contract reassessment

**Recommended framing after a pivot: downstream release-admission decision support with selectively observed outcomes.** This is a proposal, not a ratified experiment. Live enforcement remains UNVERIFIED; no model was run.

## Boundary-by-boundary feedback

| Boundary / actual action | Predicted target | Observed outcome | Does action change availability or outcome? | Can all action losses be reconstructed? | Appropriate formulation |
|---|---|---|---|---|---|
| v1 shadow scores while every CI attempt runs unchanged | Eventual unsuccessful CI completion | Delayed CI label independent of simulated choice | Not under v1's replay assumption | Yes, for its stipulated label-to-loss matrix only | Delayed supervised prediction / cost-sensitive full-information learning. Does not demonstrate deployment control. |
| A: invoke or suppress CI before creation | Result the candidate test workload would produce if invoked | Test result only if invoked; skip/deferral recorded otherwise | Yes; suppression removes the natural test label | No in general. Compute cost may be known, missed-failure cost is not. | Selective-label learning / active information acquisition; bandit formulation possible only with a well-defined observed reward and exploration/identification conditions. |
| B: admit, hold or reject every job before execution | Counterfactual CI result under admission | Admitted attempts execute; held/rejected attempts have administrative states | Yes; rejection and delay can change labels | Not from rejection status. A gate-created failure is not a workload-failure observation. | Selective feedback or partial monitoring; no automatic full-information reduction. |
| C: run/skip one expensive or sensitive job | That target job's outcome conditional on earlier work | Earlier CI always known; target label only if run | Yes for target label, potentially through queue/environment delay | Earlier CI does not determine all target-job losses | Delayed supervised prediction on selected runs; cost-sensitive admission with missing outcomes. Bandits only with additional reward/identification assumptions. |
| D: allow/deny downstream release after CI | Adverse release/health outcome for an immutable artifact and environment within a declared horizon | Earlier CI known; actual deployment telemetry only for executed releases; wait/deny recorded separately | Yes. Denial means no release outcome; canary changes exposure if implemented. | No. One CI label cannot identify deployment/canary/denial losses. | Decision theory plus selective-label risk prediction initially; genuine action-dependent bandit feedback can arise in a later properly instrumented study. |
| D with an independent unchanged reference test always executed | Future reference-test outcome, not production benefit | Reference label observed regardless of admission | Reference label may be independent if invariance is justified; release outcomes remain action-dependent | Only explicitly hypothetical costs based on that reference label | Full-information prediction for the reference-test task. This does not identify real release utility. |

The mechanics underlying admission are documented in [GitHub's deployment controls](https://docs.github.com/en/actions/how-tos/deploy/configure-and-manage-deployments/control-deployments). Statistical classifications above follow from the observation contract, not from the name of a GitHub action.

## Why moving the timestamp alone fails

At a post-CI release gate, the old CI conclusion has already been observed. Using it as input is legitimate for a new release-risk target; predicting that same already-known CI conclusion is not prospective prediction. Keeping the old headline target and treating known test results as newly useful context would produce a trivial or leaked evaluation.

For a real policy let `Y(a)` denote the relevant downstream outcome under action `a`, and let `O(a)` indicate whether its feedback is available. The frozen CSV assumption effectively supplies one action-invariant label `Y` and a declared function `L(a,Y)` for every action. Admission control generally changes `O(a)`; release/canary can also change `Y(a)`. Observing `Y(allow)` does not identify `Y(canary)`, and a blocked release provides no observed “would have failed” label. Potential-outcome notation describes missing quantities here; it does not claim causal identification.

Known components such as a fixed review fee or measured waiting time do not make the entire loss vector observable. A deny action may have a known direct cost, yet the lost opportunity or avoided failure component can remain unknown. Even a chosen-action reward may be incomplete when it requires an unobservable counterfactual; that situation needs reward redesign or partial-monitoring analysis before calling it an ordinary bandit.

## What would justify bandits?

A genuine bandit study would need: an implementable set of distinct actions; preaction context; chosen-action rewards whose observation rules and delays are defined; recorded actual action/override/propensity; adequate support for actions being evaluated; and defensible assumptions about interference, nonstationarity and observation censoring. Randomized logging or an explicit identification argument is needed for unbiased counterfactual policy evaluation; a deterministic legacy release rule usually lacks overlap. No random production exploration is proposed or authorized here.

Action-dependent feedback makes a bandit formulation possible, not automatically useful. A calibrated probability model with an expected-cost rule may remain preferable. If data come only from operator-approved releases, supervised performance is initially interpretable only within that selected population. Broad policy-value claims would require more evidence. Repeated defer/review/canary/escalate decisions could form a sequential-control problem, but merely having delayed feedback does not establish an MDP or justify reinforcement learning.

## Defensible first analysis after a separately approved pivot

Start with an observation/actuation audit and descriptive accounting of pending, approved, rejected, bypassed, overridden and unresolved requests. Separate deployment execution failures from independently monitored service-health violations. Preserve non-deployments as administrative outcomes, not successes or failures of an unexecuted release. Count missing telemetry and unequal follow-up.

Only after this record exists should one design a restricted supervised prediction study, assessed on future observed eligible releases with explicit selection limitations. An observational comparison of admission policies would not identify deployment benefit by itself. No prior CI replay number, prior seed distribution or full-information loss matrix establishes this new target's validity.
