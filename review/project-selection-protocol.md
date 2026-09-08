# Project selection and prospective precision protocol

`ci-design-v1`, 7 September 2026. Original Flask and Requests are **development only**. No new repository has been recruited or assigned by this task. Proposed counts are budget starting points; no count automatically establishes validity.

## Sampling frame and eligibility

Before model evaluation, define a versioned frame of public GitHub repositories with accessible GitHub Actions build/test workflows, a verifiable source license/access policy, and a feasible pre-execution event archive or participating collection arrangement. Identify workflows by immutable ID and prospectively inspect definitions for CI intent. Exclude release-only/deployment-only/labeling/scheduled maintenance workflows from the primary target; exclude unsupported events, archived/deleted repos lacking observation access, generated mirrors, and datasets without authentic event provenance. Do not choose repositories because a model wins on them.

Require evidence of at least six months of active CI history for recruitment planning and at least two functioning change-feature families in the development pilot. API backfill proves activity only, not historic collector label availability. Retain forks/mirrors/common upstreams under a family identifier; no family may cross partitions. Inspect shared ownership and common infrastructure; treat those as possible additional dependence and report sensitivity. Language/domain/workflow diversity should be strata in the recruitment frame, not post-hoc claims from whichever projects participate.

Do not screen by success-only runs or remove low-failure projects after collection. Historical aggregate failure counts can inform **feasibility planning before partitioning** if drawn from a disjoint earlier period with recorded access, but selection on those counts changes the target population and must be stated. Preferred selection is a prespecified stratified random order within the eligible participation frame, retaining the full invitation/eligibility/attrition ledger. Participation itself limits generalization.

## Three partitions and access boundaries

Initial budget scenario: five development repositories (including the two legacy projects only if newly collected under this contract), five distinct validation repositories, ten untouched evaluation repositories. This increases the earlier illustrative plan to reserve validation explicitly; it is not a power claim. Add repositories according to the precision plan, not a desired policy ranking.

- **Development:** design, collection pilot, feature admission, implementation checks, inner chronological tuning, variance planning. Existing exports cannot leave this partition.
- **Validation:** independent protocol rehearsal and one locked selection among a finite candidate set. If results cause an algorithm/feature/protocol revision, these repositories become development/rehearsal data and a fresh validation set is required; keep the evaluation set sealed.
- **Evaluation:** no researcher access to row labels, feature-label associations or model results before the final protocol/runner are frozen. A custodian may monitor prespecified collection-quality aggregates and execute the frozen runner; this does not permit adaptive model selection. Online models may receive matured labels within their own stream—that is the declared prequential test, not researcher tuning.

Use stable repository IDs and a frozen family map. Split assignment uses a public deterministic keyed hash order within recruitment strata after eligibility is locked; its salt and exact ordered frame are recorded before label access. The two original development assignments are forced. Never search salts for favorable label balance. Leave-one-project-out on the original two repositories is not a substitute for these partitions.

## Fixed periods, failures, and missingness

Proposed primary prospective period: six calendar months of decisions, followed by 30 days of label follow-up. Recruitment/pilot precedes that period; the exact UTC boundaries must be written to the run manifest before evaluation starts. The project model starts from declared development priors and updates prequentially from the first evaluation decision. Report the first 100 decisions separately as cold start, while retaining them in the primary estimand. No warm-up label exposure is silently granted to one model.

Aim for roughly 1,000 eligible first attempts per evaluation repository as an initial resource estimate. At 5% failure probability that is only 50 expected failed attempts, before missingness and within-change dependence; at 2% it is only 20. Obtain enough resolved failures for the intended precision by lengthening the planned calendar period **before evaluation is unlocked**, or recruiting additional projects. Never stop each project when the desired number of failures or a significant result occurs. A fixed calendar study that yields too few failures is reported as underpowered/inconclusive, not selectively pruned.

A subgroup calibration claim requires at least 50 resolved failures and 50 successes in the relevant group as a **reporting screen**, plus adequate interval width; this is not a guarantee. Do not remove groups below that screen from primary macro-cost or coverage tables. Long histories and many failures do not compensate for few repository clusters or missing context.

## Prospective precision and power-style calculation

The primary unit for cross-project inference is a paired **project mean difference**, not seed or workflow row. Normalize each cost by the maximum entry in its declared matrix; cost lies in [0,1], paired differences in [-1,1]. Let sigma be between-project SD of the paired difference and h the desired CI half-width. A normal planning approximation is

    J_precision = ceil((z_(1-alpha/2) * sigma / h)^2).

For a two-sided comparison detecting difference delta with 80% power, use

    J_power = ceil(((z_(1-alpha/(2m)) + z_0.8) * sigma / delta)^2), m=7.

The seven confirmatory contrasts are specified in the evaluation protocol. Bonferroni is used conservatively for planning; Holm is specified for analysis. These calculations assume independent project effects, a stable effect distribution and known sigma. They are optimistic normal approximations, especially for small J; actual planning must use a justified variance range with uncertainty, a small-sample t/noncentral-t calculation or justified cluster simulation, and sensitivity to family dependence. No current policy effect was fitted for this task.

The deterministic [planning script](design_precision.py), [grid](design/precision-sensitivity.csv), and [assumption record](design/precision-assumptions.json) perform **analytic sensitivity only**, not policy experiments. Selected examples:

| Assumed project SD | Target half-width/effect | Projects for individual 95% CI half-width | Approximate projects for 80% power, seven contrasts |
|---:|---:|---:|---:|
| .025 | .025 | 4 | 13 |
| .05 | .025 | 16 | 50 |
| .10 | .025 | 62 | 200 |
| .05 | .01 | 97 | 312 |

Thus ten evaluation repositories do not establish precision for modest differences. These are sensitivities to assumptions, not estimates of the required final sample. Brier-score context differences need their own variance and target; do not reuse cost variance. Algorithm seeds address Monte Carlo uncertainty inside each fixed project and cannot reduce the between-project sampling requirement.

Proposed margins pending human ratification: normalized cost .025 and Brier score .01; corresponding CI half-width targets .025/.01. They are research-scale conventions under hypothetical costs, not validated operational importance. Initial design approval must use justified external evidence or conservative variance bounds, not require prohibited policy experiments. The displayed SD grid is illustrative and is not a justified upper bound: bounded project contrasts can have SD approaching 1. If available evidence and budget cannot support a defensible range, remain NO-GO or choose explicitly exploratory scope. Following explicit design approval, development-only runs may refine the variance plan before evaluation access; that amendment must be frozen independently of evaluation results. Never use the original two-project effects as population variance estimates. A separately authorized data-only pilot can estimate accrual, missingness and failure prevalence without fitting any policy.

## Lock and no-peeking rule

Before evaluation access record: eligible frame, invitations/attrition, family map, split assignment, source/feature versions, fixed calendar window, minimum meaningful margins, planning assumptions, expected event counts, analysis family, and recruitment stop rule. Add whole repositories from the next prespecified frame positions if the **pre-evaluation** precision plan requires more. Do not enlarge the sample based on evaluation effects, CI widths containing a preferred sign, or running p-values. If budget prevents the planned precision, narrow the study to exploratory/case-study claims before accessing evaluation results.

Unresolved: collaboration/archival access, the target participation frame, final UTC periods, meaningful margins, justified initial variance assumptions and resulting evaluation J. These remain NO-GO gates, not blanks to fill after viewing results.
