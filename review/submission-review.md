# Submission-level technical review

**Verdict on public main a437abf7ecd51e1e9da50834c72b56628e33c697, before corrections: Reject, 2/10. Submission readiness: NO — significant research work remains.** The central inference does not survive strong simple baselines, and several results or explanations depend on implementation errors. The corrected revision is a more credible simulation study, but the changes do not create deployment data, establish a novel algorithm, or validate production savings.

Review date: 6 September 2026. The active checkout was fetched and reset to public main as requested. The earlier local manuscript edit was backed up before resetting. The original public-main manuscript, figures, supplements, and headline files are preserved under `review/original/`; those archived numbers are historical evidence. Current numbers are in `experiments/results/headline/corrected/`. No changes have been published or pushed.

## Scope and explanation

I read the manuscript and its numerical appendices, traced the headline configuration-to-result paths, inspected the active algorithms, loaders, delay buffer, evaluation and statistical routines, and checked the ingestion, feature, storage, and legacy experiment surfaces. I reproduced the original real-data traces and headline arithmetic, constructed adversarial regressions, and reran the affected current experiments after corrections. Stubbed and unused legacy paths were identified rather than represented as validated production components. The exact smoke generator and original real-data collection manifest were not available; the missing information is explicitly retained as a limitation.

**ELI5.** The system chooses whether to release a change, test a limited release, or hold it. A bad release is expensive, but holding a good one also costs something. The paper asks whether a learning system makes cheaper choices than fixed rules. On these examples, a very simple rule that estimates how often builds fail is often cheaper. Some originally impressive results also came from letting the learner see information too early or from a broken alarm detector.

**Technical explanation.** The active replay computes `C(chosen_action, logged_CI_label)` and reveals the resulting cost after a delay. Disjoint LinUCB fits a linear negative-cost model for each arm and adds an exploration bonus; Thompson Sampling samples Gaussian action-value parameters. Static and heuristic policies use fixed feature thresholds. This is cost-sensitive because safe blocking and failed deployment have different penalties. The evaluation supplies an action-independent proxy label, not observed counterfactual deployment outcomes. Under that assumption, estimating one failure probability and minimizing three expected-cost expressions is a natural, strong baseline. The contribution is an engineering application and an empirical comparison, not a new contextual-bandit method or a new principle of asymmetric decision-making.

## Executive findings and corrected numbers

| Comparison | Original record | Corrected result | Interpretation |
|---|---|---|---|
| Synthetic default | Static 1878; LinUCB 1879 | Static 1884; LinUCB 1863; Bayesian rate 1713.5 | Scalar estimation wins; contextual necessity not shown |
| High failure cost | Static 2564; LinUCB 2078.5 | Static 2578; LinUCB 1972; always block 1865 | Learning beats weak static comparator but loses to a constant action |
| Low safe-block cost | Static 1584; LinUCB 1163.5 | Static 1591; LinUCB 1072; always block 1005 | Same baseline problem |
| Real proxy export | Static 644.5; LinUCB 669.5; Thompson 624.2 ± 71.8 | Static 640; LinUCB 779; Thompson 665.7 ± 39.8; rolling cost rule 624.5 | Corrected ranking changes; all policies use 578 resolved labels |
| PH smoke ablation | 44 alarms/resets; full worse than no-drift | Zero resets; full = no-drift = no-delay = 1863 | Original false-alarm interpretation invalid |
| Binary-reward ablation | 2469 versus 1879 (+31.40%) | 2586 versus 1863 (+38.81%) | Changing objective and reward scale is not a clean novelty ablation |

The original gain arithmetic is correct: `(2564-2078.5)/2564 = 18.94%` and `(1584-1163.5)/1584 = 26.55%`. The issue is what these numbers establish. The corrected improvements over the same static comparator are 23.51% and 32.62%, yet neither wins against always blocking. The original synthetic fixture contains 860 successful and 290 failed rows; those counts alone imply block costs 1865 and 1005 under the two gain matrices. This falsification does not depend on an ML implementation or random seed.

Sources: `corrected/summary.json`, `corrected/cost_sweep.json`, `corrected/ablation.json`; original equivalents under `review/original/headline/`. Complete per-seed values, intervals, action traces, resolved configs, and code/data hashes are retained.

## Critical issues

### C1 — Outcome history leaked unfinished builds

**Problem.** The original loader ordered by build start but appended outcomes with start/commit time. An earlier-started build's final label entered `recent_failure_rate` before its completion. Current-run `tests_run` and duration were also supplied as pre-decision context.

**Why a reviewer cares.** Online validity requires feature availability, not just delayed reward updates. A correct pending-reward buffer does not prevent a label from leaking through context. If the decision instead occurs after CI, that CI label is already observable and cannot simultaneously be treated as unknown delayed deployment feedback.

**Evidence.** `review/evidence/original-regression-failures.txt` records the red regression: build A starts at 00:00 and fails at 01:00; build B starts at 00:01; B receives failure rate 1 instead of 0. A third build after completion sees the expected rate. The real export has 129 decisions overlapping a strictly earlier-started unfinished build, or 309 when same-start preceding rows are included. A finish-time-only intervention changes 129 rate values in the original records; that count includes window-membership changes and is not a count of changed actions. The smoke fixture has no overlaps, so the overlapping-label defect alone does not invalidate its numbers.

**Fix made.** `data/loaders.py::iter_records` uses completion-time history, excludes unresolved labels from the rate, and uses historical completed-run CI metrics. The main replay is explicitly a pre-CI proxy simulation. Artificial delay features obey the artificial reveal schedule.

**Verification and affected scope.** Overlap and artificial-history regressions pass. All real replay results require regeneration; smoke results require regeneration for historical CI features and corrected clock semantics even though overlap itself is absent. Cost sweep, robustness, ablation, and all dependent figures were rerun. The separate synthetic environment's noisy hidden-risk feature was also replaced with observed history in the drift runner. The regression proves the bug; unchanged costs under a limited intervention do not exonerate it.

### C2 — Missing cost-aware baselines invalidate the main interpretation

**Problem.** The paper compares cost-aware learning mostly with cost-insensitive static thresholds and overlooks constant actions and scalar expected-cost rules.

**Why a reviewer cares.** A gain from respecting the cost matrix does not establish a gain from context, bandit exploration, or separate per-arm models. The original “operating envelope” could be a weak-comparator effect.

**Evidence.** Always block beats the originally reported bandit in both gain settings. In corrected runs, a Beta(1,1) scalar estimator costs 1713.5 on smoke versus LinUCB 1863 and Thompson 1857.1. It beats the contextual policies at every tested cost-sweep level. On the real export, the rolling cost rule costs 624.5 versus Thompson 665.7 and LinUCB 779. Bias-only LinUCB also improves to 650 on real data, directly challenging the value of the supplied context.

**Fix made.** Added `policies/cost_rules.py`, included all three constant actions, rolling expected cost, Bayesian scalar rate, and bias-only LinUCB in the main 30-seed study. A separate scalar method-of-moments control accounts for the drift simulator's known canary multiplier.

**Verification and remaining work.** The expected-cost boundary test verifies the canary interval and invariance to uniform cost scaling. Results and traces are public artifact candidates. To rescue a contextual-learning contribution, use independently selected features and hyperparameters on held-out projects, compare against a calibrated supervised probability model, and show a benefit beyond scalar controls. Calling the Beta estimator “non-learning” would itself be incorrect: it learns one scalar online.

### C3 — The Page-Hinkley implementation generated spurious baseline drift

**Problem.** Original PH used `mean = .9999*mean + .0001*cost` from zero, an unfaded cumulative sum, no minimum sample rule, and different initialization/reset behavior from the reference variant. A positive stationary cost therefore looked like a long-lived upward change.

**Why a reviewer cares.** The 44 alarms are not evidence about a properly implemented detector's false-positive rate. Calling them “expected sensitivity” protects a bug instead of evaluating an algorithm.

**Evidence.** A constant stream of 2 for 1,150 observations causes exactly 44 original alarms; corrected code produces zero. The corrected smoke ablation also produces zero resets. `tests/test_submission_regressions.py` compares each update and reset against an independent batch-mean computation of the River upward recurrence. `review/evidence/original-regression-failures.txt` preserves the original failure.

**Fix made.** `drift/detectors.py` now uses a sample mean, fading cumulative statistic, initial minimum infinity, minimum 30 samples, and explicit reset semantics. Larger threshold means fewer alarms. Project resets clear detector/model counters; internal drift resets preserve pending feedback and cumulative counts. “No reset” no longer reports detections as model resets.

**Reference limits.** The [River source](https://raw.githubusercontent.com/online-ml/river/main/river/drift/page_hinkley.py) was inspected directly; equivalence is to its upward variant, including first-observation minimum and reset-on-next-update behavior. The cited Mouss paper is [ASCC 2004, DOI 10.1109/ASCC.2004.184970](https://doi.org/10.1109/ASCC.2004.184970), not the venue given in the old bibliography. The available original-paper PDF endpoint failed to fetch; exact historical-variant equivalence is not claimed. Fading and minimum-sample defaults are reference implementation choices, not universal PH definitions.

**Calibration and verification.** Independent fixed-deploy null streams, 30 seeds at each of failure rates .10 and .35, still produce mean alarms 3.47 and 8.47 at threshold 50. Threshold 400 is the first in the documented grid meeting the exploratory criterion of at most 10% of streams with any alarm in both cells. It triggers no resets in the 500-step drift study and therefore matches plain LinUCB. Calibration removes sensitivity as well as alarms; it does not establish useful drift adaptation. The old “44 false alarms” headline must be withdrawn, not renormalized.

### C4 — The real-data explanation contradicts project-local execution

**Problem.** The paper attributes LinUCB's aggregate excess to confusing Requests with Flask, using the aggregate block fraction as though it were Requests' fraction.

**Why a reviewer cares.** This is an incorrect mechanistic explanation, not merely cautious wording around a small sample.

**Evidence.** Original `run_online_experiment` resets every policy for every project. Original Flask: LinUCB 511, 245 blocks, static 484.5. Original Requests: LinUCB 158.5, 8 blocks, static 160. The aggregate excess is Flask +26.5 plus Requests -1.5. `review/evidence/controlled-original-attribution.json` includes per-decision UCB predictions, exploration widths, selected actions, and costs.

**Controlled diagnosis.** Changing only alpha from 1 to 5 in the original implementation reduces Flask blocks from 245 to 112 and cost from 511 to 355, while increasing Requests cost to 233.5. Correcting only history timestamps leaves both original LinUCB total costs unchanged. These interventions support project-local sensitivity to reward scale, exploration width, selected-arm feedback and feature representation. They do not prove a unique cause or a general tuning rule. The negative-cost optimistic prior and fixed alpha are mathematically visible in the UCB scores; there is no information flow between project models to support the manuscript's explanation.

**Fix and verification.** Replaced the explanation with the project decomposition and controlled interventions. Corrected full replay now gives Flask 495 and Requests 284; Requests selects 160 canaries, so this new excess is not the old blocking phenomenon. A project-stream invariance regression protects independent stochastic execution. Future causal attribution needs controlled, held-out ablations, not post hoc project narratives.

### C5 — The deployment interpretation exceeds the observed data

**Problem.** The real CSV records CI outcomes, assumes a deployment action, and invents costs of unchosen actions. It has 600 workflow rows but only 156 distinct project/SHA pairs, and 25 exact duplicate rows. Branches, workflow IDs, run IDs, attempts, collection commands and sample-frame metadata are missing.

**Why a reviewer cares.** CI failure is not release failure, repeated workflows are not independent releases, and blocking/canarying changes which downstream outcomes are observable. Counterfactual deployment savings are unidentified from this export.

**Evidence.** `review/evidence/original-audit-summary.txt` records counts and feature ranks. Flask: 70 failures/300 rows or 70/295 resolved; Requests: 16/300 or 16/283 resolved. Four nonconstant feature dimensions plus bias give rank five; claims of only a failure rate and bias are inaccurate. Dataset times span roughly 99 days and 20 days. No actual policy action or logging propensity is supplied.

**Fix made.** All current writing labels the experiment CI proxy simulation and identifies rows, unique SHAs, missing outcomes and missing provenance. Added exact-duplicate and first-per-commit sensitivity runs with 30 seeds. Exact-unique: 575 rows, static 639, LinUCB 770, Thompson 668.8, rolling rule 622.5. First-per-commit: 156 rows, static 223, LinUCB 217, Thompson 215.1, rolling rule 199.5. Neither deduplication rule is a validated deployment-unit reconstruction.

**Required research and verification.** Collect an auditable workflow-aware cohort and deployment outcomes with timestamped feature availability. Record costs, action decisions, propensities if randomized, incident/rollback attribution, and the treatment of blocked changes. Hold out independent repositories or deployment streams. These data cannot be manufactured by a code fix.

### C6 — Timing, censoring, and drift evaluation had inconsistent estimands

**Problem.** Replay converted duration in seconds into a count of subsequent builds, ignoring actual interarrival times. Censored rows were charged when BLOCK was chosen but omitted for other actions. The drift runner advanced the environment before using the old context, counted tail feedback as censored, and indexed curves by feedback arrivals. Regret also separately filtered observed and oracle costs, misaligning pairs.

**Why a reviewer cares.** These affect learning, denominators, action comparisons, drift alignment and the quantity being plotted. A curve annotated “drift at 250” is invalid if its horizontal coordinate counts reward arrivals rather than decisions.

**Fix made.** Replay uses actual finish/start event times; artificial delays are explicitly separate. Censoring is a common missing-label cohort. Drift selects actions on current contexts for exactly 0..T-1, uses deterministic action identifiers, evaluates terminal feedback, and records expected pseudo-regret against an expected hidden-state oracle. `evaluation/metrics.py` jointly masks cost/oracle pairs. Project and mode resets are isolated.

**Verification.** Clock, common-censoring, current-context/horizon and metric-alignment regressions pass. All affected replay and drift studies and figures were regenerated. The corrected PH threshold-50 cost is slightly lower than plain LinUCB on stationary simulations (538.83 versus 541.13), so the old “net-negative in all three modes” statement is false after correction. Expected pseudo-regret is slightly worse there (55.03 versus 53.50), which also illustrates why realized noise and expected regret must be distinguished.

## Major issues

### M1 — Unsupported theory and cost-sweep boundary

**Problem → importance.** There is no theorem proving a 169-updates-per-arm convergence floor or a necessary 40:1 ratio. Treating empirical traces as mathematical conditions materially overstates the contribution.

**Evidence.** The original sweep has static/LinUCB costs 1598/1486, 1535/1440, 1878/1879, 2564/2078.5, 4622/2330. Advantage decreases toward 20:1 before increasing; it is not globally monotone, and gains already occur at 5:1 and 10:1. The cost matrix changes canary failure and block-bad penalties as well as deploy failure. Dividing T by three does not count actual arm updates. [Chu et al.](https://proceedings.mlr.press/v15/chu11a.html) provide bounds under a specified linear model, not this asserted floor. [Joulani et al.](https://proceedings.mlr.press/v28/joulani13.html) and [Pike-Burke et al.](https://proceedings.mlr.press/v80/pike-burke18a.html) concern explicit delayed-feedback settings; their results cannot be transferred by analogy alone.

**Fix → verification.** Withdraw the universal envelope, convergence floor, unjustified delay bound and equal-arm arithmetic. The revised paper derives only the affine expected-cost boundaries, states assumptions and cites established methods. Verify any future guarantee by identifying an actual theorem, its assumptions, and a complete mapping to the implemented algorithm and feedback model. This work has no substantial new formal guarantee.

### M2 — Statistical claims confused bootstrap tails, equivalence and replication

**Problem → importance.** The original “p < .0001” was inferred from zero uncentered bootstrap means above LinUCB; this is not the stated centered paired test. CI inclusion was called a statistical tie. Thirty random policy seeds were treated as broad robustness, and deterministic policies were said not to need uncertainty.

**Evidence.** Recomputed original real Thompson mean 624.2, sample SD 71.7841, 95% CI [598.1829,648.4167]. The repository's centered paired bootstrap with seed 42 and 10,000 resamples gives p=.00039996 versus LinUCB and p=.11588841 versus static. A nonsignificant difference is not an equivalence test. Algorithm seeds do not sample repositories. The original synthetic mean/SD 1903.4/54.1 is the current 30-seed value; older pilot values must not be mixed into it.

**Fix → verification.** Corrected artifacts save all seed values, ddof=1 SDs, paired differences, centered add-one p-values and Holm decisions for the three declared within-condition comparisons. Corrected Thompson minus static on real data is +25.6833, 95% seed CI [12.1325,40.1167], p=.00049995; Thompson minus LinUCB is -113.3167, CI [-126.8675,-98.8833]. These remain exploratory conditional results. A strong submission needs independent environment/project replicates, meaningful effect sizes, an equivalence margin if claiming equivalence, and planned comparison families. Do not claim bootstrap-seed invariance.

### M3 — Objective and reward-scale confounding in the ablation

**Problem → importance.** Binary outcome reward removes action-specific costs and changes reward scale while holding alpha fixed. It cannot isolate the contribution of a proposed cost-sensitive algorithm.

**Evidence.** `BinaryRewardBandit` uses -1 on failure and 0 otherwise for every action. The CI proxy label is action independent, so all arms have the same expected binary reward even when their operational costs differ. Original +31.4% and corrected +38.8% are genuine measurements of this altered objective, not proof that a new component adds value.

**Fix → verification.** Relabel the ablation as an objective intervention, add scalar cost baselines, and remove “dominant novel component” claims. A future clean study should include common reward normalization with appropriately scaled exploration/noise parameters, cost-aware supervised prediction, and independently tuned alternatives. Current exploratory alpha sensitivity is evidence of sensitivity, not a validated selection procedure.

### M4 — Reproducibility depended on unavailable files and inconsistent artifacts

**Problem → importance.** Public main omitted the smoke CSV, lacked its generator, and relied on ignored result arrays for figures. Documentation mixed pilot and 30-seed results and inverted delay labels.

**Fix made.** Frozen smoke and real CSVs with hashes are under `data/fixtures/`. `experiments/reproduce_submission.py` writes all corrected runs, compressed project traces, arrays, aggregates and provenance. `paper/build_corrected.py` consumes only corrected results; `paper/build_documents.sh` builds the manuscript and supplement. Old active headline filenames are updated and originals archived. Exact tested Python package versions and root-relative commands are documented.

**Verification and limit.** The complete run reproduced 10,284 result files byte-for-byte in an isolated directory without preexisting output files, using the same Python installation; the added windowed-drift run was independently repeated there as well. Unit tests and deterministic reproduction checks accompany the artifact. Frozen data repair numerical replay reproducibility, not generator provenance or data collection reproducibility. A fresh dependency installation and a separate machine/container execution remain stronger checks than rerunning in this existing environment. Do not score the artifact as 10/10 because local commands work.

### M5 — Off-policy evaluation cannot be claimed from deterministic CI logs

**Problem → importance.** Deterministic deploy logs have no support for canary/block; a zero IPS contribution for every unsupported action can look like zero cost. Thompson's returned `1.0` is not its marginal selection probability.

**Evidence and fix.** `evaluation/replay_eval.py` now rejects the obvious all-propensity-one log when the deterministic target chooses an unsupported action; the adversarial regression previously returned a misleading result. This guard is not a general support estimator. CI proxy replay is labeled separately. SNIPS/DR remain unimplemented, and actual deployment OPE is not established. Verify future OPE on a randomized logger with known conditional probabilities and overlap, and validate the target's probability calculation. Monte Carlo can estimate Thompson action probabilities; claiming they are impossible in principle is incorrect. The current API must not be used as a production propensity logger.

### M6 — Synthetic drift still has strong, unvalidated assumptions

**Problem → importance.** Canary both reduces failure probability by 60% and has a smaller failure consequence. A blocked change still supplies its counterfactual label. Hidden state determines informative noisy context, and only one family of schedules/horizons is studied.

**Evidence.** `environment/synthetic.py` defines the 0.4 multiplier. The corrected drift study uses a distinct expected-cost oracle and scalar exposure-adjusted control. Mean expected pseudo-regret for LinUCB versus that scalar control is 53.50/23.20 stationary, 65.10/163.15 abrupt, and 68.07/73.78 gradual. However, adding a scalar estimator with a 50-label sliding window reduces mean regret to 34.09 stationary, 47.53 abrupt, and 36.41 gradual, beating default LinUCB in all three. Paired differences have 95% bootstrap intervals [-29.93,-9.48], [-27.98,-7.10], and [-43.18,-21.55], respectively. These are exploratory comparisons added during the audit, not confirmatory results. Even the apparent abrupt-drift advantage over a lifetime scalar estimator does not survive a simple windowed baseline. The calibrated PH threshold produces no adaptation benefit because it never resets in these runs.

**Fix → verification.** Document assumptions and retain complete per-seed regret/cost evidence. Test different change magnitudes, horizons, delay/censoring mechanisms and action effects on independently selected seeds before generalizing. Retain the implemented 50- and 100-label scalar controls, and evaluate feature corruption/removal on independent scenarios. Costs of an adapting policy are not stationary merely because the external environment is stationary, so every alarm cannot automatically be called false.

### M7 — Incomplete legacy implementation and ingestion defect

**Problem → importance.** The repository presents a broader controller than the validated experiment path implements. Several modules are placeholders or refer to removed legacy APIs. The GitHub collector varied `per_page` as the remaining requested count shrank, causing duplicate pages for nonmultiples of 100.

**Evidence.** `policies/offline_classifier.py`, `policies/epsilon_greedy.py`, `environment/replay.py`, `environment/delays.py`, `delayed/imputation.py`, and parts of `drift/adapt.py` contain unimplemented paths. Legacy learning/storage paths reference older schemas or imports. These are not hidden implementations of the paper's missing baselines. `ingestion/github_client.py::_collect_commits` and `_collect_workflow_runs` now keep a fixed page size; a 150-item regression verifies no duplicate range. This does not retrospectively prove how the exported fixture was collected.

**Fix → verification.** Current README defines the tested research path, identifies stubs, and does not advertise a production deployment system. Complete or retire unused modules before release as a controller. Add integration tests for persistent schema/foreign-key behavior, ingestion event/attempt selection and a genuine observation-to-action lifecycle if making system claims. These were not needed to recompute the paper and are not claimed complete.

### M8 — Limited novelty and incomplete related-work framing

**Problem → importance.** Existing cost-sensitive decision theory, bandit algorithms, adaptive software engineering, and release-risk studies cover much of the conceptual contribution. A different action vocabulary is not algorithmic novelty.

**Fix → verification.** Narrow the contribution to the specific simulation study and audit evidence, add the comparisons below, and remove claims that JIT prediction is exclusively offline or that canary cost tradeoffs/feedback loops are new. A publishable empirical contribution must demonstrate a nontrivial result beyond weak baselines, with stronger deployment validity or a clearly framed replication/negative-results contribution.

## Claim ledger for the original paper

Classifications apply to the original wording, not merely whether a JSON file contains a number. “Proven” is reserved for a deterministic calculation or invariant under explicit assumptions.

| Original claim | Classification | Evidence / required interpretation |
|---|---|---|
| Buffer means no future information reaches decisions | INCORRECT | Buffer invariant alone holds; context leaks unfinished outcomes |
| Removing delay improves smoke cost by 1.1% | WEAK | Original artificial clock; corrected event-time no-delay/full both 1863 |
| 44 valid PH false alarms on stationary smoke | INCORRECT | Constant positive input reproduces initialization bug; corrected zero |
| No-reset wrapper equals LinUCB | WELL-SUPPORTED | Corrected action/cost identity in all three drift modes; no claim about unused statistics |
| Thompson has seed variance, LinUCB is deterministic | WELL-SUPPORTED | Conditional on fixed data, configs and tie breaking; not population uncertainty |
| Real Thompson CI [598,648] | WELL-SUPPORTED | Recomputed original interval; now superseded |
| Original paired p<.0001 | INCORRECT | Actual centered test p=.00039996 |
| 17-22 censored rewards per trajectory, about 3% | INCORRECT | Raw censoring is 5 and 17 per project, 22 total; policy-dependent old skip counts confound this |
| Binary reward is 31% costlier | WELL-SUPPORTED numerically | 31.40% old arithmetic; mechanistic novelty attribution unsupported |
| Low-risk Requests over-blocking explains 3.8% aggregate excess | INCORRECT | Requests had 8 blocks and was cheaper; Flask accounts for excess |
| Thompson beats LinUCB by 6.8% on real data | SUGGESTIVE | Original conditional point estimate 6.77%; leakage/censoring and narrow sample limit inference |
| Thompson and static are statistically tied | INCORRECT | CI inclusion does not establish equivalence; corrected static cheaper |
| 19% high-failure and 27% low-block savings | WELL-SUPPORTED arithmetic; WEAK contribution | Constant blocking already cheaper than the bandit |
| Globally monotone cost advantage | INCORRECT | Old and corrected five-point tables are nonmonotone |
| 120s severe / 30s short delays | INCORRECT | Divisor interpretation inverted; synthetic decision counts are not wall-clock delay |
| Default smoke tie is project cancellation | SUGGESTIVE | Original deterministic decomposition, not stable general result |
| Posterior never collapses / convergence after T/3 arm updates | UNSUPPORTED | Finite horizon cannot prove “never”; arms have unequal counts; no such convergence theorem |
| Bandit benefit requires at least 40:1 | INCORRECT | Gains already at 5:1 and 10:1, and scalar controls dominate |
| PH resets are net-negative in every drift mode | INCORRECT as maintained claim | Corrected stationary realized mean reverses sign; reset calibration changes behavior |
| Requests 5.3% failure rate | PROVEN for raw row denominator | 16/300; resolved fraction is 16/283=5.65%; not the whole dataset's rate |
| High costs and informative context define a general operating envelope | UNSUPPORTED | No identification of necessary/sufficient conditions; missing baselines |
| Project-adaptive alpha avoids over-blocking | SUGGESTIVE mechanism, UNSUPPORTED guarantee | Exploratory controlled alpha change helps Flask and harms Requests |
| Longer trajectories amortize reset overhead | UNSUPPORTED | Plausible hypothesis, untested; frequent resets can prevent amortization |
| 20:1 is an operating boundary | INCORRECT | One accidental near-tie, unsupported boundary inference |

## Mathematical audit

1. **Objective and sign:** summing costs and updating action models with negative cost is consistent. Nonnegative finite cost validation was strengthened. The cost matrix is a hypothetical utility model; it is not estimated from measured incidents.
2. **Expected cost:** with action-independent binary labels, deploy is `10p`, canary `1+3p`, block `2-1.5p`. Pairwise equality gives deploy/canary at 1/7 and canary/block at 2/9. Deploy/block cross at 4/23, but canary is cheaper there; this crossing is not an optimal-policy boundary.
3. **Oracle:** knowing the realized label is stronger than knowing its probability. A realized oracle deploys successful rows and blocks failed rows under default costs; an expected oracle can choose canary. The corrected drift pseudo-regret uses the latter with the simulator's action-specific probabilities.
4. **LinUCB:** ridge matrices and negative-cost updates implement the stated disjoint rule. Solving linear systems is preferable to explicitly inverting matrices. Fixed alpha does not adapt automatically to cost rescaling; increasing costs changes exploitation relative to exploration. Collinear/empty features weaken statistical interpretations without breaking ridge invertibility.
5. **Thompson:** Gaussian precision/mean updates and the Cholesky covariance sampling direction are internally consistent. The likelihood is misspecified for discrete unequal-variance costs; prior/noise calibration is not empirically established. No finite-horizon convergence threshold follows from feature dimension alone.
6. **Delayed feedback:** pending contexts must be the originating contexts and updates must arrive after observation. Actual timestamps now determine replay availability. Delay feedback after a model reset may be stale relative to the new regime; preserving it is an explicit choice rather than a universal optimum.
7. **Regret:** corrected metric alignment and expected pseudo-regret remove two distinct mistakes. Observed cumulative cost, realized clairvoyant regret, expected pseudo-regret and an optimal learned policy's regret are different objects and cannot share an unlabeled axis.
8. **Formal guarantees:** there are no substantial new formal guarantees. Cost optimality of a threshold given a known probability is a simple decision-theoretic derivation, not a theorem about learning that probability under real release feedback.

## Implementation mapping and code assessment

| Method/claim | Active implementation | Assessment |
|---|---|---|
| Pre-action context | `data/loaders.py::iter_records` | Corrected completion visibility; depends on supplied metadata validity; O(n²) history scans limit scale |
| Feature vector | `policies/base.py::FeatureEncoder`, `features/extractor.py` | Fixed normalization avoids fitted train/test leakage; normalizers do not guarantee bounded inputs; sparse/duplicated functionality |
| LinUCB cost learning | `policies/linucb.py` | Core update and selection consistent; reward-scale assumptions unvalidated |
| Gaussian TS | `policies/thompson.py` | Core posterior algebra consistent; likelihood and propensity limitations remain |
| Delayed feedback | `delayed/buffer.py`, `evaluation/online_replay.py` | Buffer mechanics tested; corrected event clock and common cohort now essential to interpretation |
| Cost-aware reset wrapper | `policies/cost_sensitive_bandit.py` | Same LinUCB without detector; corrected empty-buffer injection, reset/counter semantics; legacy internal stats are not the source of headline action counts |
| PH detector | `drift/detectors.py::PageHinkleyDetector` | Corrected against named upward reference; calibration is scenario specific |
| ADWIN/DDM alternatives | `drift/detectors.py` | Not used to support paper results; custom ADWIN assumptions/input bounds not comprehensively validated |
| Drift dynamics | `environment/synthetic.py`, `experiments/run_drift_eval.py` | Corrected action/context/oracle/tail handling; strong counterfactual assumptions persist |
| Costs and regret | `rewards/cost_model.py`, `evaluation/metrics.py` | Cost mapping explicit; finite validation and joint masking fixed |
| Statistical claims | `evaluation/statistical.py` | Centered paired bootstrap and percentile intervals saved explicitly; seed inference only |
| Offline evaluation | `evaluation/replay_eval.py` | Obvious zero support rejected; does not establish general overlap or causal validity |
| Experiment reproducibility | `experiments/reproduce_submission.py` | Recreates current numerical record from fixtures; generator/collection provenance still missing |
| GitHub pagination | `ingestion/github_client.py` | Fixed varying page size; workflow/run selection remains a research design issue |
| Knowledge base | `knowledge_base/` | Useful normalized schema direction, but legacy interface/schema drift and foreign-key enforcement need integration validation |
| Production controller | Legacy experiments and stub modules | Incomplete; not a demonstrated deployed system |

The strongest code component is the explicit typed context/action/reward separation and the pending-reward mechanism with deterministic tests. The weakest part is the gap between those local contracts and end-to-end availability/measurement, plus unfinished alternative paths. Passing tests originally established local behavior but omitted exactly the adversarial overlapping-build and constant-positive-stream cases that mattered.

Minor improvements include making every hyperparameter reject NaN/infinity, specifying tie handling and zero-history semantics in the API, avoiding misleading “lagged” comments on synthetic projections, consolidating feature normalization, avoiding quadratic history scans, validating timestamp order and malformed numeric input, removing dead imports and legacy wrappers, and pinning the external reference version rather than relying only on a moving main URL. These do not compensate for the critical design issues.

## Closest related work and novelty assessment

The search was refreshed during the audit through 6 September 2026. Primary papers, author manuscripts, official project source, and operational documentation were used. This is a focused literature comparison, not an exhaustive systematic review; newly surfaced preprints are not treated as peer-reviewed evidence.

| Prior work | What it did | Difference here and surviving novelty |
|---|---|---|
| [Elkan, 2001](https://cseweb.ucsd.edu/~elkan/rescale.pdf) | Established minimum-expected-cost decision principles for unequal error penalties | A deployment cost matrix is an application; scalar cost thresholds are not new |
| [Li et al., 2010](https://arxiv.org/abs/1003.0146) | Contextual-bandit recommendation with linear action models | Different application/action semantics; disjoint LinUCB itself is inherited |
| [Kamei et al., 2013](https://posl.ait.kyushu-u.ac.jp/~kamei/publications/Kamei_TSE2013.pdf) | Change-level risk prediction and effort-aware quality assurance | Here actions are deploy/canary/block rather than inspection effort; change-risk features and cost tradeoffs are established |
| [Cabral and Minku, 2022](https://research.birmingham.ac.uk/en/publications/towards-reliable-online-just-in-time-software-defect-prediction/) | Online JIT prediction with verification latency and concept drift, evaluated on ten GitHub projects | Different output/action objective; broad “offline JIT lacks feedback/drift” novelty is false |
| [Spieker et al., RETECS](https://arxiv.org/abs/1811.04122) | Reinforcement learning for CI test prioritization/selection | Different actions and reward; RL for CI optimization is not new |
| [Kerzazi and Adams, 2016](https://mcis.cs.queensu.ca/publications/2016/saner_noureddine.pdf) | Empirical analysis of botched releases and rollback-related decisions in a commercial web application | Here labels are CI proxies, weaker in deployment validity; this work adds a bandit simulation, not first recognition of release risk |
| [Google SRE canarying guidance](https://sre.google/workbook/canarying-releases/) | Operational risk control through partial rollout and monitored evaluation | Formalizing canary as a costed arm is useful implementation detail, not invention of the tradeoff |
| [McClendon et al., 2025 preprint](https://arxiv.org/abs/2503.22595) | Bandit strategies for ML model deployment/management | Different from CI code releases, but broad “first bandits for deployment” claims fail; preprint status matters |
| [Kephart, 2005](https://dominoweb.draco.res.ibm.com/reports/rc23692.pdf) | Research challenges for autonomic systems driven by high-level objectives | Feedback-loop/MAPE-K framing is established architecture; no new autonomic-control contribution demonstrated |
| [Joulani et al., 2013](https://proceedings.mlr.press/v28/joulani13.html) | General online learning with delayed feedback | A buffer is an implementation mechanism; a new delay-learning result requires more |
| [Pike-Burke et al., 2018](https://proceedings.mlr.press/v80/pike-burke18a.html) | Delayed aggregated anonymous bandit feedback | Their observation model differs from labeled action-linked CI rewards; importing its guarantee is unjustified |

The novelty that survives is modest: a particular, now auditable three-action simulation and a negative result about comparator choice. A strong negative-results or replication paper may be possible, but that is a different contribution from an established operating envelope for adaptive deployment.

## Scores and recommendation

Scores refer to the original public-main submission, as requested, before the changes made during this audit. They are reviewer judgments, not computed metrics.

| Dimension | Score / 10 | Reason |
|---|---:|---|
| Novelty | 2 | Established methods; missing strong baselines undermine empirical distinction |
| Technical correctness | 3 | Core bandit algebra reasonable; end-to-end leakage and detector errors |
| Mathematical rigor | 2 | No new guarantee; unjustified convergence and boundary claims |
| Code quality | 5 | Clear typed core and tests; incomplete/legacy surfaces and missed integration invariants |
| Experimental methodology | 3 | Structured comparisons but weak controls, proxy labels and timing mismatch |
| Statistical validity | 3 | Recomputable seeds/CIs; incorrect p-value/equivalence language and wrong replication unit |
| Real-world validation | 2 | Two short workflow exports, no deployment outcomes or causal action data |
| Reproducibility | 4 | Headline files partly public, but smoke data/generator/curve inputs/environment incomplete |
| Related work | 3 | Basic algorithms cited; important online JIT and deployment work insufficiently distinguished |
| Writing clarity | 5 | Readable structure, but confident explanations contradict implementation |
| Strength of contribution | 2 | Main contextual-learning inference defeated by simple controls |
| Overall research quality | 2 | Multiple issues affect central claims |

**Recommendation: Reject.** “Strong Reject” would be defensible for the unsupported theory and leakage, but the repairable implementation and available evidence warrant a normal Reject rather than an allegation of misconduct. The final test run is **200 passed, 3 skipped**; the skipped tests are explicitly identified legacy placeholders, including two stale skip labels for components covered elsewhere. The corrected artifact improves execution reproducibility to approximately **7/10** and technical trust substantially; it does not raise the present research claim to acceptance. No inference about author intent follows from these defects.

## Prioritized work before submission

**P0 — must fix before submission.** The code-level timing, censoring, detector and evaluation defects are fixed and affected experiments rerun. Do not restore the old headline values. The remaining P0 work is to choose an honest contribution: either a carefully bounded simulation/negative-results paper, or collect independent deployment evidence supporting a stronger claim. Strong cost-aware and supervised controls, valid observation contracts, unit-of-analysis reconstruction and defensible uncertainty could change a reviewer decision.

**P1 — should fix before submission.** Recover or document collection and generator provenance, establish clean environment reproduction, tune all competing methods with a separate protocol, test cost/reward-scale and missingness sensitivity, and validate the intended deployment mechanism. These improve credibility but cannot rescue a claim already beaten by a constant action.

**P2 — would strengthen the paper.** Broader projects, longer trajectories, varied drift mechanisms, held-out feature ablations, stateful release effects, safety-constrained exploration and practitioner-elicited costs. Complete the unused controller interfaces only if they become part of the claimed contribution.

The strongest skeptical criticism is: “Your method learns a base failure rate under a hand-chosen cost matrix, compares against rules that ignore that matrix, and calls CI replay deployment evidence; why is a contextual bandit needed?” The corrected artifacts currently support that criticism for the replay experiments.

## What this paper is about in 5 sentences

The paper asks how to choose between deploying, canarying, and blocking a change when mistakes have different costs. It applies established contextual-bandit algorithms to that choice. Most experiments derive simulated costs from CI outcomes. Correcting timing and detector errors changes several reported results. Simple cost-aware controls outperform the contextual policies on the main replay fixtures.

## Actual contribution

An auditable three-action simulation and a negative empirical result about weak comparators; no new bandit algorithm or proven deployment operating boundary.

## Strongest result

The baseline falsification: always blocking already beats the advertised gains, and a scalar estimator improves the corrected synthetic replay further.

## Weakest result

The original 44 Page-Hinkley false alarms and its claimed explanation; a constant positive stream reproduces the implementation defect.

## Strongest part of the code

Typed separation of context and feedback, coupled with a testable pending-reward buffer.

## Weakest part of the code

End-to-end observation availability and evaluation semantics were not covered by the original tests; several advertised alternative paths remain stubs.

## Biggest methodological risk

Mistaking a gain over a cost-insensitive comparator for evidence that contextual learning is necessary.

## Biggest threat to validity

CI workflow outcomes and hypothetical counterfactual costs do not identify production deployment benefit.

## Biggest novelty concern

The main decision principle and algorithms are established, and simpler controls reproduce the replay gains.

## Experiment I would add first

A prospective, workflow-aware held-out deployment study comparing a calibrated probability-plus-cost rule, scalar/windowed controls, and bandits under the same observable feedback and independently selected settings.

## Most likely reviewer criticism

The bandit is unnecessary for the reported replay gains, and the available real data does not measure the claimed deployment objective.

## My reviewer score

2/10, Reject, for the original public-main submission.

## Submission readiness

**NO — significant research work remains.**

## Top 5 things I would fix before submitting

1. Establish deployment-valid outcomes, timing, actions and cost measurements, or explicitly submit a narrow simulation/negative-results study.
2. Retain the simple cost-aware controls and add independently tuned supervised and windowed-rate controls.
3. Preserve corrected timing, common censoring, PH calibration and decision-indexed evaluation with regression tests.
4. Obtain independent project-level evidence and a valid uncertainty/selection protocol.
5. Finish provenance and clean-environment reproduction, and keep every manuscript claim aligned with the corrected artifacts.
