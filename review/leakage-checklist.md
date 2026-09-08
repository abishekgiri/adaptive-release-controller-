# Leakage and information-integrity checklist

`ci-design-v1`. Mark a checkbox only with a linked test or data-quality artifact, never from a field name or docstring. Local validators cover declared fixture semantics; the unchecked source/collector items are required before GO.

- [x] Decision timestamps are aware UTC and precede verified attempt execution; post-start snapshots are rejected by the primary validator.
- [x] Current metadata from later receipt is rejected even if its underlying commit date is earlier.
- [x] Historical labels require strictly earlier **availability**, not merely earlier completion; equal-time decision batches see no equal-time feedback.
- [x] Overlapping slow/fast attempts reveal outcomes in availability order.
- [x] Current-attempt outcome lineage cannot masquerade as prior duration/history.
- [x] Feature allowlist, source kind, value type/range and missingness rules reject outcome/progress/final-duration inputs and fake zeros.
- [x] Empty-history rate is null with a denominator, not a fabricated zero-risk observation.
- [x] Exact export duplicates do not multiply decisions; conflicting duplicates are quarantined; attempts and workflows remain distinct.
- [x] Terminal redelivery preserves earliest receipt; conflicting labels require adjudication.
- [x] Same-SHA prior outcomes are usable only after observation, while current/future outcomes remain excluded.
- [x] Project-local history excludes other repositories; original projects cannot become evaluation data; related families cannot cross partitions.
- [x] Cancel/nonbinary labels produce no all-action loss vector; deadline rules match learning/history and scoring.
- [ ] Collector proves raw event authenticity, clock quality, source completeness and immutable first receipt; late polling is not backdated.
- [ ] Each feature's full source dependency graph is traversed and recomputed, including history manifests and exact Git refs.
- [ ] PR merge-base/head/tested revision and workflow-definition ref are correct for each trigger, without latest-state substitution.
- [ ] Diff truncation, binary files, renames and static/dynamic workflow definitions have adversarial extraction tests.
- [ ] Model input export is isolated from feedback/current job tables; a future-label mutation cannot alter earlier feature snapshots/predictions.
- [ ] Training folds, imputers, vocabularies, hashing, scaling, priors and calibration have no future/evaluation fitting.
- [ ] Related commits/reruns cannot cross independent development fold assignments; repeated-change prequential use is separately declared.
- [ ] All future learners receive exactly the same observation packets, horizon and resolution mask; secondary delay transformations also shift all history dependencies.
- [ ] Held-out researcher access is logged/sealed; no feature selection, salt search, sample-size extension or tuning follows evaluation results.
- [ ] Final source/code/protocol/data manifests are verified before the evaluation runner can launch.

The crossed items are fixture-tested local invariants, not certification of a corpus that does not yet exist. See [automated invariants](automated-invariants.md) for the executable tests and remaining integration work.
