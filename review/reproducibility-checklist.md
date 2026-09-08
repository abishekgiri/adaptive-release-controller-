# Reproducibility checklist for the new study

`ci-design-v1`. This is distinct from the prior corrected experiments' numerical reproduction.

- [x] Written decision, label, feature, duplicate, split and evaluation contracts with explicit NO-GO state.
- [x] Proposed executable normalized SQL schema; no invented migration of incomplete CSV identifiers.
- [x] Focused synthetic-record contract tests, without policy execution.
- [x] Analytic precision sensitivity with saved assumptions and no fitted policy effects.
- [x] A hash manifest for this design snapshot; verification detects silent changes.
- [ ] Recruitment frame, eligibility/workflow decisions, invitation/nonresponse/attrition ledger and repository-family clusters archived.
- [ ] Stable project split registry with real provider IDs, exact time windows and salt registered before evaluation access.
- [ ] Read-only collection account/route, pinned and tested API version, retry/rate-limit/pagination behavior and checkpoints documented.
- [ ] Raw event/response objects with hashes, receipt time, completeness, event IDs and endpoint/query provenance archived under permitted access terms.
- [ ] Author/actor identity policy, pseudonymization method, lawful redistribution and immutable change/workflow references documented.
- [ ] Every candidate feature's missingness, cardinality, quantiles and source-availability proof generated on development data; admitted set frozen.
- [ ] Corpus version manifests include excluded/missing/duplicate records and denominator/coverage reports, not only successful resolved cases.
- [ ] Precision plan uses ratified meaningful margins and justified external evidence or conservative variance assumptions, with finite-project correction and dependence sensitivity; any authorized development refinement is frozen before evaluation access.
- [ ] All model implementations, candidate configurations, optimization/encoding details and selection tie rules are reproduced on development/validation only.
- [ ] Collector-to-schema-to-feature-to-feedback integration invariants pass; no unused validator is presented as end-to-end enforcement.
- [ ] Fresh environment/container with pinned dependencies recreates normalized data/features from archived source objects and all quality reports.
- [ ] On a second independent environment, hashes or declared numerical tolerances match; external API refetch is not required for the frozen reproduction.
- [ ] A single future approved command validates approval/protocol/data hashes before model execution and regenerates results/figures after GO.
- [ ] Final deviations and amendments are recorded; old versions remain available. Evaluation exposure invalidates claims of preregistration for later amendments.

Commands currently available (no policy runs):

```sh
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/test_research_contract.py tests/test_design_gate.py -p no:cacheprovider
python -m review.design_precision
python -m review.design_gate
```

The last command is expected to report NO-GO and exit 2 for this snapshot. It is a preflight helper for the new study, not a claim that every legacy runner has already been wired to it. There is deliberately no authorized headline reproduction command for the new design yet.
