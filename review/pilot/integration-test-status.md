# Integration obligations and their actual status

The test suite in `tests/test_pilot_integration.py` executes no policy, optimizer,
model selection or holdout access. It uses real archived public API sources plus
clearly synthetic adversarial timing fixtures. Passing a fixture does not turn
that fixture into a real predecision observation.

| Original obligation | Pilot status | What is actually demonstrated |
|---|---|---|
| 1. Delivery order, duplicate delivery, clock skew | PASS for REST adapter | Reordering authentic response views preserves canonical observations; backward collector time rejected. Webhook authenticity/delivery behavior is not certified. |
| 2. Attempt pagination and time partitions | PASS for implemented data paths | Synthetic >1,000-run windows and overlapping boundaries; real attempt-specific expansion. Rate-limit retries are implemented but a live rate-limit exhaustion episode was not induced. |
| 3. PR head/base isolation | PARTIAL | Unknown tested revision or merge basis is rejected; current PR metadata is not promoted. A positive, authentic predecision PR head/base/tested-ref integration remains absent. |
| 4. Truncation, binary files, renames | PASS | Incomplete inventories are null; uncertain text counts remain null; rename path roles retained; duplicate file records rejected. |
| 5. Workflow refs and dynamic jobs | PASS for supported parser, incomplete acquisition | Unverified refs rejected; literal top-level declarations parsed; dynamic runner values/reusable dependencies stay null. Actual executed job counts cannot fill predecision fields. |
| 6. Transitive source lineage | PASS for implemented adapter | Raw hashes and receipt-journal membership checked; future/cross-repository references and altered convenience views rejected. Does not externally authenticate the collector's clock. |
| 7. Future-label mutation | PASS for data-history path | Future labels and earlier provider completions do not affect earlier histories; a later legitimate arrival can enter. |
| 8. Predictor export isolation | PASS | Actual SQL export excludes outcome/job joins; mutation test uses populated fixture tables, not only the empty real export. |
| 9. Fold/group/scaler isolation | DEFERRED, not yet applicable | No training folds, learned transforms or model runner are implemented/executed in this data-only pilot. Still required before model GO. |
| 10. All ten methods consume identical packets | DEFERRED, not yet applicable | No ten-method runner is executed. Data-only history tests cannot certify learner behavior. |
| 11. Delay perturbation across all lineage | PARTIAL | Added delay removes every affected outcome-derived history signal in the data helper. Complete delayed source/feature/runner integration remains required before delay experiments. |
| 12. Gate precedes actual model import/fit | DEFERRED, not yet applicable | Collector imports no policy/model. Existing frozen preflight still reports NO-GO; legacy headline runners have not been rewired in this pilot. |
| 13. Fresh source-to-dataset roundtrip | PASS for archived corpus | Separate virtual environment, pinned wheel/hash, offline transform; SQLite and logical dump hashes match. No authentic prospective rows exist to certify that path in the field. |

Additional tests enforce: reexports preserve reruns and same-SHA workflows;
canceled/missing conclusions have no binary label; conflicting identities are
quarantined; API access is limited to the named development projects; a real
inconsistent attempt is quarantined; raw-source/derived-value agreement;
postdecision cached diffs are rejected; and a positive synthetic preexecution
snapshot carries all 30 candidates with honest missingness.

No missing test is represented by a passing placeholder. Full prospective
PR capture and the actual evaluation runner remain unverified. These omissions
are part of the NO-GO conclusion, not a claim that all 13 obligations are complete.
