# Automated invariants and integration obligations

`ci-design-v1`. Added implementation: `data/research_contract.py`, reference SQL `review/design/dataset-schema.sql`, and focused tests `tests/test_research_contract.py`. The tests use tiny synthetic records, no learning algorithms, no stochastic policy runs and no network. The legacy experiment code is untouched.

## Implemented and tested now

| Invariant | Exact regression test(s) |
|---|---|
| No unfinished overlapping outcome | `test_overlapping_attempt_observations_are_not_available_early` |
| Same-time decision batch excludes same-time feedback | `test_same_timestamp_feedback_is_after_decision_batch` |
| Use collector availability, not earlier completion | `test_feedback_uses_availability_not_earlier_completion` |
| Prior same-change label can be used only after observation | `test_same_sha_feedback_is_legitimate_only_after_observation` |
| Correct project and missing-label history | `test_history_excludes_other_projects_and_canceled_rows` |
| Primary histories do not silently mix the rerun population | `test_primary_history_does_not_mix_rerun_training_population` |
| Aware UTC and pre-execution snapshot | `test_non_utc_decisions_rejected`, `test_post_start_prediction_is_not_primary_eligible` |
| Forbidden inputs, source availability and current-label relabeling | `test_forbidden_features_rejected`, `test_later_metadata_cannot_be_backdated_to_decision`, `test_current_outcome_cannot_masquerade_as_prior_duration` |
| Explicit nulls; invalid values rejected | `test_null_and_zero_are_not_interchangeable`, `test_nonfinite_or_duplicate_feature_rejected`, `test_change_counts_enforce_their_type_and_range`, `test_empty_rate_requires_explicit_missingness` |
| History cannot be labeled change context; metadata matches unit | `test_history_cannot_masquerade_as_change_feature`, `test_model_metadata_cannot_disagree_with_attempt` |
| Idempotent duplicates without collapsing attempts/workflows | `test_export_duplicates_are_idempotent_but_attempts_are_distinct`, `test_two_workflows_same_sha_are_not_collapsed_to_latest` |
| Exact attempt linkage and immutable terminal semantics | `test_wrong_attempt_feedback_rejected`, `test_conflicting_feedback_requires_adjudication`, `test_terminal_redelivery_retains_first_observation_time` |
| Canceled/never-started/missing labels retain honest semantics | `test_nonbinary_outcomes_supply_no_action_loss_vector`, `test_never_started_cancellation_is_retained_without_binary_label` |
| Full-information cost reconstruction and explicit timeout label | `test_binary_label_reveals_all_losses_and_timeout_is_explicit_failure`, `test_json_protocol_matches_contract_and_requires_no_go` |
| Same fixed follow-up in scoring and history | `test_followup_deadline_is_common_to_scoring_and_history` |
| Development-only originals and family/repository split integrity | `test_current_projects_are_development_only`, `test_family_and_repository_cannot_cross_splits` |
| SQL uniqueness/FKs and conclusion-label constraints | `test_normalized_schema_enforces_repository_identity_and_foreign_keys`, `test_normalized_schema_retains_attempts_and_enforces_label_semantics` |
| Planning does not ignore effects of precision/multiplicity | `test_precision_planning_penalizes_more_contrasts_and_smaller_effects` |

Gate tests in `tests/test_design_gate.py` verify source mutation detection, explicit NO-GO, missing approval/evidence and incomplete repository assignments. The hash snapshot is integrity evidence, not an externally registered preregistration or cryptographic proof of researcher blinding.

## Exact tests required before connecting a real collector

These are **not yet implemented or passing**; no fake passing placeholders are included.

1. `test_collector_redelivery_out_of_order_and_clock_skew`: replay authentic development payload fixtures in different delivery orders; original authenticated receipt determines visibility and time uncertainty is flagged, never guessed.
2. `test_attempt_pagination_and_time_partition_completeness`: overlapping API boundary pages, >1,000 filtered runs, retries/rate limits and attempts >1 preserve exact IDs without gaps or double counts.
3. `test_pr_diff_uses_captured_head_base_not_latest`: later PR commits/default-branch movement leave sealed earlier features unchanged.
4. `test_diff_truncation_binary_and_rename_semantics`: partial pages cannot create complete totals; binary unknown counts propagate null; rename counted once with both path roles preserved.
5. `test_workflow_definition_ref_and_dynamic_jobs`: event-dependent definition refs, reusable workflows and dynamic matrices cannot import actual executed-job counts into static features.
6. `test_transitive_feature_lineage_closure`: reject any dependency whose source receipt is future/current-outcome or wrong unit; recompute exact value from all referenced source hashes.
7. `test_future_label_mutation_preserves_all_earlier_inputs`: changing every future label/status/late source object cannot affect earlier snapshot rows or history views.
8. `test_prediction_export_contains_no_outcome_or_current_job_join`: inspect exported schema/values and the actual query path, not only dataclass annotations.
9. `test_temporal_fold_group_and_transform_isolation`: move a later repeated SHA or extreme future feature/label; earlier fit/scaler/vocabulary/prior/calibration remain unchanged and boundary-straddling groups are counted as excluded.
10. `test_all_methods_receive_equivalent_packets_and_cutoffs`: a future runner records the exact original features, label timestamps and loss vectors consumed by each learner; no early tail updates or selected-method censor mask.
11. `test_delay_sensitivity_shifts_all_lineage`: perturbing feedback delays also shifts every outcome-derived feature and follow-up eligibility consistently.
12. `test_execution_gate_precedes_any_model_import_or_fit`: a blocked/stale approval or changed source/data manifest stops the **actual** headline runner before learning starts.
13. `test_fresh_environment_source_to_dataset_roundtrip`: archived authentic sources reproduce the declared normalized release and quality reports under pinned dependencies.

Passing local structural tests cannot authenticate data source semantics. These integration checks and a manually audited sample of real payload-to-feature histories are part of the collection GO gate.
