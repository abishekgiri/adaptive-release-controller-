"""Contract and data-integrity fixtures only; no policies or experiments run."""
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sqlite3
import json

import pytest

from data.research_contract import (AttemptKey, Decision, FeatureValue, Feedback,
    canonicalize_decisions, canonicalize_feedback, validate_decision,
    validate_features, validate_feedback, visible_history, loss_vector, validate_splits,
    evaluation_losses)
from review.design_precision import required_clusters

T=datetime(2026,1,1,tzinfo=timezone.utc)
HASH='a'*64


def decision(run=1, minute=1, repo=1, attempt=1, sha='abc'):
    return Decision(AttemptKey(repo,run,attempt),7,sha,'main','push',T,
                    T+timedelta(minutes=minute),T+timedelta(minutes=minute,seconds=1),HASH)


def feedback(d, minute=2, status='failure'):
    return Feedback(d.key,status,T+timedelta(minutes=minute),HASH,
                    T+timedelta(minutes=minute,seconds=-1))


def test_overlapping_attempt_observations_are_not_available_early():
    slow,fast,current=decision(1,1),decision(2,2),decision(3,4)
    labels=[feedback(slow,60),feedback(fast,3)]
    assert visible_history(current,[slow,fast],labels)==(labels[1],)


def test_same_timestamp_feedback_is_after_decision_batch():
    prior,current=decision(1,1),decision(2,2)
    assert visible_history(current,[prior],[feedback(prior,2)])==()


def test_same_sha_feedback_is_legitimate_only_after_observation():
    prior,current=decision(1,1,sha='shared'),decision(2,3,sha='shared')
    assert len(visible_history(current,[prior],[feedback(prior,2)]))==1
    assert visible_history(current,[prior],[feedback(prior,4)])==()


def test_feedback_uses_availability_not_earlier_completion():
    prior,current=decision(1,1),decision(2,5)
    f=replace(feedback(prior,10),completed_at=T+timedelta(minutes=2))
    assert visible_history(current,[prior],[f])==()


def test_history_excludes_other_projects_and_canceled_rows():
    a,b,current=decision(1,1),decision(2,1,repo=2),decision(3,4)
    assert visible_history(current,[a,b],[feedback(a,2,'cancelled'),feedback(b,2)])==()


def test_primary_history_does_not_mix_rerun_training_population():
    rerun,current=decision(1,1,attempt=2),decision(2,4)
    f=feedback(rerun,2)
    assert visible_history(current,[rerun],[f])==()
    assert visible_history(current,[rerun],[f],include_reruns=True)==(f,)


@pytest.mark.parametrize('stamp',[T.replace(tzinfo=None),T.astimezone(timezone(timedelta(hours=1)))])
def test_non_utc_decisions_rejected(stamp):
    with pytest.raises(ValueError,match='UTC'):
        validate_decision(replace(decision(),decision_at=stamp))


def test_post_start_prediction_is_not_primary_eligible():
    d=decision()
    with pytest.raises(ValueError,match='precede'):
        validate_decision(replace(d,decision_at=d.started_at))


@pytest.mark.parametrize('name',['outcome','current_duration_s','tests_passed','conclusion','risk_from_future'])
def test_forbidden_features_rejected(name):
    with pytest.raises(ValueError,match='forbidden'):
        validate_features(decision(),(FeatureValue(name,1,T,'metadata',HASH),))


def test_later_metadata_cannot_be_backdated_to_decision():
    with pytest.raises(ValueError,match='unavailable'):
        validate_features(decision(),(FeatureValue('files_changed',2,T+timedelta(minutes=2),'diff',HASH),))


def test_current_outcome_cannot_masquerade_as_prior_duration():
    d=decision()
    with pytest.raises(ValueError,match='current-attempt'):
        validate_features(d,(FeatureValue('prior_workflow_duration_s',2,T,'history',HASH,d.key),))


def test_null_and_zero_are_not_interchangeable():
    d=decision()
    validate_features(d,(FeatureValue('files_changed',None,T,'diff',HASH,missing_reason='incomplete_diff'),))
    validate_features(d,(FeatureValue('files_changed',0,T,'diff',HASH),))
    with pytest.raises(ValueError,match='zero'):
        validate_features(d,(FeatureValue('files_changed',0,T,'diff',HASH,missing_reason='incomplete_diff'),))


def test_history_cannot_masquerade_as_change_feature():
    with pytest.raises(ValueError,match='masquerade'):
        validate_features(decision(),(FeatureValue('files_changed',3,T,'history',HASH),))


def test_nonfinite_or_duplicate_feature_rejected():
    x=FeatureValue('files_changed',float('nan'),T,'diff',HASH)
    with pytest.raises(ValueError,match='finite'):
        validate_features(decision(),(x,))
    x=replace(x,value=1)
    with pytest.raises(ValueError,match='duplicate'):
        validate_features(decision(),(x,x))


@pytest.mark.parametrize('value',['failure',True,-1,{'outcome':1}])
def test_change_counts_enforce_their_type_and_range(value):
    with pytest.raises(ValueError,match='type/range'):
        validate_features(decision(),(FeatureValue('files_changed',value,T,'diff',HASH),))


def test_empty_rate_requires_explicit_missingness():
    with pytest.raises(ValueError,match='empty history'):
        validate_features(decision(),(
            FeatureValue('project_failure_rate_7d',0,T,'history',HASH),
            FeatureValue('project_resolved_count_7d',0,T,'history',HASH)))


def test_model_metadata_cannot_disagree_with_attempt():
    with pytest.raises(ValueError,match='identity'):
        validate_features(decision(),(FeatureValue('event_type','pull_request',T,'metadata',HASH),))


def test_json_protocol_matches_contract_and_requires_no_go():
    spec=json.loads(Path('review/design/evaluation-spec.json').read_text())
    assert spec['status']=='NO-GO' and not spec['seed_runs_authorized_now']
    assert len(spec['primary_contrasts'])==7
    for outcome,status in enumerate(('success','failure')):
        vector=loss_vector(feedback(decision(),status=status))
        assert list(vector.values())==[row[outcome] for row in spec['cost_matrices']['primary']]


def test_export_duplicates_are_idempotent_but_attempts_are_distinct():
    a=decision()
    b=decision(attempt=2,minute=3)
    assert len(canonicalize_decisions([a,a,b]))==2
    with pytest.raises(ValueError,match='conflicting'):
        canonicalize_decisions([a,replace(a,head_sha='different')])


def test_two_workflows_same_sha_are_not_collapsed_to_latest():
    a,b=decision(1),replace(decision(2),workflow_id=8)
    assert len(canonicalize_decisions([a,b]))==2


def test_wrong_attempt_feedback_rejected():
    d=decision()
    with pytest.raises(ValueError,match='exact attempt'):
        validate_feedback(d,replace(feedback(d),key=AttemptKey(1,1,2)))


def test_conflicting_feedback_requires_adjudication():
    f=feedback(decision())
    assert len(canonicalize_feedback([f,f]))==1
    with pytest.raises(ValueError,match='conflicting'):
        canonicalize_feedback([f,replace(f,conclusion='success')])


def test_terminal_redelivery_retains_first_observation_time():
    f=feedback(decision())
    late=replace(f,available_at=f.available_at+timedelta(minutes=5),source_sha256='b'*64)
    assert canonicalize_feedback([late,f])==(f,)


def test_never_started_cancellation_is_retained_without_binary_label():
    d=replace(decision(),started_at=None)
    f=replace(feedback(d,status='cancelled'),completed_at=None)
    validate_feedback(d,f)
    assert loss_vector(f) is None


def test_followup_deadline_is_common_to_scoring_and_history():
    a=decision()
    late=replace(feedback(a),available_at=T+timedelta(days=31),completed_at=T+timedelta(days=31))
    b=replace(decision(2),decision_at=T+timedelta(days=32),started_at=T+timedelta(days=32,seconds=1))
    assert evaluation_losses(a,late,T+timedelta(days=40)) is None
    assert visible_history(b,[a],[late])==()


@pytest.mark.parametrize('status',['cancelled','neutral','skipped','action_required','stale','startup_failure','unknown'])
def test_nonbinary_outcomes_supply_no_action_loss_vector(status):
    assert loss_vector(feedback(decision(),status=status)) is None


def test_binary_label_reveals_all_losses_and_timeout_is_explicit_failure():
    assert loss_vector(feedback(decision()))=={'deploy':10,'canary':4,'block':.5}
    assert loss_vector(feedback(decision(),status='timed_out'))==loss_vector(feedback(decision()))
    assert loss_vector(feedback(decision(),status='success'))=={'deploy':0,'canary':1,'block':2}


def split(repo,slug,group='g',part='development'):
    return dict(repository_id=repo,slug=slug,family_id=group,split=part)


@pytest.mark.parametrize('slug',['pallets/flask','PSF/Requests'])
def test_current_projects_are_development_only(slug):
    with pytest.raises(ValueError,match='development only'):
        validate_splits([split(1,slug,part='evaluation')])


def test_family_and_repository_cannot_cross_splits():
    with pytest.raises(ValueError,match='family'):
        validate_splits([split(1,'a/a'),split(2,'b/b',part='validation')])
    with pytest.raises(ValueError,match='duplicated'):
        validate_splits([split(1,'a/a'),split(1,'a/a','h','evaluation')])


def schema_db():
    db=sqlite3.connect(':memory:')
    db.executescript(Path('review/design/dataset-schema.sql').read_text())
    return db


def test_normalized_schema_enforces_repository_identity_and_foreign_keys():
    db=schema_db()
    with pytest.raises(sqlite3.IntegrityError):
        db.execute("INSERT INTO repositories VALUES(1,'psf/requests','r','evaluation')")
    with pytest.raises(sqlite3.IntegrityError):
        db.execute("INSERT INTO workflows VALUES(99,1,'ci.yml')")


def test_normalized_schema_retains_attempts_and_enforces_label_semantics():
    db=schema_db()
    db.execute("INSERT INTO repositories VALUES(1,'test/repo','r','development')")
    db.execute("INSERT INTO source_objects VALUES(?,1,'metadata','2026-01-01T00:00:00Z',NULL,NULL,NULL,'raw.json',1)",(HASH,))
    db.execute("INSERT INTO changes VALUES(1,'abc','parent','abc','first_parent',NULL,NULL,?)",(HASH,))
    db.execute("INSERT INTO workflows VALUES(1,7,'ci.yml')")
    for n in (1,2):
        db.execute("INSERT INTO attempts VALUES(1,1,?,7,'abc','abc','main','push','2026-01-01T00:00:00Z',NULL,?)",(n,HASH))
    assert db.execute('SELECT count(*) FROM attempts').fetchone()[0]==2
    with pytest.raises(sqlite3.IntegrityError):
        db.execute("INSERT INTO feedback VALUES(1,1,1,'cancelled','cancelled',0,'2026-01-02T00:00:00Z',NULL,?)",(HASH,))


def test_precision_planning_penalizes_more_contrasts_and_smaller_effects():
    base=required_clusters(.1,.025,power=.8)
    assert required_clusters(.1,.0125,power=.8)>=4*base-3
    assert required_clusters(.1,.025,power=.8,comparisons=7)>base
    with pytest.raises(ValueError):required_clusters(0,.01)
