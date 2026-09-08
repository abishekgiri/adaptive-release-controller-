"""Data-only integration tests: authentic archived API sources plus adversarial fixtures."""
import copy
import json
from pathlib import Path
from datetime import timedelta
import sqlite3
from urllib.parse import parse_qs, urlparse
import pytest

from review.pilot.collector import SourceArchive, checked_run, windowed_runs, dt, parse_http, ROOT, Client
from review.pilot.pipeline import (canonical_observations, diff_features, workflow_features,
    source_closure, history_features, shift_outcome_availability, normalize, predictor_export, build_snapshot)

PILOT=ROOT/'review/pilot'


@pytest.fixture
def acquired():
    return json.loads((PILOT/'acquisition-state.json').read_text())


def test_collector_redelivery_out_of_order_and_clock_skew(acquired, tmp_path):
    expected=canonical_observations(acquired)
    changed=copy.deepcopy(acquired)
    changed['rounds'].reverse()
    for rnd in changed['rounds']:rnd['observations'].reverse()
    assert canonical_observations(changed)==expected
    archive=SourceArchive(tmp_path)
    with pytest.raises(ValueError,match='clock'):
        archive.record('/repos/pallets/flask',200,{},b'{}','2026-01-02T00:00:00Z','2026-01-01T00:00:00Z')


def test_attempt_pagination_and_time_partition_completeness():
    base=dt('2026-01-01T00:00:00Z')
    rows=[{'id':i+1,'run_attempt':1,'created_at':(base+timedelta(seconds=i)).isoformat().replace('+00:00','Z')} for i in range(1205)]
    def get(endpoint):
        query=parse_qs(urlparse(endpoint).query);start,end=query['created'][0].split('..')
        found=[r for r in rows if dt(start)<=dt(r['created_at'])<=dt(end)]
        page=int(query.get('page',[1])[0]);size=int(query.get('per_page',[100])[0])
        part=found[(page-1)*size:page*size]
        return {'total_count':len(found),'workflow_runs':part},{'response_headers':{'link':'x; rel="next"' if page*size<len(found) else ''}}
    result=windowed_runs(get,'pallets/flask','2026-01-01T00:00:00Z','2026-01-01T00:30:00Z')
    assert len(result)==1205
    assert set(result)=={(i+1,1) for i in range(1205)}


def test_pr_diff_uses_captured_head_base_not_latest(acquired):
    samples=[s for s in acquired['samples'] if s['event']=='pull_request']
    assert samples
    assert all(not s['frozen_diff_basis_verified'] for s in samples)
    # Current PR file lists/unknown merge bases are never passed off as frozen diffs.
    rows,_,_=canonical_observations(acquired)
    assert all(r['tested_sha'] is None and not r['primary_eligible'] for r in rows if r['event']=='pull_request')


def test_diff_truncation_binary_and_rename_semantics():
    file={'filename':'tests/new.py','previous_filename':'src/old.py','status':'renamed','additions':2,'deletions':1,'patch':'@@ test'}
    complete=diff_features([file],True)
    assert complete['files_changed']==1 and complete['top_directory_count']==2
    assert complete['total_churn']==3 and complete['test_file_changed'] is True
    assert all(v is None for v in diff_features([file],False).values())
    binary=dict(file);binary.pop('patch')
    result=diff_features([binary],True)
    assert result['files_changed']==1 and result['additions'] is None
    with pytest.raises(ValueError,match='repeated'):diff_features([file,file],True)


def test_workflow_definition_ref_and_dynamic_jobs():
    import base64
    content=lambda s:{'encoding':'base64','content':base64.b64encode(s.encode()).decode()}
    config=content('jobs:\n  test:\n    runs-on: ${{ matrix.os }}\n    strategy:\n      matrix:\n        os: [ubuntu-latest, windows-latest]\n')
    assert workflow_features(config,True)=={'declared_job_count':1,'declared_runner_types':None}
    assert workflow_features(config,False)=={'declared_job_count':None,'declared_runner_types':None}
    unsafe=content('!!python/object/apply:os.system ["false"]')
    import yaml
    with pytest.raises(yaml.YAMLError):workflow_features(unsafe,True)


def test_transitive_feature_lineage_closure(tmp_path):
    archive=SourceArchive(tmp_path)
    one=archive.record('/repos/pallets/flask/commits/abc',200,{},b'{"sha":"abc","files":[]}',
        '2026-01-01T00:00:00Z','2026-01-01T00:00:01Z')
    one.update(repository_id=596892,purpose='diff')
    assert source_closure(archive,[one],'2026-01-01T00:00:02Z',596892)
    forged=dict(one,received_at='2025-12-31T00:00:00Z')
    with pytest.raises(ValueError,match='journal'):source_closure(archive,[forged],'2026-01-01T00:00:02Z',596892)
    with pytest.raises(ValueError,match='future'):source_closure(archive,[one],'2026-01-01T00:00:00Z',596892)
    with pytest.raises(ValueError,match='repository'):source_closure(archive,[one],'2026-01-01T00:00:02Z',1)
    child=archive.record('/repos/pallets/flask/history',200,{},json.dumps({'dependencies':[one]}).encode(),
        '2026-01-01T00:00:01Z','2026-01-01T00:00:02Z')
    child.update(repository_id=596892,purpose='history')
    assert source_closure(archive,[child],'2026-01-01T00:00:03Z',596892,history=True)


def history_rows():
    return [{'repository_id':1,'run_id':1,'attempt':1,'workflow_id':1,'tested_sha':'same','primary_eligible':True,
        'decision_at':'2026-01-01T00:00:00Z','started_at':'2026-01-01T00:00:01Z',
        'completed_at':'2026-01-01T00:02:00Z','label':1,'label_available_at':'2026-01-01T00:03:00Z'}]


def test_future_label_mutation_preserves_all_earlier_inputs():
    rows=history_rows();current=dict(rows[0],run_id=2,decision_at='2026-01-01T00:01:00Z')
    before=history_features(current,rows,'2026-01-01T00:00:00Z')
    rows[0]['label']=0;rows[0]['completed_at']='2026-01-01T00:00:30Z'
    assert history_features(current,rows,'2026-01-01T00:00:00Z')==before
    current['decision_at']='2026-01-01T00:04:00Z'
    after,_=history_features(current,rows,'2026-01-01T00:00:00Z')
    assert after['same_change_prior_resolved']==1


def test_prediction_export_contains_no_outcome_or_current_job_join(tmp_path):
    # Test the actual export query on a populated database, not an empty export alone.
    con=sqlite3.connect(':memory:')
    con.executescript('CREATE TABLE decisions(repository_id,run_id,attempt,decision_at,primary_eligible); CREATE TABLE feature_values(repository_id,run_id,attempt,feature_name,value_json,missing_reason,available_at); CREATE TABLE feedback(label); CREATE TABLE observed_jobs(conclusion);')
    con.execute("INSERT INTO decisions VALUES(1,2,1,'t',1)")
    con.execute("INSERT INTO feature_values VALUES(1,2,1,'files_changed','5',NULL,'t')")
    con.execute('INSERT INTO feedback VALUES(1)');con.execute("INSERT INTO observed_jobs VALUES('failure')")
    before=predictor_export(con)
    con.execute('UPDATE feedback SET label=0');con.execute("UPDATE observed_jobs SET conclusion='success'")
    assert predictor_export(con)==before and len(before)==1


def test_delay_sensitivity_shifts_all_lineage():
    rows=history_rows();current=dict(rows[0],run_id=2,decision_at='2026-01-01T00:04:00Z')
    before,_=history_features(current,rows,'2026-01-01T00:00:00Z')
    shifted=shift_outcome_availability(rows,600)
    after,_=history_features(current,shifted,'2026-01-01T00:00:00Z')
    assert before['project_resolved_count_7d']==1
    assert after['project_resolved_count_7d']==0 and after['prior_workflow_duration_s'] is None
    assert after['same_change_prior_resolved']==0


def test_fresh_environment_source_to_dataset_roundtrip(tmp_path):
    # Separate clean interpreter/venv execution is recorded in pilot evidence too.
    report=normalize(PILOT,tmp_path)
    expected=json.loads((PILOT/'schema-validation.json').read_text())
    assert report['logical_dataset_sha256']==expected['logical_dataset_sha256']
    assert report['sqlite_sha256']==expected['sqlite_sha256']
    assert report['foreign_key_violations']==[]


def test_reruns_multiple_workflows_reexports_and_canceled_runs(acquired):
    rows,dups,quarantine=canonical_observations(acquired)
    assert any(r['attempt']>1 for r in rows) and dups>0 and not quarantine
    assert any(r['conclusion']=='cancelled' and r['label'] is None for r in rows)
    groups={}
    for r in rows:groups.setdefault((r['repo'],r['head_sha']),set()).add(r['workflow_id'])
    assert any(len(g)>1 for g in groups.values())
    original=len(rows);changed=copy.deepcopy(acquired);changed['rounds']+=copy.deepcopy(acquired['rounds'])
    assert len(canonical_observations(changed)[0])==original


def test_missing_conclusions_and_conflicting_attempts_fail_closed(acquired):
    changed=copy.deepcopy(acquired);changed['rounds']=changed['rounds'][:1]
    first=copy.deepcopy(changed['rounds'][0]['observations'][0]);first['run']['conclusion']=None
    assert checked_run(first['run'],changed['repositories'][first['repo']]['metadata']['id'])=='missing_conclusion'
    changed['rounds'][0]['observations']=[first]
    rows,_,_=canonical_observations(changed)
    assert rows[0]['label'] is None and rows[0]['label_available_at'] is None
    conflict=copy.deepcopy(first);conflict['run']['head_sha']='different'
    changed['rounds'][0]['observations'].append(conflict)
    rows,_,quarantine=canonical_observations(changed)
    assert not rows and quarantine


def test_no_holdout_or_mutating_api_endpoint(tmp_path):
    client=Client(SourceArchive(tmp_path))
    with pytest.raises(ValueError,match='allowlist'):client.get('/repos/unknown/evaluation/actions/runs')
    with pytest.raises(ValueError):client.get('/repos/pallets/flask/../../secrets')
    assert parse_http(b'HTTP/2.0 200 OK\r\nDate: test\r\n\r\n{}')[0]==200


def test_collected_timestamp_inconsistency_is_quarantined(acquired):
    assert any('start predates creation' in error['error'] for error in acquired['errors'])


def test_cached_postdecision_diff_cannot_enter_snapshot(acquired):
    rows,_,_=canonical_observations(acquired)
    r=next(r for r in rows if r['event']=='push' and r['scope_eligible'])
    cfg=acquired['repositories'][r['repo']]
    # Hypothetical timing fixture using real captured sources: NOT a real eligible decision.
    r=copy.deepcopy(r);start=dt(cfg['metadata_source']['received_at'])
    r['created_at']=(start-timedelta(seconds=2)).isoformat();r['decision_at']=(start+timedelta(seconds=1)).isoformat()
    r['started_at']=(start+timedelta(seconds=2)).isoformat()
    # A genuine first-run summary arrives later than this fixture's cutoff; reject it.
    with pytest.raises(ValueError,match='future'):
        build_snapshot(r,rows,acquired,SourceArchive(PILOT))


def test_transformed_values_must_recompute_from_authentic_sources(acquired):
    from review.pilot.pipeline import validate_acquisition_state
    archive=SourceArchive(PILOT)
    validate_acquisition_state(acquired,archive)
    forged=copy.deepcopy(acquired)
    forged['samples'][0]['files'][0]['additions']+=100000
    with pytest.raises(ValueError,match='raw source'):validate_acquisition_state(forged,archive)


def test_synthetic_preexecution_snapshot_can_populate_all_candidates(tmp_path):
    # Transport-shaped fixture only; never counted as pilot observations.
    from review.pilot.collector import now
    archive=SourceArchive(tmp_path)
    metadata={'id':596892,'created_at':'2010-01-01T00:00:00Z','default_branch':'main','language':'Python'}
    meta=archive.record('/repos/pallets/flask',200,{},json.dumps(metadata).encode(),'2026-01-01T00:00:00Z','2026-01-01T00:00:01Z')
    run={'id':1,'run_attempt':1,'workflow_id':4,'head_sha':'abc','head_branch':'main','event':'push','status':'queued'}
    snap=archive.record('/repos/pallets/flask/actions/runs',200,{},json.dumps({'workflow_runs':[run]}).encode(),'2026-01-01T00:00:02Z','2026-01-01T00:00:03Z')
    current={'repo':'pallets/flask','repository_id':596892,'run_id':1,'attempt':1,'workflow_id':4,'tested_sha':'abc','branch':'main','event':'push',
      'created_at':'2026-01-01T00:00:02Z','decision_at':'2026-01-01T00:00:03Z','started_at':'2026-01-01T00:00:04Z','snapshot_source':snap}
    state={'repositories':{'pallets/flask':{'metadata':metadata,'metadata_source':meta}},'samples':[],'started_at':'2026-01-01T00:00:00Z'}
    packet=build_snapshot(current,[],state,archive)
    assert len(packet)==30 and packet['files_changed']['value'] is None
    assert packet['project_failure_rate_7d']['value'] is None
    assert packet['project_resolved_count_7d']['value']==0
    assert packet['workflow_identity']['value']=='596892:4'
