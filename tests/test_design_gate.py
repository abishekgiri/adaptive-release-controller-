import json
from pathlib import Path

from review.design_gate import approval_errors, integrity_errors, sha, snapshot_digest, execution_blockers


def test_frozen_source_mutation_is_detected(tmp_path):
    p=tmp_path/'protocol.md';p.write_text('version one')
    files={'protocol.md':sha(p)}
    lock={'files':files,'snapshot_sha256':snapshot_digest(files)}
    assert integrity_errors(tmp_path,lock)==[]
    p.write_text('changed after freeze')
    assert integrity_errors(tmp_path,lock)==['changed or missing: protocol.md']


def test_unapproved_snapshot_cannot_claim_go():
    lock={'status':'NO-GO','snapshot_sha256':'x','human_approval':None}
    assert len(approval_errors(lock))==2
    lock['status']='GO'
    lock['human_approval']={'approved_by':'reviewer','approved_at':'2026-01-01','snapshot_sha256':'old'}
    assert approval_errors(lock)==['no human approval bound to this snapshot']


def test_declared_go_still_requires_real_splits_dates_and_evidence(tmp_path):
    folder=tmp_path/'review/design';folder.mkdir(parents=True)
    files={};lock={'files':files,'snapshot_sha256':snapshot_digest(files),'status':'GO',
                   'human_approval':{'approved_by':'fixture','approved_at':'2026-01-01',
                                     'snapshot_sha256':snapshot_digest(files)}}
    (folder/'design-lock.json').write_text(json.dumps(lock))
    spec={'status':'GO','seed_runs_authorized_now':True,'project_registry':'registry.json'}
    (folder/'evaluation-spec.json').write_text(json.dumps(spec))
    (folder/'registry.json').write_text(json.dumps({'repositories':[]}))
    errors=execution_blockers(tmp_path)
    assert 'repository assignments are incomplete' in errors
    assert 'fixed evaluation dates are missing' in errors
    assert 'unverified GO gate: collection_pilot' in errors


def test_manifest_cannot_reference_outside_repository(tmp_path):
    files={'../outside':'x'}
    import pytest
    with pytest.raises(ValueError,match='escapes'):
        integrity_errors(tmp_path,{'files':files,'snapshot_sha256':snapshot_digest(files)})
