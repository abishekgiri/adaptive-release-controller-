"""Freeze/check the new research design. Never runs policies or approves a study."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from data.research_contract import validate_splits

ROOT=Path(__file__).resolve().parents[1]
LOCK=Path('review/design/design-lock.json')
GO_CHECKS=('collection_pilot','source_lineage_integration','project_frame_and_splits',
           'precision_plan','runner_integration','fresh_environment','holdout_governance')
FILES=(
    'data/research_contract.py', 'tests/test_research_contract.py', 'tests/test_design_gate.py',
    'review/design_precision.py', 'review/design_gate.py',
    'review/data-collection-plan.md', 'review/decision-feedback-contract.md',
    'review/feature-specification.md', 'review/evaluation-protocol.md',
    'review/project-selection-protocol.md', 'review/normalized-dataset-schema.md',
    'review/leakage-checklist.md', 'review/reproducibility-checklist.md',
    'review/automated-invariants.md', 'review/unresolved-design-decisions.md',
    'review/design/dataset-schema.sql', 'review/design/evaluation-spec.json',
    'review/design/project-registry.template.json', 'review/design/precision-sensitivity.csv',
    'review/design/precision-assumptions.json', 'review/design/README.md',
    'review/research-design-review.md', 'review/evidence/research-design-data-audit.json',
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def local_path(root,relative):
    path=(root/relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError('manifest path escapes repository')
    return path


def snapshot_digest(files):
    return hashlib.sha256(json.dumps(files,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def freeze(root=ROOT):
    files={p:sha(local_path(root,p)) for p in FILES}
    lock={'version':'ci-design-v1','status':'NO-GO',
          'created_at':datetime.now(timezone.utc).isoformat(),
          'files':files,'snapshot_sha256':snapshot_digest(files),
          'human_approval':None,'go_evidence':{k:None for k in GO_CHECKS},
          'note':'Local integrity snapshot, not external preregistration or execution approval.'}
    (root/LOCK).write_text(json.dumps(lock,indent=2,sort_keys=True)+'\n')
    return lock


def integrity_errors(root,lock):
    errors=[]
    if snapshot_digest(lock['files']) != lock.get('snapshot_sha256'):
        errors.append('manifest digest mismatch')
    for p,expected in lock['files'].items():
        path=local_path(root,p)
        if not path.is_file() or sha(path)!=expected:
            errors.append('changed or missing: '+p)
    return errors


def approval_errors(lock):
    errors=[]
    if lock.get('status')!='GO':
        errors.append('design status is NO-GO')
    approval=lock.get('human_approval')
    if (not isinstance(approval,dict) or not approval.get('approved_by')
        or not approval.get('approved_at')
        or approval.get('snapshot_sha256')!=lock.get('snapshot_sha256')):
        errors.append('no human approval bound to this snapshot')
    return errors


def execution_blockers(root=ROOT):
    lock=json.loads((root/LOCK).read_text())
    errors=integrity_errors(root,lock)+approval_errors(lock)
    if set(lock.get('files',{})) != set(FILES):
        errors.append('frozen file set does not cover the required design assets')
    spec=json.loads((root/'review/design/evaluation-spec.json').read_text())
    if spec.get('status')!='GO' or not spec.get('seed_runs_authorized_now'):
        errors.append('specification does not authorize policy execution')
    if not spec.get('margins_human_ratified'):
        errors.append('meaningful margins are not ratified')
    if not spec.get('repository_assignments_complete'):
        errors.append('repository assignments are incomplete')
    if not spec.get('evaluation_utc_start') or not spec.get('evaluation_utc_end'):
        errors.append('fixed evaluation dates are missing')
    registry=json.loads(local_path(root,'review/design/'+spec['project_registry']).read_text())
    try:
        rows=registry['repositories']
        validate_splits(rows)
        if {r['split'] for r in rows} != {'development','validation','evaluation'}:
            raise ValueError('all three partitions required')
    except (ValueError,KeyError) as exc:
        errors.append('invalid/unassigned project registry: '+str(exc))
    for check in GO_CHECKS:
        proof=lock.get('go_evidence',{}).get(check)
        if not isinstance(proof,dict) or proof.get('passed') is not True:
            errors.append('unverified GO gate: '+check)
            continue
        if not proof.get('path') or not proof.get('sha256'):
            errors.append('missing evidence artifact: '+check)
            continue
        path=local_path(root,proof['path'])
        if not path.is_file() or sha(path)!=proof['sha256']:
            errors.append('evidence artifact mismatch: '+check)
    return errors


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze',action='store_true',help='write a NO-GO snapshot; never approve execution')
    args=parser.parse_args()
    if args.freeze:
        print('Frozen NO-GO design:',freeze()['snapshot_sha256'])
        return 0
    errors=execution_blockers()
    if errors:
        print('NO-GO: policy experiments must not run.\n'+'\n'.join('- '+e for e in errors))
        return 2
    print('GO preflight: all declared evidence/approval hashes match. This does not authenticate human identity or replace a scientific review.')
    return 0


if __name__=='__main__':
    raise SystemExit(main())
