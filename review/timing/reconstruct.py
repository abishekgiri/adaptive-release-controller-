"""Read-only reconstruction of archived pilot timing, with unknowns preserved.
No network, decision policy, enforcement, or new repository observations.
"""
import csv
import hashlib
import json
from pathlib import Path
import statistics
from datetime import datetime

ROOT=Path(__file__).resolve().parents[2]
PILOT=ROOT/'review/pilot'
OUT=ROOT/'review/timing/evidence'


def timestamp(x):
    return datetime.fromisoformat(x.replace('Z','+00:00')) if x else None


def distribution(values):
    if not values:return {'n':0,'min':None,'median':None,'p95':None,'max':None}
    values=sorted(values)
    i=.95*(len(values)-1);lo=int(i);hi=min(lo+1,len(values)-1)
    return {'n':len(values),'min':min(values),'median':statistics.median(values),
        'p95':values[lo]+(values[hi]-values[lo])*(i-lo),'max':max(values)}


def main():
    state=json.loads((PILOT/'acquisition-state.json').read_text())
    attempts=json.loads((PILOT/'attempt-diagnostics.json').read_text())
    journal=[json.loads(x) for x in (PILOT/'source-journal.jsonl').read_text().splitlines()]
    rows=[]
    for sample in state['samples']:
        run=next(r for r in attempts if r['repo']==sample['repo'] and r['run_id']==sample['run_id'] and r['attempt']==1)
        jobs=[j for j in sample.get('jobs',[]) if j.get('started_at')]
        first=min(jobs,key=lambda j:(timestamp(j['started_at']),j['id'])) if jobs else None
        created=min((j['created_at'] for j in sample.get('jobs',[]) if j.get('created_at')),default=None)
        steps=[(j,x) for j in jobs for x in j.get('steps',[]) if x.get('started_at') and x.get('number',0)>1
               and not x.get('name','').startswith(('Post ','Complete job'))]
        first_step=min(steps,key=lambda jx:(timestamp(jx[1]['started_at']),jx[0]['id'],jx[1]['number'])) if steps else None
        sources=[run['snapshot_source'],*sample.get('commit_sources',[])]
        if sample.get('workflow_source'):sources.append(sample['workflow_source'])
        receipt=run['first_observed_at']
        sources_ready=max(s['received_at'] for s in sources)
        rows.append({'repository':sample['repo'],'run_id':run['run_id'],'attempt':1,'head_sha':run['head_sha'],
          'workflow_created_at':run['created_at'],'run_started_at':run['started_at'],
          'workflow_requested_webhook_received_at':None,'queued_observed_at':None,
          'first_run_api_receipt_at':receipt,'earliest_job_created_at':created,
          'earliest_job_started_at':first['started_at'] if first else None,'earliest_job_id':first['id'] if first else None,
          'first_reported_non_setup_step_at':first_step[1]['started_at'] if first_step else None,
          'first_reported_non_setup_step_name':first_step[1]['name'] if first_step else None,
          'first_user_controlled_executable_at_verified':None,
          'retrospective_source_bundle_available_at':sources_ready,'prospective_feature_snapshot_complete_at':None,
          'decision_ready_at':None,'enforcement_acknowledged_at':None,'usable_margin_s':None,
          'run_to_earliest_job_s':(timestamp(first['started_at'])-timestamp(run['started_at'])).total_seconds() if first else None,
          'run_to_earliest_job_created_s':(timestamp(created)-timestamp(run['created_at'])).total_seconds() if created else None,
          'optimistic_job_margin_from_api_receipt_s':(timestamp(first['started_at'])-timestamp(receipt)).total_seconds() if first else None,
          'job_margin_from_retrospective_bundle_s':(timestamp(first['started_at'])-timestamp(sources_ready)).total_seconds() if first else None,
          'exact_frozen_diff_basis_verified':sample.get('frozen_diff_basis_verified',False),
          'run_source_sha256':run['snapshot_source']['sha256'],
          'job_source_sha256':';'.join(s['sha256'] for s in sample.get('job_sources',[])),
          'evidence_class':'retrospective_reconstruction_not_prospective_timing'})
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'reconstructed-timelines.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    per_repo={}
    for repo in sorted(state['repositories']):
        rs=[r for r in rows if r['repository']==repo]
        per_repo[repo]={k:distribution([r[k] for r in rs if r[k] is not None]) for k in
            ('run_to_earliest_job_s','run_to_earliest_job_created_s','optimistic_job_margin_from_api_receipt_s','job_margin_from_retrospective_bundle_s')}
        per_repo[repo]['samples']=len(rs)
    spans=[(timestamp(r['received_at'])-timestamp(r['requested_at'])).total_seconds() for r in journal]
    scoped=[r for r in attempts if r['scope_eligible']]
    summary={'evidence_class':'archived_api_behavior_and_local_request_latency_only','samples':len(rows),
      'with_jobs':sum(r['earliest_job_started_at'] is not None for r in rows),
      'with_non_setup_step_proxy':sum(r['first_reported_non_setup_step_at'] is not None for r in rows),
      'scoped_first_attempts':len(scoped),'scoped_start_equals_creation':sum(r['started_at']==r['created_at'] for r in scoped),
      'job_created_field_present':sum('created_at' in j for s in state['samples'] for j in s.get('jobs',[])),
      'sampled_jobs':sum(len(s.get('jobs',[])) for s in state['samples']),
      'api_request_to_durable_receipt_s':distribution(spans),'by_repository':per_repo,
      'actual_webhook_latency':None,'actual_prospective_feature_assembly_latency':None,'actual_decision_latency':None,
      'actual_enforcement_latency':None,'usable_margin_distribution':distribution([]),
      'live_enforcement':'UNVERIFIED','prospective_micro_pilot':'NOT RUN: user explicitly prohibited live repository enforcement testing',
      'permissions_from_archived_metadata':{repo:cfg['metadata']['permissions'] for repo,cfg in state['repositories'].items()},
      'input_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
         (PILOT/'acquisition-state.json',PILOT/'attempt-diagnostics.json',PILOT/'source-journal.jsonl')}}
    (OUT/'timing-summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
