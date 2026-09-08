"""Deterministic, data-only normalization, provenance checks and diagnostics."""
from __future__ import annotations
import base64
from collections import Counter
from datetime import timedelta
import fnmatch
import hashlib
import json
from pathlib import Path, PurePosixPath
import sqlite3
import statistics
import yaml

from data.research_contract import FEATURE_GROUPS, CONCLUSIONS, Decision, AttemptKey, FeatureValue, validate_features
from review.pilot.collector import ROOT, SourceArchive, dt, digest, write_json, workflow_selected, checked_run, PRE

DEPENDENCIES={'package.json','package-lock.json','yarn.lock','pnpm-lock.yaml','pyproject.toml','poetry.lock',
 'Pipfile','Pipfile.lock','go.mod','go.sum','pom.xml','build.gradle','build.gradle.kts','Cargo.toml','Cargo.lock'}
RISKY=('.github/','deploy/','deployment/','infra/','infrastructure/','k8s/','migrations/','terraform/')
DIFF_NAMES=tuple(k for k,v in FEATURE_GROUPS.items() if v=='change')


def diff_features(files, complete):
    """Complete paths required. Missing patches cannot establish text line totals."""
    result={k:None for k in DIFF_NAMES}
    if not complete: return result
    paths=[]; names=[]; text_known=True
    for item in files:
        name=item.get('filename')
        if not isinstance(name,str) or not name or name in names: raise ValueError('ambiguous/repeated diff file')
        names.append(name);paths.append(name)
        if item.get('status')=='renamed':
            if not item.get('previous_filename'): raise ValueError('rename missing old path')
            paths.append(item['previous_filename'])
        if not isinstance(item.get('patch'),str): text_known=False
        for k in ('additions','deletions'):
            if type(item.get(k)) is not int or item[k]<0: text_known=False
    exts=Counter()
    for name in names:
        base=PurePosixPath(name).name
        ext=PurePosixPath(base).suffix.lower() or ('<dotfile>' if base.startswith('.') else '<none>')
        exts[ext]+=1
    result.update(files_changed=len(files),extension_counts=dict(sorted(exts.items())),
        top_directory_count=len({p.split('/')[0] if '/' in p else '<root>' for p in paths}),
        dependency_file_changed=any(PurePosixPath(p).name in DEPENDENCIES or fnmatch.fnmatchcase(PurePosixPath(p).name,'requirements*.txt') for p in paths),
        test_file_changed=any(any(c in ('test','tests','__tests__') for c in p.split('/')) or
          any(fnmatch.fnmatchcase(PurePosixPath(p).name,pat) for pat in ('test_*.py','*_test.py','*.test.*','*.spec.*')) for p in paths),
        configuration_changed=any(p.startswith('.github/') or PurePosixPath(p).name in ('.gitlab-ci.yml','Makefile','tox.ini','pytest.ini') or
          any(fnmatch.fnmatchcase(PurePosixPath(p).name,pat) for pat in ('Dockerfile*','docker-compose*.yml')) or
          p.endswith(('.yaml','.yml','.toml','.ini')) for p in paths),
        risky_path_changed=any(p.startswith(RISKY) for p in paths))
    if text_known:
        a=sum(f['additions'] for f in files);d=sum(f['deletions'] for f in files)
        result.update(additions=a,deletions=d,total_churn=a+d)
    return result


def workflow_features(content, ref_verified):
    result={'declared_job_count':None,'declared_runner_types':None}
    if not ref_verified or not content or content.get('encoding')!='base64': return result
    raw=base64.b64decode(content['content'],validate=False)
    if len(raw)>262144: raise ValueError('workflow exceeds safe parser budget')
    config=yaml.safe_load(raw)
    if not isinstance(config,dict) or not isinstance(config.get('jobs'),dict): return result
    jobs=config['jobs']; result['declared_job_count']=len(jobs)
    runners=[]
    for job in jobs.values():
        if not isinstance(job,dict) or 'uses' in job: return result
        value=job.get('runs-on')
        labels=[value] if isinstance(value,str) else value
        if not isinstance(labels,list) or any(not isinstance(v,str) or '${{' in v for v in labels): return result
        runners.extend(labels)
    result['declared_runner_types']=sorted(runners)
    return result


def source_closure(archive, references, decision_at, repository_id, *, history=False):
    """References explicitly enumerate every primitive source; no claimed times accepted."""
    if not references: raise ValueError('lineage is empty')
    times=[]
    journal=[json.loads(line) for line in archive.journal.read_text().splitlines()]
    receipts={(r['sha256'],r['endpoint'],r['received_at']) for r in journal if r['status']==200}
    for ref in references:
        if (ref['sha256'],ref['endpoint'],ref['received_at']) not in receipts:
            raise ValueError('source receipt is not in the acquisition journal')
        payload=archive.read(ref)
        if ref.get('repository_id')!=repository_id: raise ValueError('lineage repository mismatch')
        if ref.get('purpose')=='diff' and (not isinstance(payload,dict) or 'files' not in payload or 'sha' not in payload):
            raise ValueError('outcome cannot masquerade as a diff source')
        receipt=dt(ref['received_at'])
        if receipt>dt(decision_at) or (history and receipt>=dt(decision_at)):
            raise ValueError('future source dependency')
        if ref.get('purpose') not in ('diff','metadata','workflow_definition','repository_snapshot','history'):
            raise ValueError('source role not allowed')
        # An aggregate can contain explicit child references; walk all of them.
        if ref.get('purpose')=='history' and isinstance(payload,dict) and 'dependencies' in payload:
            source_closure(archive,payload['dependencies'],decision_at,repository_id,history=True)
        times.append(receipt)
    return max(times)


def canonical_observations(state):
    groups={};duplicates=0;quarantine=[]
    for rnd in state['rounds']:
        for obs in rnd['observations']:
            r=obs['run'];repo=obs['repo'];checked_run(r,state['repositories'][repo]['metadata']['id'])
            key=(repo,r['id'],r['run_attempt']);groups.setdefault(key,[]).append(obs)
    rows=[]
    for key,observations in sorted(groups.items()):
        observations=sorted(observations,key=lambda x:(x['source']['received_at'],x['source']['sha256']))
        identities={(x['run']['workflow_id'],x['run']['head_sha'],x['run']['head_branch'],x['run']['event']) for x in observations}
        if len(identities)!=1:
            quarantine.append({'key':key,'reason':'conflicting immutable attempt identity'});continue
        fingerprints=[json.dumps(x['run'],sort_keys=True,separators=(',',':')) for x in observations]
        duplicates+=len(fingerprints)-len(set(fingerprints))
        terminals=[x for x in observations if x['run'].get('status')=='completed' and x['run'].get('conclusion')]
        if len({x['run']['conclusion'] for x in terminals})>1:
            quarantine.append({'key':key,'reason':'conflicting terminal outcomes'});continue
        first=observations[0];latest=observations[-1];r=latest['run'];terminal=terminals[0] if terminals else None
        starts=sorted({x['run']['run_started_at'] for x in observations if x['run'].get('run_started_at')})
        if len(starts)>1:
            quarantine.append({'key':key,'reason':'inconsistent attempt start'});continue
        started=starts[0] if starts else None
        if dt(first['source']['received_at'])<dt(r['created_at']):
            quarantine.append({'key':key,'reason':'source received before provider creation'});continue
        if terminal and started and dt(terminal['source']['received_at'])<dt(started):
            quarantine.append({'key':key,'reason':'terminal receipt before start'});continue
        pre=first['run'].get('status') in PRE
        captured=pre and (started is None or dt(first['source']['received_at'])<dt(started))
        decision_at=first['source']['received_at'] if captured else None
        conclusion=terminal['run']['conclusion'] if terminal else None
        normalized=conclusion if conclusion in CONCLUSIONS else 'unknown' if conclusion else None
        repo=key[0];config=state['repositories'][repo]
        workflow=next((w for w in config['workflows'] if w['id']==r['workflow_id']),None)
        eligible_scope=bool(workflow and workflow_selected(workflow) and r['event'] in ('push','pull_request') and r['run_attempt']==1)
        # Public run head alone does not authenticate the PR tested merge revision.
        tested_verified=r['event']=='push'
        row={'repo':repo,'repository_id':config['metadata']['id'],'run_id':r['id'],'attempt':r['run_attempt'],
            'workflow_id':r['workflow_id'],'workflow_name':r['name'],'head_sha':r['head_sha'],
            'tested_sha':r['head_sha'] if tested_verified else None,'tested_sha_verified':tested_verified,
            'branch':r['head_branch'],'event':r['event'],'created_at':r['created_at'],'started_at':started,
            'decision_at':decision_at,'snapshot_source':first['source'],'latest_source':latest['source'],
            'outcome_source':terminal['source'] if terminal else None,'raw_conclusion':conclusion,'conclusion':normalized,
            'label':{'success':0,'failure':1,'timed_out':1}.get(normalized),
            'label_available_at':terminal['source']['received_at'] if terminal else None,
            'completed_at':None,'completed_at_reason':'run endpoint has no authenticated attempt completion timestamp',
            'scope_eligible':eligible_scope,'primary_eligible':bool(captured and started and tested_verified and eligible_scope),
            'pr_numbers':sorted({p['number'] for p in r.get('pull_requests',[]) if p.get('number')}),
            'observation_count':len(observations),'first_observed_at':first['source']['received_at']}
        rows.append(row)
    return rows,duplicates,quarantine


def history_features(current, rows, coverage_start):
    cutoff=dt(current['decision_at']);visible=[]
    for row in rows:
        if row['repository_id']!=current['repository_id'] or not row['primary_eligible'] or row['label'] is None: continue
        if (row['run_id'],row['attempt'])==(current['run_id'],current['attempt']): continue
        available=dt(row['label_available_at'])
        if available>=cutoff or available>dt(row['decision_at'])+timedelta(days=30): continue
        visible.append(row)
    seven=[r for r in visible if dt(r['label_available_at'])>=cutoff-timedelta(days=7)]
    work=[r for r in seven if r['workflow_id']==current['workflow_id']]
    same=[r for r in visible if r['tested_sha']==current['tested_sha']]
    rate=lambda xs:sum(r['label'] for r in xs)/len(xs) if xs else None
    known=[r for r in visible if r['workflow_id']==current['workflow_id'] and r['completed_at'] and r['started_at']]
    last=max(known,key=lambda r:(r['label_available_at'],r['run_id'],r['attempt'])) if known else None
    return {'project_failure_rate_7d':rate(seven),'project_resolved_count_7d':len(seven),
        'workflow_failure_rate_7d':rate(work),'workflow_resolved_count_7d':len(work),
        'same_change_prior_failures':sum(r['label'] for r in same),'same_change_prior_resolved':len(same),
        'prior_workflow_duration_s':(dt(last['completed_at'])-dt(last['started_at'])).total_seconds() if last else None,
        'author_failure_rate_90d':None,'author_resolved_count_90d':None,
        'repository_commits_30d':None,'history_age_s':max(0,(cutoff-dt(coverage_start)).total_seconds()),
        'history_left_truncated':True},visible


def shift_outcome_availability(rows, seconds):
    shifted=json.loads(json.dumps(rows))
    for r in shifted:
        if r.get('label_available_at'):
            r['label_available_at']=(dt(r['label_available_at'])+timedelta(seconds=seconds)).isoformat().replace('+00:00','Z')
    return shifted


def build_snapshot(current, rows, state, archive):
    """Derive all 30 frozen candidates from sealed, legitimately available inputs.
    Unavailable inputs remain null. Historical diagnostic enrichments never enter
    an earlier snapshot; only cached exact push diffs can be reused later.
    """
    cfg=state['repositories'][current['repo']];rid=current['repository_id'];cutoff=current['decision_at']
    def ref(source,purpose):return dict(source,repository_id=rid,purpose=purpose)
    snap=ref(current['snapshot_source'],'metadata');meta=ref(cfg['metadata_source'],'repository_snapshot')
    values={k:None for k in FEATURE_GROUPS};lineage={k:[snap] for k in values}
    missing={k:'not_captured' for k in values}
    values.update(workflow_identity=f"{rid}:{current['workflow_id']}",event_type=current['event'],attempt_number=current['attempt'])
    if dt(meta['received_at'])<=dt(cutoff):
        values.update(repository_language=cfg['metadata'].get('language'),
            repository_age_days=(dt(cutoff)-dt(cfg['metadata']['created_at'])).days,
            branch_class='pr' if current['event']=='pull_request' else 'default' if current['branch']==cfg['metadata']['default_branch'] else 'nondefault')
        for name in ('repository_language','repository_age_days','branch_class'):lineage[name]=[meta,snap]
    available_samples=[]
    for sample in state['samples']:
        if sample['repo']!=current['repo'] or sample['head_sha']!=current['tested_sha'] or not sample.get('frozen_diff_basis_verified'):continue
        sources=sample.get('commit_sources',[])
        if sources and all(dt(x['received_at'])<=dt(cutoff) for x in sources): available_samples.append(sample)
    if available_samples:
        sample=min(available_samples,key=lambda x:max(r['received_at'] for r in x['commit_sources']))
        vals=diff_features(sample.get('files',[]),sample.get('complete_files',False));values.update(vals)
        for name in vals:
            lineage[name]=[ref(x,'diff') for x in sample['commit_sources']]
            missing[name]='incomplete_diff'
        ws=sample.get('workflow_source')
        if ws and sample.get('workflow_ref_verified') and dt(ws['received_at'])<=dt(cutoff):
            wf=workflow_features(sample.get('workflow_content'),True);values.update(wf)
            for name in wf:lineage[name]=[ref(ws,'workflow_definition')];missing[name]='dynamic_definition'
    history,visible=history_features(current,rows,state['started_at'])
    # Coverage is left truncated and intermittent for polling; do not invent a verified age.
    history['history_age_s']=None
    initial=ref(cfg['metadata_source'],'history')
    historical_sources=[ref(r['outcome_source'],'history') for r in visible] or [initial]
    if all(dt(r['received_at'])<dt(cutoff) for r in historical_sources):
        values.update(history)
        for name in history:lineage[name]=historical_sources;missing[name]='no_history'
    for name in ('author_failure_rate_90d','author_resolved_count_90d'):missing[name]='unknown_identity'
    packet={};typed=[]
    d=Decision(AttemptKey(rid,current['run_id'],current['attempt']),current['workflow_id'],current['tested_sha'],
        current['branch'],current['event'],dt(current['created_at']),dt(cutoff),dt(current['started_at']),current['snapshot_source']['sha256'])
    for name,value in values.items():
        sources=lineage[name];group=FEATURE_GROUPS[name]
        # Omission evidence may be the snapshot itself; it is not a measured diff/history.
        if value is not None:
            source_closure(archive,sources,cutoff,rid,history=group=='history')
        available=max(x['received_at'] for x in sources)
        if group=='history' and dt(available)>=dt(cutoff):
            # No earlier initialization ledger means the entire snapshot cannot be certified.
            raise ValueError('history initialization not available before decision')
        kind={'change':'diff','workflow':'metadata','repository':'repository_snapshot','history':'history'}[group]
        if name in ('declared_job_count','declared_runner_types'):kind='workflow_definition'
        reason=missing[name] if value is None else None
        typed.append(FeatureValue(name,value,dt(available),kind,sources[0]['sha256'],missing_reason=reason))
        packet[name]={'value':value,'missing_reason':reason,'available_at':available,'sources':sources}
    validate_features(d,tuple(typed))
    return packet


def diagnostic_features(sample, config):
    # Values are explicitly retrospective diagnostics, never model inputs.
    result={k:None for k in FEATURE_GROUPS}
    if sample.get('commit'):
        result.update(diff_features(sample.get('files',[]),sample.get('complete_files',False)))
    result.update(workflow_features(sample.get('workflow_content'),sample.get('workflow_ref_verified',False)))
    result.update(repository_language=config['metadata'].get('language'))
    return result


def predictor_export(connection):
    """The actual query contains no outcome or observed-job join."""
    return connection.execute('''SELECT d.repository_id,d.run_id,d.attempt,d.decision_at,
        f.feature_name,f.value_json,f.missing_reason,f.available_at
        FROM decisions d JOIN feature_values f USING(repository_id,run_id,attempt)
        WHERE d.primary_eligible=1 ORDER BY d.repository_id,d.run_id,d.attempt,f.feature_name''').fetchall()


def validate_acquisition_state(state, archive):
    """Reject edited convenience views; transformations must match raw sources."""
    for repo,cfg in state['repositories'].items():
        if cfg['metadata']!=archive.read(cfg['metadata_source']):raise ValueError('repository view differs from raw source')
        raw_flows=[w for src in cfg['workflow_sources'] for w in archive.read(src)['workflows']]
        if cfg['workflows']!=raw_flows:raise ValueError('workflow view differs from raw source')
    observations={}
    for rnd in state['rounds']:
        for obs in rnd['observations']:
            raw=archive.read(obs['source']);run=obs['run']
            candidates=raw.get('workflow_runs',[raw])
            if run not in candidates:raise ValueError('run view differs from raw source')
            observations[(obs['repo'],run['id'],run['run_attempt'])]=run
    for sample in state['samples']:
        run=observations[(sample['repo'],sample['run_id'],sample['run_attempt'])]
        if sample['head_sha']!=run['head_sha'] or sample['event']!=run['event']:raise ValueError('sample identity differs from run')
        if 'commit' not in sample:continue
        raws=[archive.read(src) for src in sample['commit_sources']]
        if sample['commit']!=raws[0] or sample.get('files')!=[f for raw in raws for f in raw.get('files',[])]:raise ValueError('diff view differs from raw source')
        if sample['frozen_diff_basis_verified']!=(sample['event']=='push' and len(sample['commit'].get('parents',[]))==1):raise ValueError('unverified diff basis promoted')
        if sample.get('workflow_content') and sample['workflow_content']!=archive.read(sample['workflow_source']):raise ValueError('workflow content differs from raw source')
        if sample.get('jobs')!=[j for src in sample.get('job_sources',[]) for j in archive.read(src)['jobs']]:raise ValueError('jobs differ from raw source')


def normalize(folder, output=None):
    folder=Path(folder);output=Path(output) if output else folder;output.mkdir(parents=True,exist_ok=True)
    archive=SourceArchive(folder);state=json.loads((folder/'acquisition-state.json').read_text())
    journal=[json.loads(x) for x in archive.journal.read_text().splitlines()]
    for ref in journal: archive.read(ref)
    validate_acquisition_state(state,archive)
    rows,duplicates,quarantine=canonical_observations(state)
    con=sqlite3.connect(':memory:');con.executescript((ROOT/'review/design/dataset-schema.sql').read_text())
    for repo,cfg in sorted(state['repositories'].items()):
        con.execute('INSERT INTO repositories VALUES (?,?,?,?)',(cfg['metadata']['id'],repo,repo.split('/')[0],'development'))
    known_sources=set()
    for ref in sorted(journal,key=lambda x:(x['received_at'],x['sha256'])):
        if ref['sha256'] in known_sources: continue
        repo=next((r for r in state['repositories'] if ref['endpoint']=='/repos/'+r or ref['endpoint'].startswith('/repos/'+r+'/')),None)
        if not repo: raise ValueError('unassigned source repository')
        known_sources.add(ref['sha256'])
        con.execute('INSERT INTO source_objects VALUES (?,?,?,?,?,?,?,?,?)',
            (ref['sha256'],state['repositories'][repo]['metadata']['id'],'raw_api',ref['received_at'],None,
             ref['api_version'],ref['response_headers'].get('x-github-request-id'),'raw/'+ref['sha256']+'.json',1 if ref['status']==200 else 0))
    for repo,cfg in sorted(state['repositories'].items()):
        for w in cfg['workflows']:
            con.execute('INSERT INTO workflows VALUES (?,?,?)',(cfg['metadata']['id'],w['id'],w['path']))
    for name,group in FEATURE_GROUPS.items():
        con.execute('INSERT INTO feature_definitions VALUES (?,?,?,?,?)',(name,group,'ci-design-v1','json',1))
    matrices=json.loads((ROOT/'review/design/evaluation-spec.json').read_text())['cost_matrices']
    for name,costs in sorted(matrices.items()):con.execute('INSERT INTO cost_matrices VALUES (?,?,?,?,?,?,?)',(name,*[v for r in costs for v in r]))
    normalized=[];omitted=[];strict_features=[]
    for r in rows:
        rid=r['repository_id'];key=(rid,r['run_id'],r['attempt']);sha=r['latest_source']['sha256']
        cfg=state['repositories'][r['repo']]
        if not r['tested_sha_verified'] or not any(w['id']==r['workflow_id'] for w in cfg['workflows']):
            omitted.append({'key':key,'reason':'tested revision unverified or workflow identity unavailable'});continue
        existing=con.execute('SELECT 1 FROM changes WHERE repository_id=? AND head_sha=? AND tested_sha=?',(rid,r['head_sha'],r['tested_sha'])).fetchone()
        if not existing:con.execute('INSERT INTO changes VALUES (?,?,?,?,?,?,?,?)',
            (rid,r['head_sha'],None,r['tested_sha'],'unresolved',None,None,sha))
        con.execute('INSERT INTO attempts VALUES (?,?,?,?,?,?,?,?,?,?,?)',
            (*key,r['workflow_id'],r['head_sha'],r['tested_sha'],r['branch'],r['event'],r['created_at'],r['started_at'],sha))
        if r['conclusion']:
            con.execute('INSERT INTO feedback VALUES (?,?,?,?,?,?,?,?,?)',
                (*key,r['raw_conclusion'],r['conclusion'],r['label'],r['label_available_at'],None,r['outcome_source']['sha256']))
        normalized.append(r)
        if not r['decision_at']:continue
        con.execute('INSERT INTO decisions VALUES (?,?,?,?,?,?,?,?)',
            (*key,r['decision_at'],'ci-design-v1',int(r['primary_eligible']),None if r['primary_eligible'] else 'unverified_start_or_scope',r['snapshot_source']['sha256']))
        for matrix in matrices: con.execute('INSERT INTO decision_scenarios VALUES (?,?,?,?)',(*key,matrix))
        if r['primary_eligible']:
            packet=build_snapshot(r,rows,state,archive)
            for name,entry in sorted(packet.items()):
                value=entry['value'];reason=entry['missing_reason']
                con.execute('INSERT INTO feature_values VALUES (?,?,?,?,?,?,?)',
                    (*key,name,json.dumps(value,sort_keys=True) if value is not None else None,reason,entry['available_at']))
                for ref in entry['sources']:
                    con.execute('INSERT OR IGNORE INTO feature_lineage VALUES (?,?,?,?,?)',(*key,name,ref['sha256']))
            strict_features.append({'key':key,'features':packet})
    # Populate diagnostic jobs only for attempts that have valid normalized identity.
    normalized_keys={(r['repo'],r['run_id'],r['attempt']) for r in normalized}
    for sample in state['samples']:
        key=(sample['repo'],sample['run_id'],1)
        if key not in normalized_keys: continue
        rid=state['repositories'][sample['repo']]['metadata']['id']
        for job in sample.get('jobs',[]):
            src=next(s for s in sample['job_sources'] if any(j['id']==job['id'] for j in archive.read(s)['jobs']))
            con.execute('INSERT INTO observed_jobs VALUES (?,?,?,?,?,?,?,?)',
                (rid,sample['run_id'],1,job['id'],job.get('started_at'),job.get('completed_at'),job.get('conclusion'),src['sha256']))
    integrity=con.execute('PRAGMA integrity_check').fetchall();foreign=con.execute('PRAGMA foreign_key_check').fetchall()
    if integrity!=[('ok',)] or foreign:raise ValueError('schema validation failed')
    target=output/'pilot.sqlite';temp=output/'pilot.sqlite.tmp'
    if temp.exists():temp.unlink()
    con.commit();disk=sqlite3.connect(temp);con.backup(disk);disk.close();temp.replace(target)
    logical='\n'.join(con.iterdump())+'\n';(output/'dataset.sql').write_text(logical)
    export=predictor_export(con);write_json(output/'predictor-export.json',export)
    write_json(output/'attempt-diagnostics.json',rows)
    diagnostics=[]
    for sample in state['samples']:
        try:
            values=diagnostic_features(sample,state['repositories'][sample['repo']])
            diagnostics.append({'repo':sample['repo'],'head_sha':sample['head_sha'],'run_id':sample['run_id'],
                'frozen_diff_basis_verified':sample.get('frozen_diff_basis_verified',False),
                'values':values,'available_before_original_decision':False,'purpose':'retrospective_extraction_diagnostic'})
        except (ValueError,yaml.YAMLError) as exc:
            quarantine.append({'sample':sample['head_sha'],'reason':str(exc)})
    write_json(output/'feature-diagnostics.json',diagnostics)
    write_json(output/'strict-feature-snapshots.json',strict_features)
    report={'schema_version':'ci-design-v1','raw_receipts':len(journal),'unique_raw_objects':len(known_sources),
        'raw_checksums_valid':True,'sqlite_integrity':integrity,'foreign_key_violations':foreign,
        'canonical_attempts':len(rows),'normalized_attempts':len(normalized),'omitted_from_normalized_schema':omitted,
        'exact_run_record_reobservations':duplicates,'conflicts_quarantined':quarantine,
        'preexecution_snapshots':sum(r['decision_at'] is not None for r in rows),
        'primary_eligible_decisions':sum(r['primary_eligible'] for r in rows),'predictor_export_rows':len(export),
        'logical_dataset_sha256':digest(logical.encode()),'sqlite_sha256':digest(target.read_bytes()),
        'limits':['Historical API rows never become decision snapshots','PR tested revision unverified: omitted, not fabricated',
                  'Predecision adapters require authentic receipt lineage; prospective field coverage remains unmeasured',
                  'Run completion timestamp is null; job maximum and updated_at never substituted']}
    write_json(output/'schema-validation.json',report)
    con.close()
    return report


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,default=ROOT/'review/pilot');p.add_argument('--output',type=Path)
    a=p.parse_args();print(json.dumps(normalize(a.input,a.output),indent=2))
