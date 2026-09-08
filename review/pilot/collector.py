"""Read-only, development-allowlisted acquisition. Never imports a model.
Raw objects are append-only; receipt journals retain every request/delivery.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from datetime import datetime, timedelta, timezone
from urllib.parse import urlencode, urlparse, parse_qs

ROOT = Path(__file__).resolve().parents[2]
DEVELOPMENT = ('pallets/flask', 'psf/requests', 'expressjs/express')
API_VERSION = '2026-03-10'
PRE = {'queued', 'requested', 'waiting', 'pending'}


def now():
    return datetime.now(timezone.utc).isoformat(timespec='microseconds').replace('+00:00', 'Z')


def dt(value):
    value = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise ValueError('aware UTC required')
    return value


def digest(data):
    return hashlib.sha256(data).hexdigest()


def atomic(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name+'.tmp')
    with tmp.open('wb') as f:
        f.write(data); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def write_json(path, value):
    atomic(path, (json.dumps(value, indent=2, sort_keys=True)+'\n').encode())


def parse_http(output):
    output = output.replace(b'\r\n', b'\n')
    head, sep, body = output.partition(b'\n\n')
    if not sep or not head.startswith(b'HTTP/'):
        raise ValueError('HTTP status/headers missing')
    lines = head.decode().splitlines()
    headers = {}
    for line in lines[1:]:
        k, _, v = line.partition(':'); headers[k.lower()] = v.strip()
    return int(lines[0].split()[1]), headers, body


class SourceArchive:
    def __init__(self, folder):
        self.folder = Path(folder)
        self.raw = self.folder/'raw'; self.raw.mkdir(parents=True, exist_ok=True)
        self.journal = self.folder/'source-journal.jsonl'

    def record(self, endpoint, status, headers, body, requested_at, received_at=None):
        sha = digest(body); path = self.raw/(sha+'.json')
        if not path.exists(): atomic(path, body)
        elif path.read_bytes()!=body: raise ValueError('source hash conflict')
        receipt = received_at or now()
        if dt(receipt)<dt(requested_at): raise ValueError('collector clock moved backwards')
        row = {'endpoint':endpoint,'status':status,'sha256':sha,
               'requested_at':requested_at,'received_at':receipt,
               'api_version':API_VERSION,'method':'GET',
               'response_headers':{k:headers[k] for k in ('date','link','etag','x-github-request-id',
                   'x-ratelimit-remaining','x-ratelimit-reset','retry-after') if k in headers}}
        with self.journal.open('a') as f:
            f.write(json.dumps(row,sort_keys=True)+'\n');f.flush();os.fsync(f.fileno())
        return row

    def read(self, record):
        raw=(self.raw/(record['sha256']+'.json')).read_bytes()
        if digest(raw)!=record['sha256']: raise ValueError('raw source checksum mismatch')
        return json.loads(raw)


class Client:
    def __init__(self, archive, max_calls=700):
        self.archive=archive; self.max_calls=max_calls; self.calls=0

    def get(self, endpoint):
        if not any(endpoint.startswith('/repos/'+repo+'/') or endpoint=='/repos/'+repo for repo in DEVELOPMENT):
            raise ValueError('endpoint outside development allowlist')
        if any(x in endpoint for x in ('..','://','\\')): raise ValueError('unsafe endpoint')
        for retry in range(3):
            self.calls+=1
            if self.calls>self.max_calls: raise RuntimeError('pilot request budget exceeded')
            requested=now(); tick=time.monotonic()
            r=subprocess.run(['gh','api','--hostname','github.com','--method','GET','--include',
                '-H','Accept: application/vnd.github+json','-H','X-GitHub-Api-Version: '+API_VERSION,
                endpoint],capture_output=True,timeout=60)
            if not r.stdout.startswith(b'HTTP/'):
                raise RuntimeError('GitHub transport/authentication failed; no token recorded')
            status,headers,body=parse_http(r.stdout)
            record=self.archive.record(endpoint,status,headers,body,requested)
            elapsed=time.monotonic()-tick
            wall=(dt(record['received_at'])-dt(requested)).total_seconds()
            if abs(elapsed-wall)>2: raise ValueError('clock discontinuity during request')
            if status==200: return self.archive.read(record),record
            if status not in (429,500,502,503,504) and not (status==403 and headers.get('x-ratelimit-remaining')=='0'):
                raise RuntimeError(f'GitHub returned {status} for {endpoint}')
            wait=int(headers.get('retry-after','2'))
            if status==403: wait=max(1,int(headers.get('x-ratelimit-reset','0'))-int(time.time())+1)
            if wait>30: raise RuntimeError('rate limit requires later resume; sources retained')
            time.sleep(min(30,wait*(retry+1)))
        raise RuntimeError('bounded request retries exhausted')


def paginated(get, endpoint, key=None, cap=10000):
    page=1; result=[]; records=[]; complete=False
    while len(result)<cap:
        separator='&' if '?' in endpoint else '?'
        payload,record=get(endpoint+separator+urlencode({'per_page':100,'page':page}))
        records.append(record)
        batch=payload[key] if key else payload
        if not isinstance(batch,list): raise ValueError('list response required')
        result.extend(batch)
        link=record.get('response_headers',{}).get('link','')
        if 'rel="next"' not in link:
            complete=True;break
        page+=1
    return result[:cap],records,complete


def windowed_runs(get, repo, start, end):
    """Complete bounded historical window; split the provider's 1,000-run cap.
    Boundary overlap is intentional and deduplicated by run+attempt identity.
    """
    if dt(end)<dt(start): raise ValueError('reversed window')
    endpoint='/repos/'+repo+'/actions/runs?'+urlencode({'created':start+'..'+end})
    probe,_=get(endpoint+'&per_page=1&page=1')
    if probe['total_count']>1000:
        if (dt(end)-dt(start)).total_seconds()<=1:
            raise ValueError('more than 1000 runs in indivisible time window')
        mid=(dt(start)+(dt(end)-dt(start))/2).replace(microsecond=0).isoformat().replace('+00:00','Z')
        left=windowed_runs(get,repo,start,mid);right=windowed_runs(get,repo,mid,end)
        return {**left,**right}
    rows,_,complete=paginated(get,endpoint,'workflow_runs')
    if not complete: raise ValueError('incomplete run window')
    return {(int(r['id']),int(r['run_attempt'])):r for r in rows}


def checked_run(row, repo_id):
    for field in ('id','workflow_id','run_attempt'):
        if type(row.get(field)) is not int or row[field]<1: raise ValueError('ambiguous run identity')
    if row.get('repository',{}).get('id',repo_id)!=repo_id: raise ValueError('wrong repository')
    if not row.get('head_sha') or not row.get('head_branch'): raise ValueError('missing change/ref')
    created=dt(row['created_at']);started=row.get('run_started_at')
    if started and dt(started)<created: raise ValueError('run start predates creation')
    if row.get('status')=='completed' and not row.get('conclusion'):
        return 'missing_conclusion'
    return 'valid'


def workflow_selected(row):
    import re
    value=(row.get('name','')+' '+row.get('path','')).lower()
    if re.search(r'release|publish|deploy|documentation|label|codeql|security',value): return False
    return bool(re.search(r'\bci\b|test|build|coverage|integration',value))


def capture_round(client, archive, state, *, initial=False):
    boundary=now()
    if initial:
        state.update(version='ci-design-v1-pilot-1',parent_snapshot='71c8a3d61f377b028cc6d37116cf5900eb180b01e3d44aabbb051775d36146cb',
                     started_at=boundary,repositories={},rounds=[],samples=[],errors=[])
    round_info={'started_at':boundary,'observations':[]}
    for repo in DEVELOPMENT:
        prefix='/repos/'+repo
        if initial:
            metadata,meta_source=client.get(prefix)
            flows,flow_sources,flow_complete=paginated(client.get,prefix+'/actions/workflows','workflows')
            state['repositories'][repo]={'metadata':metadata,'metadata_source':meta_source,'workflows':flows,
                'workflow_sources':flow_sources,'workflow_list_complete':flow_complete}
            old_date=(dt(boundary)-timedelta(days=183)).date().isoformat()
            old,old_source=client.get(prefix+'/actions/runs?'+urlencode({'created':'<='+old_date,'per_page':1}))
            state['repositories'][repo]['six_month_history']={'count':old['total_count'],
                'source':old_source,'sample_created_at':old['workflow_runs'][0]['created_at'] if old['workflow_runs'] else None}
        config=state['repositories'][repo]
        endpoint=prefix+'/actions/runs?'+urlencode({'created':'<='+boundary})
        rows,sources,complete=paginated(client.get,endpoint,'workflow_runs',cap=300 if initial else 100)
        # Keep the exact page receipt for each run; do not credit page 3 at page 1's time.
        receipt_map={}
        for source in sources:
            for row in archive.read(source)['workflow_runs']:
                receipt_map.setdefault((row['id'],row['run_attempt']),source)
        seen=set()
        for row in rows:
            key=(row['id'],row['run_attempt'])
            if key in seen: continue
            seen.add(key)
            try: checked_run(row,config['metadata']['id'])
            except ValueError as exc:
                state['errors'].append({'repo':repo,'run_id':row.get('id'),'error':str(exc)});continue
            source=receipt_map[key]
            round_info['observations'].append({'repo':repo,'run':row,'source':source,'kind':'summary'})
            if initial and row['run_attempt']>1:
                for attempt in range(1,row['run_attempt']):
                    try:
                        prior,prior_source=client.get(prefix+f'/actions/runs/{row["id"]}/attempts/{attempt}')
                        if prior.get('run_attempt')!=attempt: raise ValueError('attempt endpoint returned wrong attempt')
                        checked_run(prior,config['metadata']['id'])
                        round_info['observations'].append({'repo':repo,'run':prior,'source':prior_source,'kind':'attempt'})
                    except (RuntimeError,ValueError) as exc:
                        state['errors'].append({'repo':repo,'run_id':row['id'],'attempt':attempt,'error':str(exc)})
        config.setdefault('round_coverage',[]).append({'upper_boundary':boundary,'rows':len(rows),
            'pages':len(sources),'complete_all_history':complete,'budget_truncated':not complete})
    round_info['ended_at']=now();state['rounds'].append(round_info)


def collect_changes(client, state):
    for repo,config in state['repositories'].items():
        selected={w['id'] for w in config['workflows'] if workflow_selected(w)}
        candidates={}
        for obs in state['rounds'][0]['observations']:
            r=obs['run']
            if obs['repo']==repo and r['workflow_id'] in selected and r['event'] in ('push','pull_request') and r['run_attempt']==1:
                candidates.setdefault(r['head_sha'],r)
        for sha in sorted(candidates,key=lambda x:digest(x.encode()))[:20]:
            row=candidates[sha];sample={'repo':repo,'run_id':row['id'],'run_attempt':1,'event':row['event'],
                                      'head_sha':sha,'collected_at':now(),'diagnostic_only':True}
            try:
                commit,cs=client.get('/repos/'+repo+'/commits/'+sha+'?per_page=100&page=1')
                sample.update(commit=commit,commit_sources=[cs])
                # Every extra page must resolve the exact same commit, including file totals.
                files=list(commit.get('files',[]));page=1;link=cs.get('response_headers',{}).get('link','')
                while 'rel="next"' in link:
                    page+=1
                    extra,es=client.get('/repos/'+repo+'/commits/'+sha+f'?per_page=100&page={page}')
                    if extra.get('sha')!=commit['sha']: raise ValueError('commit changed across pages')
                    files.extend(extra.get('files',[]));sample['commit_sources'].append(es)
                    link=es.get('response_headers',{}).get('link','')
                sample['files']=files
                sample['complete_files']=len(files)<3000
                # A historical API head is insufficient evidence of the tested PR merge ref.
                sample['frozen_diff_basis_verified']=row['event']=='push' and len(commit.get('parents',[]))==1
                sample['base_sha']=commit['parents'][0]['sha'] if sample['frozen_diff_basis_verified'] else None
                path=next((w['path'] for w in config['workflows'] if w['id']==row['workflow_id']),None)
                sample['workflow_path']=path
                if path and path.startswith('.github/workflows/'):
                    try:
                        content,ws=client.get('/repos/'+repo+'/contents/'+path+'?'+urlencode({'ref':sha}))
                        sample.update(workflow_content=content,workflow_source=ws,
                                      workflow_ref_verified=row['event']=='push')
                    except RuntimeError as exc: sample['workflow_error']=str(exc)
                # Jobs are only post-execution QA and never predecision features.
                jobs,js,jcomplete=paginated(client.get,'/repos/'+repo+f'/actions/runs/{row["id"]}/attempts/1/jobs','jobs')
                sample.update(jobs=jobs,job_sources=js,jobs_complete=jcomplete)
            except (RuntimeError,ValueError) as exc:
                sample['error']=str(exc)
            state['samples'].append(sample)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['collect','poll'])
    parser.add_argument('--output',type=Path,default=ROOT/'review/pilot')
    args=parser.parse_args()
    from review.design_gate import integrity_errors, LOCK
    lock=json.loads((ROOT/LOCK).read_text())
    if integrity_errors(ROOT,lock): raise ValueError('frozen protocol changed')
    folder=args.output;archive=SourceArchive(folder);client=Client(archive)
    state_path=folder/'acquisition-state.json'
    if args.mode=='collect':
        if state_path.exists(): raise ValueError('initial collection exists; use poll or new explicit directory')
        state={}
        try:
            capture_round(client,archive,state,initial=True)
            write_json(state_path,state)
            collect_changes(client,state)
        finally:
            write_json(state_path,state)
    else:
        state=json.loads(state_path.read_text())
        capture_round(client,archive,state)
        write_json(state_path,state)
    print(json.dumps({'mode':args.mode,'calls':client.calls,'rounds':len(state['rounds']),
                      'change_samples':len(state['samples']),'errors':len(state['errors'])}))


if __name__=='__main__': main()
