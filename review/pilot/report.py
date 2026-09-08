"""Descriptive pilot audit only: no fitting, policy costs, rankings or tuning."""
import csv
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import statistics
import subprocess
from collections import Counter,defaultdict

from data.research_contract import FEATURE_GROUPS
from review.pilot.collector import ROOT,dt,digest,write_json,workflow_selected,API_VERSION
from review.design_precision import required_clusters

FOLDER=ROOT/'review/pilot'


def stats(values):
    if not values:return {'n':0,'unique':0,'distribution':None,'variance':None}
    counts=Counter(json.dumps(x,sort_keys=True) for x in values)
    result={'n':len(values),'unique':len(counts),'mode_fraction':max(counts.values())/len(values),
            'distribution':dict(counts),'variance':None}
    if all(type(v) in (int,float) for v in values):
        x=sorted(values)
        def q(p):
            i=(len(x)-1)*p;lo=math.floor(i);hi=math.ceil(i)
            return x[lo]+(x[hi]-x[lo])*(i-lo)
        result.update(min=min(x),p01=q(.01),p25=q(.25),median=q(.5),p75=q(.75),p99=q(.99),max=max(x),
                      variance=statistics.pvariance(x))
    return result


def cluster_interval(rows):
    """Exploratory within-repository SHA-cluster bootstrap; not policy seeds."""
    groups=defaultdict(list)
    for r in rows:
        if r['label'] is not None:groups[r['head_sha']].append(r['label'])
    clusters=list(groups.values())
    if len(clusters)<2:return None
    rng=random.Random(20260907);means=[]
    for _ in range(2000):
        sample=[clusters[rng.randrange(len(clusters))] for _ in clusters]
        means.append(sum(map(sum,sample))/sum(map(len,sample)))
    means.sort();return [means[49],means[1949]]


def generate():
    state=json.loads((FOLDER/'acquisition-state.json').read_text())
    rows=json.loads((FOLDER/'attempt-diagnostics.json').read_text())
    features=json.loads((FOLDER/'feature-diagnostics.json').read_text())
    schema=json.loads((FOLDER/'schema-validation.json').read_text())
    journal=[json.loads(x) for x in (FOLDER/'source-journal.jsonl').read_text().splitlines()]
    report={};feature_rows=[]
    for repo,cfg in state['repositories'].items():
        allrows=[r for r in rows if r['repo']==repo];scope=[r for r in allrows if r['scope_eligible']]
        binary=[r for r in scope if r['label'] is not None]
        statuses=Counter(r['conclusion'] or 'missing' for r in allrows)
        scoped_status=Counter(r['conclusion'] or 'missing' for r in scope)
        wf=Counter(str(r['workflow_id']) for r in allrows);swf=Counter(str(r['workflow_id']) for r in scope)
        shas=Counter(r['head_sha'] for r in scope)
        new=[r for r in allrows if dt(r['created_at'])>=dt(state['started_at'])]
        samples=[s for s in state['samples'] if s['repo']==repo]
        job_start_lags=[]
        for sample in samples:
            rr=next((x for x in allrows if x['run_id']==sample['run_id'] and x['attempt']==1),None)
            starts=[dt(j['started_at']) for j in sample.get('jobs',[]) if j.get('started_at')]
            if rr and rr['started_at'] and starts:job_start_lags.append((min(starts)-dt(rr['started_at'])).total_seconds())
        prevalence=sum(r['label'] for r in binary)/len(binary) if binary else None
        m=cfg['metadata']
        report[repo]={'repository_id':m['id'],'language':m['language'],'git_size_kib_proxy':m['size'],
            'created_at':m['created_at'],'fork':m['fork'],'archived':m['archived'],
            'six_month_history_evidence':cfg['six_month_history'],'workflows_listed':len(cfg['workflows']),
            'selected_workflows':[{'id':w['id'],'name':w['name'],'path':w['path']} for w in cfg['workflows'] if workflow_selected(w)],
            'observed_workflows':len(wf),'observations_per_workflow':dict(wf),'scoped_observations_per_workflow':dict(swf),
            'distinct_runs':len({r['run_id'] for r in allrows}),'attempts':len(allrows),
            'first_attempts':sum(r['attempt']==1 for r in allrows),'rerun_attempts':sum(r['attempt']>1 for r in allrows),
            'unique_head_shas':len({r['head_sha'] for r in allrows}),
            'observed_pr_numbers':len({p for r in allrows for p in r['pr_numbers']}),
            'pr_attempts_without_pr_number':sum(r['event']=='pull_request' and not r['pr_numbers'] for r in allrows),
            'unique_changes_exact':None,'change_identity_limitation':'head SHA is a commit proxy; PR and tested merge identities are incomplete',
            'all_status_counts':dict(statuses),'scope_status_counts':dict(scoped_status),'scope_first_attempts':len(scope),
            'scoped_unique_head_shas':len(shas),'largest_sha_cluster':max(shas.values(),default=0),
            'sha_cluster_size_distribution':dict(Counter(shas.values())),
            'resolved_binary':len(binary),'scoped_failure_rate':prevalence,
            'scoped_run_start_equals_creation':sum(r['started_at'] is not None and dt(r['started_at'])==dt(r['created_at']) for r in scope),
            'first_job_minus_reported_run_start_seconds':stats(job_start_lags),
            'descriptive_binary_variance':prevalence*(1-prevalence) if prevalence is not None else None,
            'exploratory_sha_bootstrap_95':cluster_interval(binary),
            'earliest_created_at':min(r['created_at'] for r in allrows),'latest_created_at':max(r['created_at'] for r in allrows),
            'source_time_span_days':(max(dt(r['created_at']) for r in allrows)-min(dt(r['created_at']) for r in allrows)).total_seconds()/86400,
            'new_runs_observed_during_session':len(new),'verified_preexecution':sum(r['decision_at'] is not None for r in allrows),
            'primary_eligible':sum(r['primary_eligible'] for r in allrows),
            'sampled_changes':len(samples),'complete_file_inventories':sum(s.get('complete_files',False) for s in samples),
            'verified_push_diff_basis_samples':sum(s.get('frozen_diff_basis_verified',False) for s in samples),
            'sample_job_statuses':dict(Counter(j.get('conclusion') or 'missing' for s in samples for j in s.get('jobs',[])))}
    # Metadata availability is shown for diagnostic extraction; not retrospectively assigned to a decision.
    row_lookup={(r['repo'],r['run_id']):r for r in rows if r['attempt']==1}
    for f in features:
        r=row_lookup[(f['repo'],f['run_id'])]
        f['values'].update(workflow_identity=f"{r['repository_id']}:{r['workflow_id']}",event_type=r['event'],attempt_number=1)
    for cohort in ['ALL',*sorted(report)]:
        selected=[f for f in features if cohort=='ALL' or f['repo']==cohort]
        for name,group in FEATURE_GROUPS.items():
            vals=[f['values'][name] for f in selected if f['values'][name] is not None];profile=stats(vals)
            if not vals:
                classification='NOT COLLECTABLE';reason='No valid source/availability under this pilot route; not a claim of impossibility with prospective capture.'
            elif profile['unique']==1 and name=='attempt_number':
                classification='UNUSABLE';reason='Constant 1 by primary cohort definition; retain as identity, exclude from primary predictors.'
            elif profile['unique']==1 and group=='change':
                classification='UNUSABLE';reason='Constant among sampled diagnostic changes; no demonstrated contextual variation here.'
            else:
                classification='QUESTIONABLE';reason='Retrospectively extractable; no genuine predecision observation or predictive value demonstrated.'
            if name=='prior_workflow_duration_s':reason='No authenticated workflow-attempt completion timestamp; job maximum/updated_at prohibited.'
            if name.startswith('author_'):reason='Stable authorship plus eligible observed history not established; actor is not author.'
            feature_rows.append({'cohort':cohort,'feature':name,'group':group,'classification':classification,
                'diagnostic_samples':len(selected),'diagnostic_nonnull':len(vals),
                'diagnostic_collection_success_rate':len(vals)/len(selected) if selected else None,
                'diagnostic_missingness_rate':1-len(vals)/len(selected) if selected else None,
                'unique_values':profile['unique'],'population_variance_descriptive':profile.get('variance'),
                'distribution_json':json.dumps(profile,sort_keys=True),
                'repository_coverage':';'.join(sorted({f['repo'] for f in selected if f['values'][name] is not None})),
                'strict_predecision_samples':0,'strict_missingness_rate':None,'strict_timestamp_validity_rate':None,
                'leakage_risk':'Historical receipt is postdecision; cannot be backdated. PR diff/tested-ref proof incomplete.' if group=='change' else 'Source must be available by decision; no predecision corpus exists.',
                'reason':reason,'predictive_value_tested':False})
    with (FOLDER/'feature-feasibility.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=feature_rows[0]);w.writeheader();w.writerows(feature_rows)
    write_json(FOLDER/'data-quality.json',report)
    first=min(dt(x['requested_at']) for x in journal);last=max(dt(x['received_at']) for x in journal)
    total_runs=sum(r['distinct_runs'] for r in report.values())
    total_scope=sum(r['scope_first_attempts'] for r in report.values())
    new_rows=[r for r in rows if dt(r['created_at'])>=dt(state['started_at'])]
    new_scope=[r for r in new_rows if r['scope_eligible']]
    gaps=[(dt(b['started_at'])-dt(a['ended_at'])).total_seconds()/3600 for a,b in zip(state['rounds'],state['rounds'][1:])]
    max_gap=max(gaps,default=0)
    if schema['preexecution_snapshots'] or schema['primary_eligible_decisions']:
        raise ValueError('New prospective snapshots require reviewing the pilot verdict and feature profiles before report generation')
    quality=['# Pilot data-quality report','',f'Collection: {first.isoformat()} through {last.isoformat()}. {len(state["rounds"])} polling rounds. Public development repositories only.','',
      'The latest-300-per-project frame is bounded and outcome-unfiltered. It is not a complete historical cohort, a fixed-calendar sample, or a representative prevalence estimate. Earlier attempts of observed reruns are expanded, which explains more attempts than runs.','',
      '| Development repository | Runs | Attempts | First / rerun | Head SHAs | Scoped first attempts | Success / failure / other | Scoped binary failure rate |',
      '|---|---:|---:|---:|---:|---:|---|---:|']
    for repo,r in report.items():
        c=r['scope_status_counts'];other=r['scope_first_attempts']-r['resolved_binary']
        quality.append(f'| {repo} | {r["distinct_runs"]} | {r["attempts"]} | {r["first_attempts"]} / {r["rerun_attempts"]} | {r["unique_head_shas"]} | {r["scope_first_attempts"]} | {c.get("success",0)} / {c.get("failure",0)+c.get("timed_out",0)} / {other} | {r["scoped_failure_rate"]:.3%} |')
    quality += ['',f'Total: **{total_runs} workflow runs, {len(rows)} valid attempts, three repositories**. There are {total_scope} retrospectively scope-eligible first attempts, but **zero primary-eligible decision snapshots**. Scope eligibility here means workflow/event/attempt rules only, not temporal eligibility.',
      '',f'Exact reobservations across poll/export records: {schema["exact_run_record_reobservations"]}. These are repeated observations, not new independent runs or evidence of duplicate execution. Conflicting canonical identities/outcomes: {len(schema["conflicts_quarantined"])}. Acquisition anomalies quarantined: {len(state["errors"])}.',
      '',f'The schema contains {schema["normalized_attempts"]} verified push-attempt identities. {len(schema["omitted_from_normalized_schema"])} attempts are retained in raw/diagnostic archives but omitted from normalized attempts because the tested revision or workflow identity is not proved. No guessed tested SHA or historical decision timestamp is inserted.',
      '', f'In {sum(r["scoped_run_start_equals_creation"] for r in report.values())}/{total_scope} scoped first attempts, provider run_started_at equals created_at exactly. Sampled earliest job starts occur 2–12 seconds later. Blindly mapping run_started_at to the frozen execution-start boundary makes created_at <= decision_at < started_at an empty interval in these observations. The provider run lifecycle timestamp is not yet validated as the intended execution boundary. The later job time is not automatically a valid substitute; obtain authoritative semantics before any new capture claim.',
      '', 'Run completion time remains null. Label availability is the actual first successful terminal-response receipt. The maximum completed job time is a diagnostic and is not substituted for run-label availability. Current updated_at is not a completion timestamp.',
      '', 'Unique changes cannot be counted exactly across PR/merge semantics. Distinct head SHAs and provider PR numbers are reported separately in data-quality.json; neither is asserted to equal independent changes. Repeated SHAs cluster workflows and reruns.',
      '', '## Per-project qualification, coverage and clustering']
    for repo,r in report.items():
        quality += ['',f'### {repo}',f'Provider ID {r["repository_id"]}; {r["language"]}; provider repository-size proxy {r["git_size_kib_proxy"]:,} KiB (not source LOC); nonfork={not r["fork"]}, active={not r["archived"]}. Historical activity older than six months is confirmed by an archived API query (returned count {r["six_month_history_evidence"]["count"]}; filtered counts are not asserted exhaustive).',
          f'Recent sampled coverage {r["earliest_created_at"]}–{r["latest_created_at"]} ({r["source_time_span_days"]:.1f} days). Listed workflows {r["workflows_listed"]}; observed workflows {r["observed_workflows"]}. Selected CI workflow IDs/names: '+', '.join(f'{w["id"]} ({w["name"]})' for w in r['selected_workflows'])+'.',
          f'Observations per workflow: `{json.dumps(r["observations_per_workflow"],sort_keys=True)}`.',
          f'All-attempt terminal categories: `{json.dumps(r["all_status_counts"],sort_keys=True)}`. Scoped categories: `{json.dumps(r["scope_status_counts"],sort_keys=True)}`.',
          f'Scoped head-SHA clusters {r["scoped_unique_head_shas"]}; largest contains {r["largest_sha_cluster"]} first workflow attempts. PR numbers present {r["observed_pr_numbers"]}; PR-attempt rows without a PR number {r["pr_attempts_without_pr_number"]}.',
          f'Change extraction {r["complete_file_inventories"]}/{r["sampled_changes"]} complete inventories; verified push diff basis {r["verified_push_diff_basis_samples"]}/{r["sampled_changes"]}; all sources were retrieved after their original CI runs.',
          f'New runs observed during this session {r["new_runs_observed_during_session"]}; valid preexecution snapshots {r["verified_preexecution"]}. These are sparse polling rounds, not an estimate of a continuously operated collector’s capture probability.']
    quality += ['', f'The largest gap between polling rounds was {max_gap:.2f} hours. Across the session, {len(new_rows)} additional runs were observed, {len(new_scope)} in the selected first-attempt/event/workflow scope; none was captured before its reported start. This demonstrates failure of this intermittent session to create eligible snapshots, not a general impossibility result for polling.',
      '', 'The three projects qualify for retrospective development diagnostics. None yet qualifies as a verified prospective data source under the frozen contract. Python/JavaScript and different test/job structures provide some heterogeneity, but two Python web libraries and one JavaScript web framework form a narrow convenience sample. Three distinct owner/project IDs are an upper bound on independent pilot clusters, not proof of independence or population coverage. No owner participation has been obtained.',
      '', 'The 60 sampled change inventories have source checksums and deterministic extraction. Collection success and variation are not predictive validation. Samples use SHA-hash order and are not selected by CI label. See feature-feasibility.csv for every feature and repository.']
    (FOLDER/'data-quality-report.md').write_text('\n'.join(quality)+'\n')
    rates=[r['scoped_failure_rate'] for r in report.values()]
    precision=['# Precision feasibility scenarios','', 'No policy was fitted or compared. Descriptive outcome variability is **not** the between-project variance of paired policy-cost differences. The latter remains unidentified by this pilot. Do not substitute one for the other.',
      '', '| Repository | Resolved scoped attempts | Failure rate | Within-project Bernoulli variance | Exploratory SHA-cluster bootstrap 95% interval |',
      '|---|---:|---:|---:|---|']
    for repo,r in report.items():precision.append(f'| {repo} | {r["resolved_binary"]} | {r["scoped_failure_rate"]:.4f} | {r["descriptive_binary_variance"]:.4f} | {r["exploratory_sha_bootstrap_95"]} |')
    precision += ['',f'Across these **three selected projects**, sample variance of observed failure proportions is {statistics.variance(rates):.6f}, SD {statistics.stdev(rates):.4f}. This is a descriptive statistic with only two variance degrees of freedom, incompatible observation windows and possible shared infrastructure. It is not a stable population variance estimate. No defensible population variance confidence interval follows from this convenience sample.',
      '', 'The displayed intervals resample whole head-SHA clusters within project (2,000 descriptive bootstrap draws, fixed seed 20260907). They retain within-SHA dependence but not all serial, workflow or cross-project dependence. Treat them as exploratory and potentially too narrow, not confirmatory coverage guarantees. No policy seeds were run.',
      '', 'Effective project-level replication is at most three and may be smaller; genuinely independent future evaluation projects recruited: zero. Proven usable predecision observations per project: zero. Retrospective row counts cannot estimate a future 1,000-valid-decision accrual rate because preexecution capture and diff coverage are unknown.',
      '', '## Unchanged-design sensitivity scenarios',
      'Seven primary contrasts, family alpha .05, planning power .80, proposed normalized-cost effect .025. Scenarios below use the existing frozen normal approximation; they do not amend the evaluation protocol or estimate effects.',
      '', '| Scenario | Assumed paired project-cost SD | Approximate independent evaluation repositories |',
      '|---|---:|---:|']
    for label,sd in [('Optimistic',.025),('Moderate',.05),('Conservative illustration',.10),('High heterogeneity',.20),('Bounded-difference stress case',1.0)]:
        precision.append(f'| {label} | {sd} | {required_clusters(sd,.025,power=.8,comparisons=7)} |')
    precision += ['', 'These are optimistic known-SD normal calculations. Actual planning needs justified variance assumptions, finite-project correction, dependence sensitivity and a recruitment budget. “Conservative illustration” is not an upper bound. Pilot failure-rate heterogeneity does not tell us which cost-difference scenario applies. Smaller effects or Brier margins require their own scenarios.',
      '', 'At an assumed 2% failure rate, 1,000 valid decisions yield only 20 expected failures; at 5%, 50. Correlation, incomplete capture and nonbinary terminations reduce information further. None of the scoped pilot projects meets the frozen 50-failure/50-success subgroup reporting screen, and that screen itself never guaranteed validity.',
      '', 'No recruitment commitments, long-run collector capture fraction, verified predecision accrual rate or feasible independent-project budget has been established. Precision gate: FAIL for scaling now. Narrow exploratory scope or an explicitly reviewed acquisition/precision plan is needed before expansion.']
    (FOLDER/'precision-scenarios.md').write_text('\n'.join(precision)+'\n')
    all_features=[f for f in feature_rows if f['cohort']=='ALL']
    varied=[f['feature'] for f in all_features if f['unique_values']>1 and f['group']=='change']
    unavailable=[f['feature'] for f in all_features if f['diagnostic_nonnull']==0]
    feasibility=['# Data-only development feasibility decision','', '**NO-GO — do not scale the current collection route into the frozen empirical study.**',
      '', 'Public REST acquisition and deterministic extraction work. The central pre-execution observation contract has not been demonstrated: there are zero authentic predecision snapshots, PR tested/diff references are incompletely established, and no participating event-capture arrangement is available. This verdict concerns the currently demonstrated route, not a proof that owner-assisted prospective collection is impossible.',
      '', f'The session made {len(journal)} archived read-only requests across three explicitly designated development repositories. It recovered {total_runs} distinct runs and {len(rows)} valid attempts, with 60 deterministic change samples. The later polling round found {len(new_rows)} additional runs, including {len(new_scope)} scoped first attempts, all first seen after their reported start. The longest unmonitored gap was {max_gap:.2f} hours. This is not a continuous capture study and cannot estimate reliable queued-run capture probability.',
      '', '| Gate | Decision | Evidence / limitation |','|---|---|---|',
      '| A — Collection | CONDITIONAL PASS | Public identities, historical outcomes and sampled diff inventories are retrievable. Reliable preexecution collection is not demonstrated; historical frame is deliberately bounded. |',
      '| B — Timing | FAIL | Zero verified preexecution snapshots; actual receipt timestamps are honest but cannot recover earlier availability. All scoped first attempts have run_started_at==created_at, leaving no permissible interval under that field mapping. Actual execution-boundary semantics, PR refs and completion fields remain unverified. |',
      '| C — Context | CONDITIONAL PASS | Several change attributes vary in retrospective extraction. None has survived as a verified predecision predictor; variation alone establishes no risk-prediction value. |',
      '| D — Outcomes | FAIL | Both binary classes exist in scoped historical records, but zero temporally eligible observations and insufficient per-project event counts for the frozen subgroup screen. |',
      '| E — Project diversity | FAIL | Three narrow development projects; no recruited independent evaluation cohort or participation frame. |',
      '| F — Reproducibility | CONDITIONAL PASS | Archived-source transformation is hash-verifiable; fresh-environment evidence is recorded separately. Authentic receipt times cannot be reproduced by later API refetch, and long-running/webhook operation is untested. |',
      '| G — Precision | FAIL | Paired cost-difference variance and prospective valid-observation accrual are unknown; no realistic commitment to the required independent-project scenarios. |',
      '', '## Which features survived?',
      '**No feature is certified for the future evaluation yet.** Retrospective extraction shows variation in: '+', '.join(varied)+'. These remain QUESTIONABLE for predecision use. The CSV separates retrospective collection rates from strict rates; strict missingness/validity is null when the eligible denominator is zero.',
      '', 'No measured value under this pilot route for: '+', '.join(unavailable)+'. These are NOT COLLECTABLE from the current archived cohort under the required semantics; this is not a universal statement about future collectors. Attempt number is constant by primary design and UNUSABLE as a primary predictor. Any constant change indicator is reported separately as UNUSABLE in that sampled cohort.',
      '', 'Prior duration fails because authentic run completion is absent. Author history lacks verified eligible outcome history and defensible identity attribution. Project/workflow rates cannot be retroactively populated with labels that were first observed today. Current source metadata can support future decisions once actually observed; it cannot repair old snapshots.',
      '', '## Interpretation and next decision',
      'Meaningful contextual prediction is **plausible but unestablished**. The source APIs expose nonconstant change content, which justifies investigating an acquisition route. There are no predecision data here with which to establish contextual predictive adequacy, and no model comparison was performed.',
      '', 'The largest unresolved threat is the operational timing contract. In every scoped first attempt, provider run_started_at equals created_at; the currently assumed field mapping makes the frozen permitted decision interval empty. Job execution starts later in sampled data, but substituting the first job time without validating its semantics would change the evidence. Establish the actual execution boundary with authoritative provider/event evidence, or seek a newly reviewed protocol; never invent an earlier snapshot. Capture selection and temporal provenance also matter: a collector that observes runs only after they start or finish cannot support this pre-execution study, regardless of how many diffs or policy seeds are added. The sparse rounds and long unmonitored gap neither validate nor disprove the reliability of a properly operated prospective collector.',
      '', 'Genuinely independent repositories obtainable for evaluation are unknown; zero have been recruited. The three queried projects are consumed as development data and cannot be recycled as holdouts. Distinct owner IDs and language differences do not prove statistical independence.',
      '', 'This remains worth pursuing **only if** participating repositories or a trustworthy prospective event archive can supply the sealed pre-execution sources, and a credible recruitment/precision budget follows. Do not spend on a larger retrospective scrape to rescue the paper. If that acquisition route is unavailable, stop this empirical design or explicitly change the research question through a new reviewed protocol. The frozen protocol has not been edited.',
      '', 'The next decision is whether to secure owner-assisted pre-execution collection and run a bounded data-only development capture pilot. No invitations, webhook installations, repository changes, holdout queries or model runs are authorized by this report.']
    (FOLDER/'data-feasibility-report.md').write_text('\n'.join(feasibility)+'\n')
    source_files=['review/pilot/collector.py','review/pilot/pipeline.py','review/pilot/report.py','tests/test_pilot_integration.py','review/pilot/repository-selection.md']
    raw_hashes={p.name:digest(p.read_bytes()) for p in sorted((FOLDER/'raw').glob('*.json'))}
    lock=json.loads((ROOT/'review/design/design-lock.json').read_text())
    base=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    manifest={'pilot_version':'ci-design-v1-pilot-1','schema_version':'ci-design-v1','parent_design_sha256':lock['snapshot_sha256'],
      'collector_base_commit':base,'collector_is_uncommitted':True,'code_identified_by_sha256_not_base_commit_alone':True,
      'source_sha256':{p:digest((ROOT/p).read_bytes()) for p in source_files},
      'collection_started_at':first.isoformat(),'collection_ended_at':last.isoformat(),'api_version':API_VERSION,
      'repositories':{r:{'id':v['repository_id'],'split':'development','family':r.split('/')[0]} for r,v in report.items()},
      'requests':len(journal),'endpoints':sorted({x['endpoint'] for x in journal}),'raw_source_sha256':raw_hashes,
      'source_journal_sha256':digest((FOLDER/'source-journal.jsonl').read_bytes()),
      'acquisition_state_sha256':digest((FOLDER/'acquisition-state.json').read_bytes()),
      'transformed_sqlite_sha256':schema['sqlite_sha256'],'logical_dataset_sha256':schema['logical_dataset_sha256'],
      'platform':platform.platform(),'python':platform.python_version(),'pyyaml':__import__('yaml').__version__,
      'gh_version':subprocess.check_output(['gh','--version'],text=True).splitlines()[0],
      'new_policy_runs':0,'holdout_repositories_queried':[],'manuscript_modified':False,'final_decision':'NO-GO',
      'scope':'NO-GO for scaling currently demonstrated acquisition route; prospective owner-assisted feasibility remains unresolved'}
    write_json(FOLDER/'collection-manifest.json',manifest)
    print(json.dumps({'requests':len(journal),'runs':total_runs,'attempts':len(rows),'features':len(feature_rows),'scoped':total_scope,'decision':'NO-GO'}))


if __name__=='__main__':generate()
