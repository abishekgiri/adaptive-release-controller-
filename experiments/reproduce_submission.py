"""Reproduce the corrected submission audit from frozen public fixtures.

Run from the repository root: python -m experiments.reproduce_submission
All stochastic comparisons use seeds 0..29. Dataset uncertainty is not estimated
by resampling algorithm seeds. Original published artifacts live in review/original.
"""
from dataclasses import asdict, replace
from pathlib import Path
import csv
import hashlib
import json
import platform
import shutil
import sys
import numpy as np

from data.schemas import Action
from drift.detectors import PageHinkleyConfig, PageHinkleyDetector
from experiments.run_bandits import load_config, run_experiment
from experiments.run_ablations import AblationConfig, run_ablation_experiment, build_summary
from experiments.run_cost_sweep import COST_LEVELS
from experiments.run_drift_eval import run_drift_study
from evaluation.statistical import BootstrapConfig, bootstrap_ci, paired_bootstrap_pvalue, holm_bonferroni
from rewards.cost_model import CostConfig

ROOT = Path('experiments/results/headline/corrected')
SEEDS = list(range(30))
BOOT = BootstrapConfig(seed=42)


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n')


def aggregate(runs):
    result = {}
    for pid in runs[0]['policies']:
        costs = [r['policies'][pid]['cumulative_cost'] for r in runs]
        mean, lo, hi = bootstrap_ci(costs, BOOT)
        result[pid] = dict(mean_cost=mean, sd_cost=float(np.std(costs, ddof=1)),
            ci95=[lo,hi], per_seed_costs=costs,
            seed0=runs[0]['policies'][pid])
    # Prespecified descriptive seed comparisons, family of three within dataset.
    pairs = [('thompson','linucb'), ('thompson','static_rules'), ('thompson','cost_rule')]
    pvalues = [paired_bootstrap_pvalue(np.array(result[a]['per_seed_costs']),
               np.array(result[b]['per_seed_costs']), BOOT) for a,b in pairs]
    tests = []
    for (a,b),p,reject in zip(pairs,pvalues,holm_bonferroni(pvalues)):
        differences = np.array(result[a]['per_seed_costs'])-result[b]['per_seed_costs']
        m,l,h = bootstrap_ci(differences,BOOT)
        tests.append(dict(pair=[a,b], mean_difference=m, ci95=[l,h],
                          p_two_sided_centered=p, holm_reject=reject))
    return dict(policies=result, paired_seed_comparisons=tests)


def run_config(config):
    print(f'Running {config.config_name}: 30 seeds',flush=True)
    return aggregate([run_experiment(replace(config,results_root=ROOT/'runs'),seed) for seed in SEEDS])


def dataset_sensitivity():
    source = Path('data/fixtures/github_actions_real.csv')
    with source.open() as f:
        reader=csv.DictReader(f); fields=reader.fieldnames; rows=list(reader)
    variants={}
    for name,key in [('exact_unique',lambda r:tuple(r.items())),
                     ('first_per_commit',lambda r:(r['gh_project_name'],r['git_trigger_commit']))]:
        seen=set(); selected=[]
        for r in sorted(rows,key=lambda r:r['tr_started_at']):
            k=key(r)
            if k not in seen: selected.append(r); seen.add(k)
        path=ROOT/'fixtures'/f'{name}.csv'; path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('w',newline='') as f:
            w=csv.DictWriter(f,fields); w.writeheader();w.writerows(selected)
        cfg=replace(load_config('experiments/configs/real_github_actions.json'),
                    config_name=name,dataset_path=path,min_builds=1)
        variants[name]=run_config(cfg)
        variants[name]['rows']=len(selected)
    return variants


def calibration():
    # Fixed-action stationary null, independent seeds. This is a diagnostic
    # calibration, not a guarantee under an adapting policy's cost distribution.
    results={}
    for threshold in [20,50,100,200,400,800]:
        cells=[]
        for p in [.10,.35]:
            counts=[]
            for seed in range(1000,1030):
                stream=10*(np.random.default_rng(seed).random(1150)<p)
                detector=PageHinkleyDetector(PageHinkleyConfig(lambda_=threshold))
                counts.append(sum(detector.update(float(x)) for x in stream))
            cells.append(dict(failure_probability=p,per_seed_alarms=counts,
                mean_alarms=float(np.mean(counts)),fraction_any=float(np.mean(np.array(counts)>0))))
        results[str(threshold)]=cells
    candidates=[int(k) for k,v in results.items() if max(c['fraction_any'] for c in v)<=.1]
    return dict(seeds=list(range(1000,1030)), stream_length=1150,
                null='iid Bernoulli failure, fixed deploy, cost 0 or 10',
                criterion='fraction with any alarm <= 0.1 in both null cells',
                selected_threshold=min(candidates) if candidates else None, cells=results)


def drift_comparisons(report):
    comparisons = {}
    for mode, condition in report['conditions'].items():
        policies = condition['policies']
        differences = np.array(policies['moment_rate_window_50']['per_seed_regrets']) - policies['linucb']['per_seed_regrets']
        mean,lo,hi = bootstrap_ci(differences,BOOT)
        comparisons[mode] = dict(pair=['moment_rate_window_50','linucb'],
                                mean_difference=mean,ci95=[lo,hi],per_seed_differences=differences.tolist())
    return dict(note='Exploratory paired environment-seed comparison added during audit; no confirmatory family claim',comparisons=comparisons)


def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    reports={}
    for name in ['online_smoke','real_github_actions','robustness_high_failure','robustness_low_block',
                 'robustness_short_delay','robustness_long_delay']:
        cfg=load_config(f'experiments/configs/{name}.json')
        reports[name]=run_config(cfg)
    default=load_config('experiments/configs/online_smoke.json')
    reports['artificial_default_delay']=run_config(replace(default,config_name='artificial_default_delay',timing_mode='duration_steps'))
    sweep={}
    for level in COST_LEVELS:
        cfg=replace(default,config_name='cost_'+level.label.replace(':','_'),cost_config=level.to_cost_config())
        sweep[level.label]=dict(cost_config=asdict(cfg.cost_config),**run_config(cfg))
    dump(ROOT/'cost_sweep.json',sweep)
    abl=AblationConfig(results_root=ROOT/'runs')
    results,resets=run_ablation_experiment(abl,0)
    ablation=build_summary(abl,0,results,resets)
    ablation['resolved_config']={**asdict(abl),'dataset_path':str(abl.dataset_path),'results_root':str(abl.results_root)}
    ablation['note']='Deterministic policies: one seed. Immediate feedback is an optimistic hypothetical intervention.'
    dump(ROOT/'ablation.json',ablation)
    sensitivity=dataset_sensitivity()
    dump(ROOT/'sensitivity.json',sensitivity)
    # Alpha sensitivity on the same fixed data is exploratory, not tuning and testing.
    alpha={}
    real=load_config('experiments/configs/real_github_actions.json')
    for value in [.1,1.,5.,10.]:
        cfg=replace(real,config_name=f'alpha_{value}',linucb_alpha=value,cost_sensitive_alpha=value,results_root=ROOT/'runs')
        alpha[str(value)]=run_experiment(cfg,0)
    dump(ROOT/'alpha_sensitivity.json',alpha)
    cal=calibration();dump(ROOT/'page_hinkley_calibration.json',cal)
    import experiments.run_drift_eval as drift
    drift.CALIBRATED_THRESHOLD=cal['selected_threshold']
    drift_report=run_drift_study(SEEDS,500,CostConfig(),ROOT/'drift')
    dump(ROOT/'drift/paired_window_comparisons.json',drift_comparisons(drift_report))
    dump(ROOT/'summary.json',dict(seeds=SEEDS,bootstrap=asdict(BOOT),conditions=reports,
        evaluation='CI proxy simulation; common observed-label cohort; event-time except labelled delay stress',
        uncertainty='Algorithm seed variability conditional on fixed datasets; not project population uncertainty'))
    # Replace old headline locations too so no stale active headline remains.
    aliases={'gain_high_failure__seed0.json':'robustness_high_failure',
             'gain_low_block__seed0.json':'robustness_low_block'}
    for name,condition in aliases.items():
        dump(ROOT.parent/name,dict(corrected=True,source=f'corrected/summary.json#/conditions/{condition}',
                                  **reports[condition]))
    dump(ROOT.parent/'ablation__seed0.json',ablation)
    dump(ROOT.parent/'cost_sweep__seeds0-29.json',sweep)
    (ROOT.parent/"real_github_actions").mkdir(parents=True,exist_ok=True)
    for seed in SEEDS:
        shutil.copyfile(ROOT/'runs'/'real_github_actions'/str(seed)/'online_summary.json',
                        ROOT.parent/'real_github_actions'/f'seed{seed}__online_summary.json')
    files=sorted([*Path('data/fixtures').glob('*.csv'),*[p for d in ['data','delayed','drift','environment','evaluation','experiments','features','policies','rewards'] for p in Path(d).glob('*.py')],*Path('experiments/configs').glob('*.json')])
    dump(ROOT/'manifest.json',dict(base_commit='a437abf7ecd51e1e9da50834c72b56628e33c697',
        python=sys.version,numpy=np.__version__,platform=platform.platform(),
        command='python -m experiments.reproduce_submission',
        sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}))
    print('Corrected reproduction complete.',flush=True)

if __name__=='__main__': main()
