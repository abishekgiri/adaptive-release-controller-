"""Build manuscript numbers, source of truth, supplement, and scientific figures."""
from pathlib import Path
import json, shutil, re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
RESULT=ROOT/'experiments/results/headline/corrected'
PAPER=ROOT/'paper'
FIG=PAPER/'figures'
S=json.loads((RESULT/'summary.json').read_text())['conditions']
SW=json.loads((RESULT/'cost_sweep.json').read_text())
AB=json.loads((RESULT/'ablation.json').read_text())['policies']
DR=json.loads((RESULT/'drift/drift_eval_summary.json').read_text())['conditions']
CAL=json.loads((RESULT/'page_hinkley_calibration.json').read_text())
SENS=json.loads((RESULT/'sensitivity.json').read_text())
LABEL={'static_rules':'Static rules','linucb':'LinUCB','thompson':'Thompson',
       'cost_rule':'Rolling cost rule','bayesian_rate':'Bayesian rate','always_block':'Always block',
       'always_deploy':'Always deploy','always_canary':'Always canary','linucb_bias_only':'Bias-only LinUCB',
       'heuristic_score':'Heuristic','linucb_with_drift':'LinUCB wrapper',
       'linucb_with_drift_full':'PH threshold 50','linucb_with_drift_calibrated':'PH threshold 400',
       'moment_rate':'Scalar moment rate','moment_rate_window_50':'Rate window 50','moment_rate_window_100':'Rate window 100'}
COLORS=['#3d405b','#0077b6','#e07a5f','#2a9d8f','#6a4c93','#8d6e63']
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                     'figure.dpi':130,'savefig.dpi':180,'axes.prop_cycle':plt.cycler(color=COLORS)})


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(map(str,row))+' |' for row in rows])


def fmt(v):return f'{v:,.1f}'.rstrip('0').rstrip('.')
def mean(condition,policy):return S[condition]['policies'][policy]['mean_cost']
def display(condition,pid):
    r=S[condition]['policies'][pid]
    return fmt(r['mean_cost'])+(f" ± {r['sd_cost']:.1f}" if pid=='thompson' else '')
def save(fig,name):
    fig.tight_layout();fig.savefig(FIG/f'{name}.png',bbox_inches='tight')
    fig.savefig(FIG/f'{name}.pdf',bbox_inches='tight',metadata={'CreationDate':None,'ModDate':None});plt.close(fig)
def costs(condition,pid):
    seeds=range(30) if pid=='thompson' else [0]
    return np.stack([np.load(RESULT/'runs'/condition/str(seed)/f'step_costs_{pid}.npy') for seed in seeds])
def curves(ax,condition):
    policies=['static_rules','linucb','thompson','bayesian_rate' if condition=='online_smoke' else 'cost_rule','always_block']
    for pid in policies:
        a=costs(condition,pid);curve=np.nancumsum(a,axis=1).mean(axis=0)
        ax.plot(np.arange(1,len(curve)+1),curve,label=LABEL[pid],lw=1.6)
    ax.axvline(600 if condition=='online_smoke' else 300,color='grey',ls='--',lw=.8)
    ax.set(xlabel='Decision index (projects concatenated)',ylabel='Cumulative observed proxy cost')
    ax.legend(fontsize=8,ncol=2,loc='upper left')


def build_figures():
    FIG.mkdir(exist_ok=True)
    for condition,name in [('online_smoke','fig_cumulative_cost_synthetic'),('real_github_actions','fig_cumulative_cost_real')]:
        fig,ax=plt.subplots(figsize=(7,3.8));curves(ax,condition);save(fig,name)
    fig,axs=plt.subplots(1,2,figsize=(11,3.8))
    for ax,c in zip(axs,['online_smoke','real_github_actions']):curves(ax,c);ax.set_title(c.replace('_',' '))
    save(fig,'fig_cumulative_cost_curves')
    fig,axs=plt.subplots(1,2,figsize=(9,3.8),sharey=True)
    pids=['static_rules','linucb','cost_rule']
    for ax,project in zip(axs,['pallets/flask','psf/requests']):
        bottom=np.zeros(3)
        for action,color in zip(['deploy','canary','block'],COLORS):
            counts=np.array([S['real_github_actions']['policies'][p]['seed0']['per_project'][project]['action_counts'][action] for p in pids])
            ax.bar(range(3),counts,bottom=bottom,label=action,color=color);bottom+=counts
        ax.set_xticks(range(3),[LABEL[p] for p in pids],rotation=15,ha='right');ax.set_title(project);ax.set_ylabel('Actions (including unresolved rows)')
    axs[0].legend(fontsize=8);save(fig,'fig_action_distribution')
    fig,ax=plt.subplots(figsize=(7,3.8))
    for pid in ['static_rules','linucb','thompson','bayesian_rate','always_block']:
        a=costs('online_smoke',pid).flatten();a=np.sort(a[np.isfinite(a)])
        ax.step(a,np.arange(1,len(a)+1)/len(a),where='post',label=LABEL[pid])
    ax.set(xlabel='Synthetic proxy cost per decision',ylabel='Empirical cumulative fraction');ax.legend(fontsize=8);save(fig,'fig_cost_cdf_per_step')
    fig,ax=plt.subplots(figsize=(7,3.8));levels=['5:1','10:1','20:1','40:1','100:1']
    for pid in ['static_rules','linucb','thompson','bayesian_rate','always_block']:
        ax.plot(levels,[SW[l]['policies'][pid]['mean_cost'] for l in levels],marker='o',label=LABEL[pid])
    ax.set(xlabel='Cost-matrix index (deploy failure / block bad)',ylabel='Cumulative synthetic proxy cost');ax.legend(fontsize=8,ncol=2);save(fig,'fig_cost_sweep')
    fig,axs=plt.subplots(1,3,figsize=(11,3.5),sharey=True)
    for ax,mode in zip(axs,['none','abrupt','gradual']):
        for pid in ['static_rules','linucb','linucb_with_drift_full','moment_rate_window_50']:
            a=np.load(RESULT/'drift'/f'step_regrets_{mode}_{pid}.npy');ax.plot(np.arange(1,len(a)+1),np.cumsum(a),label=LABEL[pid])
        if mode=='abrupt':ax.axvline(250,color='grey',ls='--',lw=.8)
        ax.set(title=mode,xlabel='Decision index')
    axs[0].set_ylabel('Cumulative expected pseudo-regret');axs[-1].legend(fontsize=7);save(fig,'fig_drift_recovery_curves')
    fig,ax=plt.subplots(figsize=(8,3.8));modes=['none','abrupt','gradual'];pids=['linucb','linucb_with_drift_full','moment_rate','moment_rate_window_50']
    for i,pid in enumerate(pids):ax.bar(np.arange(3)+(i-1.5)*.18,[DR[m]['policies'][pid]['mean_regret'] for m in modes],width=.18,label=LABEL[pid])
    ax.set_xticks(range(3),modes);ax.set_ylabel('Mean expected pseudo-regret');ax.legend(fontsize=8,ncol=2);save(fig,'fig_drift_mode_bars')
    fig,ax=plt.subplots(figsize=(7,3.5));names=['full','no_delay','no_cost','no_drift']
    ax.bar(names,[AB[n]['cumulative_cost'] for n in names],color=COLORS[:4]);ax.set_ylabel('Synthetic proxy cost (deterministic)');save(fig,'fig_ablation_bars')
    fig,axs=plt.subplots(1,2,figsize=(9,3.5))
    for ax,c in zip(axs,['online_smoke','real_github_actions']):
        ax.hist(S[c]['policies']['thompson']['per_seed_costs'],bins=9,color=COLORS[2],alpha=.85)
        for pid,col in [('static_rules',COLORS[0]),('bayesian_rate' if c=='online_smoke' else 'cost_rule',COLORS[3])]:ax.axvline(mean(c,pid),color=col,label=LABEL[pid])
        ax.set(title=c.replace('_',' '),xlabel='Thompson cumulative proxy cost',ylabel='Seeds');ax.legend(fontsize=8)
    save(fig,'fig_thompson_seed_distribution')
    for name in ['fig_cost_cdf_per_step','fig_drift_recovery_curves','fig_thompson_seed_distribution']:
        shutil.copyfile(FIG/f'{name}.pdf',PAPER/'supplementary/figures'/f'{name}.pdf')


def build_text():
    pids=['static_rules','linucb','thompson','cost_rule','bayesian_rate','linucb_bias_only','always_deploy','always_canary','always_block']
    main_table=table(['Policy','Synthetic cost','Real proxy cost'],[[LABEL[p],display('online_smoke',p),display('real_github_actions',p)] for p in pids])
    gain_table=table(['Setting','Static','LinUCB','Thompson mean','Always block'],[[n,*[fmt(mean(c,p)) for p in ['static_rules','linucb','thompson','always_block']]] for n,c in [('High failure','robustness_high_failure'),('Low block','robustness_low_block')]])
    sweep_table=table(['Cost index','Static','LinUCB','Bayesian rate'],[[l,*[fmt(SW[l]['policies'][p]['mean_cost']) for p in ['static_rules','linucb','bayesian_rate']]] for l in ['5:1','10:1','20:1','40:1','100:1']])
    drift_table=table(['Policy','Stationary','Abrupt','Gradual'],[[LABEL[p],*[fmt(DR[m]['policies'][p]['mean_regret']) for m in ['none','abrupt','gradual']]] for p in ['static_rules','linucb','thompson','linucb_with_drift_full','linucb_with_drift_calibrated','moment_rate','moment_rate_window_50','moment_rate_window_100']])
    values={'main_table':main_table,'gain_table':gain_table,'sweep_table':sweep_table,'drift_table':drift_table}
    for key,c,p in [('bayes_smoke','online_smoke','bayesian_rate'),('linucb_smoke','online_smoke','linucb'),('static_smoke','online_smoke','static_rules'),('rule_real','real_github_actions','cost_rule'),('static_real','real_github_actions','static_rules'),('linucb_real','real_github_actions','linucb')]:values[key]=fmt(mean(c,p))
    ts=S['real_github_actions']['policies']['thompson'];values['ts_real']=display('real_github_actions','thompson');values['ts_real_ci']=f"{fmt(ts['mean_cost'])} (95% seed-bootstrap interval [{fmt(ts['ci95'][0])}, {fmt(ts['ci95'][1])}])"
    values['ablation_values']=', '.join(f"{n.replace('_',' ')} {fmt(AB[n]['cumulative_cost'])}" for n in ['full','no_delay','no_cost','no_drift'])
    values['delay_values']=', '.join(f"{q}: {fmt(mean(c,'linucb'))}" for q,c in [('short','robustness_short_delay'),('reference','artificial_default_delay'),('long','robustness_long_delay')])
    values['ph_threshold']=str(CAL['selected_threshold'])
    values['calibrated_drift_resets']='zero model resets in all three modes'
    for key,c in [('dedup_values','exact_unique'),('commit_values','first_per_commit')]:values[key]=', '.join(f"{LABEL[p]} {fmt(SENS[c]['policies'][p]['mean_cost'])}" for p in ['static_rules','linucb','thompson','cost_rule'])
    text=(PAPER/'manuscript-template.md').read_text().replace('Noureddine et al., 2016','Kerzazi and Adams, 2016')
    for key,value in values.items():text=text.replace('{{'+key+'}}',value)
    assert '{{' not in text
    (PAPER/'adaptive-deployment-control.md').write_text(text)
    truth=['# Corrected source of truth','',
        'Generated from the corrected artifacts by `python paper/build_corrected.py`. Original a437abf results are superseded and preserved in `review/original/`.',
        '', '## Protocol', '', 'Seeds 0-29; 10,000 bootstrap draws; bootstrap seed 42. Main replay uses completion-time feedback and historical CI features. Artificial delay conditions use the explicitly labeled decision clock. All real policies use 578 observed outcomes. Intervals describe algorithm randomness conditional on the supplied datasets.', '',
        'Dataset and code hashes, exact environment, and command: `experiments/results/headline/corrected/manifest.json`. Per-run resolved configs and project traces: `corrected/runs/<condition>/<seed>/`.', '',
        '## Main claims', '',main_table,'',
        'Source: `corrected/summary.json#/conditions/{online_smoke,real_github_actions}/policies/<policy>`. Mean, sample SD, CI, per-seed costs, and seed-0 project actions are stored together.', '',
        '## Cost settings', '',gain_table,'',sweep_table,'',
        'Sources: `corrected/summary.json#/conditions/robustness_*` and `corrected/cost_sweep.json`. Full cost matrices are included. No monotonicity or 40:1 boundary is claimed.', '',
        '## Ablation', '',values['ablation_values']+'. All model-reset counts are zero.', '',
        'Source: `corrected/ablation.json`; deterministic seed 0. Binary reward changes objective and scale.', '',
        '## Drift', '',drift_table,'','Source: `corrected/drift/drift_eval_summary.json`. Table is expected pseudo-regret, not realized-label regret.', '',
        'Calibration: `corrected/page_hinkley_calibration.json`; selected threshold '+values['ph_threshold']+'. No guarantee of a population false-alarm rate.', '',
        '## Additional checks','', 'Sensitivity: `corrected/sensitivity.json` and `corrected/alpha_sensitivity.json`. Original per-project attribution and UCB scores: `review/evidence/controlled-original-attribution.json`.', '',
        'Retired claims: 19%/27% as evidence for contextual necessity; 44 valid Page-Hinkley false alarms; O(d²) convergence floor; cross-project confusion; monotone cost advantage; p<0.0001 from uncentered bootstrap tails; statistical equivalence from CI inclusion.', '',
        '## Figure lineage','', 'All ten figures are regenerated by `paper/build_corrected.py` from the above JSON and `runs/.../step_costs_*.npy` or `drift/step_regrets_*.npy`. No figure imports archived original results. Curves retain project boundaries; missing costs are excluded explicitly.']
    (PAPER/'source-of-truth.md').write_text('\n'.join(truth)+'\n')
    supplementary=['---',"title: 'Supplementary Evidence for the Corrected CI Decision Simulation'","author: 'Abishek Kumar Giri'","date: '6 September 2026'",'---','',
        '# Scope','', 'This supplement replaces all numerical tables and figures from the original submission. The source-of-truth file and machine-readable artifacts define the corrected numerical record. Original materials are retained under review/original. These are simulation experiments, not causal release-cost estimates.','',
        '# Full replay conditions','']
    for c,v in S.items():
        supplementary += ['## '+c.replace('_',' '),'',table(['Policy','Mean cost','Sample SD','95% seed CI'],[[LABEL.get(p,p),fmt(r['mean_cost']),fmt(r['sd_cost']),f"{fmt(r['ci95'][0])} to {fmt(r['ci95'][1])}"] for p,r in v['policies'].items()]),'']
    supplementary += ['# Page Hinkley calibration','', 'Fixed deploy, IID failure labels, 1,150 observations, seeds 1000-1029. The selected threshold is 400; selection criterion is no more than 10% of streams with any alarm in each of two null cells. This finite-sample screen is not a theoretical error bound.','',
        table(['Threshold','Failure rate','Mean alarms','Fraction with any alarm'],[[k,c['failure_probability'],fmt(c['mean_alarms']),f"{c['fraction_any']:.3f}"] for k,cells in sorted(CAL['cells'].items(),key=lambda x:int(x[0])) for c in cells]),'',
        '# Drift evaluation','', 'Mean cumulative expected pseudo-regret uses all 500 decisions. Realized costs exclude a common censored-delay mask. Terminal observations are included in cost evaluation but not in model learning. The moment-rate control accounts for the synthetic canary multiplier; CI replay controls assume an action-independent label.','',drift_table,'']
    for name,caption in [('fig_drift_recovery_curves','Expected pseudo-regret by decision index'),('fig_cost_cdf_per_step','Empirical synthetic cost distribution'),('fig_thompson_seed_distribution','Thompson seed costs and simple comparators'),('fig_action_distribution','Project-local real-data action counts'),('fig_drift_mode_bars','Expected pseudo-regret across drift modes'),('fig_ablation_bars','Corrected deterministic ablation costs')]:supplementary += [f'![{caption}](../figures/{name}.png){{width=95%}}','']
    supplementary += ['# Reproducibility','', 'Run the commands in the root README from a clean checkout. The frozen CSV hashes and code hashes are in corrected/manifest.json. This run used Python 3.13 and the exact dependency versions in requirements-audit.txt. The main build scripts use Pandoc and a TeX installation.','',
        '# Remaining limitations','', 'The synthetic fixture generator is unavailable. The real export lacks collection provenance and workflow identities; it cannot be interpreted as independent deployment decisions. Deterministic policy repetition does not supply dataset uncertainty. Hyperparameter sensitivity is exploratory on the same export. Real opportunity costs, deployment incidents, action-dependent censoring and policy effects on future releases are unmeasured. The original empirical operating-boundary claim is withdrawn.']
    (PAPER/'supplementary/supplementary.md').write_text('\n'.join(supplementary)+'\n')
    # Supersede previously standalone subfiles to avoid leaving contradictory text.
    for name in ['appendix_a','appendix_b','diagnostic_tables','setup_tables','threats_extended']:
        (PAPER/'supplementary'/f'{name}.tex').write_text('% Superseded by the generated corrected supplementary.tex.\n% Source: supplementary.md; build with paper/build_documents.sh.\n')
    (RESULT.parent/'README.md').write_text('# Headline artifacts\n\nThe numerical record is `corrected/`, generated by `python -m experiments.reproduce_submission`. Old headline paths now contain corrected results or an explicit source pointer. Original published values are archived in `review/original/headline/` and must not be cited as current evidence. See `paper/source-of-truth.md` and `review/submission-review.md`.\n')

if __name__=='__main__':
    build_figures();build_text();print('Built corrected figures and manuscript sources.')
