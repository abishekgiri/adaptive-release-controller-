"""Controlled diagnosis against the preserved, unmodified public-main worktree.
Run with its path as argv[1]; writes only under the active repository's review/.
"""
import sys
from pathlib import Path
OUT=Path(__file__).resolve().parent/'evidence'
sys.path.insert(0,str(Path(sys.argv[1]).resolve()))
from dataclasses import replace,asdict
from datetime import timedelta
import json
import numpy as np
from data.loaders import TravisTorrentLoader
from data.schemas import Action, Outcome
from policies.linucb import LinUCBPolicy,LinUCBConfig
from policies.base import FeatureEncoder
from rewards.cost_model import CostConfig
from evaluation.online_replay import run_online_trajectory

class Scores(LinUCBPolicy):
    def select_action(self,context):
        x=self._encoder.encode(context)
        scores={}
        for a in Action:
            point=float(np.linalg.solve(self._A[a],self._b[a])@x)
            width=float(self._config.alpha*np.sqrt(max(0,x@np.linalg.solve(self._A[a],x))))
            scores[a.value]={'predicted_reward':point,'width':width,'ucb':point+width}
        action,prop=super().select_action(context)
        self.trace.append({'step':context.step,'action':action.value,'scores':scores})
        return action,prop

root=Path(__file__).resolve().parents[1]
records=list(TravisTorrentLoader(root/'data/fixtures/github_actions_real.csv',min_builds=1,min_history_days=0))
output={}
for project in sorted({r.context.project_slug for r in records}):
    rows=[r for r in records if r.context.project_slug==project]
    finish_fixed=[]
    overlaps=0; changed=0
    for i,r in enumerate(rows):
        t=r.started_at
        # Isolate availability timestamps, retaining the original denominator's
        # treatment of censored outcomes to avoid combining separate fixes.
        relevant=[q for q in rows[:i] if q.finished_at and t-timedelta(days=7)<=q.finished_at<=t]
        rate=round(sum(q.outcome==Outcome.FAILURE for q in relevant)/len(relevant),4) if relevant else 0.
        changed+=rate!=r.context.recent_failure_rate
        overlaps+=any(q.finished_at and q.finished_at>t for q in rows[:i])
        finish_fixed.append(replace(r,context=replace(r.context,recent_failure_rate=rate)))
    output[project]={'overlapping_decisions':overlaps,'rate_changed_by_finish_clock':changed,'variants':{}}
    for name,data,alpha in [('original',rows,1),('finish_rate_only',finish_fixed,1),
                            ('alpha_0.1',rows,.1),('alpha_5',rows,5),('alpha_10',rows,10)]:
        p=Scores(LinUCBConfig(alpha=alpha,lambda_reg=1),FeatureEncoder.DIM,np.random.default_rng(0),policy_id='linucb')
        p.trace=[]
        result=run_online_trajectory(p,data,cost_config=CostConfig(),rng=np.random.default_rng(0),trajectory_id=project)
        output[project]['variants'][name]={'cost':result.cumulative_cost,'actions':result.action_counts,
            'score_trace':p.trace,'cost_trace':[{**asdict(s),'cost':s.cost if np.isfinite(s.cost) else None} for s in result.step_records]}
OUT.mkdir(exist_ok=True)
(OUT/'controlled-original-attribution.json').write_text(json.dumps(output,indent=2,allow_nan=False))
for project,v in output.items():
 print(project,v['overlapping_decisions'],v['rate_changed_by_finish_clock'])
 for name,r in v['variants'].items(): print(name,r['cost'],r['actions'])
