"""Recompute the public-main evidence before applying correctness fixes."""
import csv
import hashlib
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path
import numpy as np
from experiments.run_bandits import OnlineExperimentConfig, build_policies, load_records_by_project
from evaluation.online_replay import run_online_experiment
from evaluation.statistical import BootstrapConfig, bootstrap_ci, paired_bootstrap_pvalue
from policies.base import FeatureEncoder
from rewards.cost_model import CostConfig

ROOT = Path(__file__).resolve().parents[1]
out = {"revision": "a437abf7ecd51e1e9da50834c72b56628e33c697"}
for name, path in [("real", ROOT / "experiments/results/headline/github_actions_real.csv"),
                   ("smoke", Path("/private/tmp/arc-submission-audit/travistorrent_smoke.csv"))]:
    cfg = OnlineExperimentConfig(name, path, min_builds=1, min_history_days=0)
    groups = load_records_by_project(cfg)
    rows = list(csv.DictReader(path.open()))
    info = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "rows": len(rows),
            "identical_duplicate_rows": len(rows)-len({tuple(r.items()) for r in rows}), "projects": {}}
    for project, records in groups.items():
        x = np.stack([FeatureEncoder().encode(r.context) for r in records])
        overlap_steps = 0
        for i, current in enumerate(records):
            if any(prev.started_at < current.started_at < prev.finished_at
                   for prev in records[:i] if prev.started_at and prev.finished_at):
                overlap_steps += 1
        info["projects"][project] = dict(
            n=len(records), unique_commits=len({r.context.commit_sha for r in records}),
            outcomes=dict(Counter(r.outcome.value for r in records)),
            overlapping_decisions=overlap_steps, feature_rank=int(np.linalg.matrix_rank(x)),
            variable_dimensions=np.flatnonzero(np.ptp(x, axis=0)).tolist(),
            first=str(records[0].started_at), last=str(records[-1].started_at))
    results = run_online_experiment(build_policies(cfg, 0), groups, cost_config=CostConfig(), rng=np.random.default_rng(0))
    info["seed0"] = {pid: [asdict(r) for r in rs] for pid,rs in results.items()}
    out[name] = info
files = [ROOT / f"experiments/results/headline/real_github_actions/seed{i}__online_summary.json" for i in range(30)]
costs = [json.loads(f.read_text())["policies"]["thompson"]["cumulative_cost"] for f in files]
bc = BootstrapConfig(seed=42)
out["real_thompson"] = dict(costs=costs, mean=float(np.mean(costs)), sample_sd=float(np.std(costs,ddof=1)),
    ci=bootstrap_ci(costs,bc), p_vs_linucb=paired_bootstrap_pvalue(np.array(costs),np.full(30,669.5),bc),
    p_vs_static=paired_bootstrap_pvalue(np.array(costs),np.full(30,644.5),bc))
(ROOT / "review/evidence/original-audit.json").write_text(json.dumps(out,indent=2))
print(json.dumps({k: ({kk:vv for kk,vv in v.items() if kk != "seed0"} if isinstance(v,dict) else v) for k,v in out.items()},indent=2))
