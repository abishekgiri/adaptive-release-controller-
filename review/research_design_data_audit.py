"""Descriptive data audit only: no policy fitting, seeds, or experiment execution.

Run from repository root: python -m review.research_design_data_audit
"""
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
import csv
import hashlib
import json

import numpy as np

from data.loaders import TravisTorrentLoader
from policies.base import FeatureEncoder


def describe(values):
    x = np.asarray(values, dtype=float)
    return {"unique": len(set(values)), "min": float(x.min()),
            "max": float(x.max()), "zero_count": int((x == 0).sum()),
            "median": float(np.median(x))}


def audit(path):
    rows = list(csv.DictReader(path.open()))
    columns = list(rows[0]) if rows else []
    raw = {k: {"missing": sum(not r[k].strip() for r in rows),
               "unique_nonempty": len({r[k] for r in rows if r[k].strip()})}
           for k in columns}
    records = list(TravisTorrentLoader(path, min_builds=0, min_history_days=0,
                                     timing_mode="event_time"))
    by_project = defaultdict(list)
    raw_project = defaultdict(list)
    for r in records:
        by_project[r.context.project_slug].append(r)
    for r in rows:
        raw_project[r["gh_project_name"]].append(r)
    projects = {}
    for slug, recs in by_project.items():
        source = raw_project[slug]
        features = [asdict(r.context) for r in recs]
        names = [k for k in features[0] if k not in {"commit_sha", "project_slug", "step"}]
        matrix = np.vstack([FeatureEncoder().encode(r.context) for r in recs])
        labels = Counter(r.outcome.value for r in recs)
        by_sha = defaultdict(list)
        for row in source:
            by_sha[row["git_trigger_commit"]].append(row)
        inconsistent = {k: sum(len({r.get(k, '') for r in group}) > 1
                               for group in by_sha.values())
                        for k in ["tr_status", "gh_is_pr", "git_author_name"]}
        starts = [r.started_at for r in recs if r.started_at]
        before = []
        same_sha_prior_labels = 0
        missing_history = 0
        for r in recs:
            observed = [p for p in before if p.finished_at and r.started_at
                        and p.finished_at <= r.started_at
                        and p.outcome.value in {"success", "failure"}]
            if any(p.context.commit_sha == r.context.commit_sha for p in observed):
                same_sha_prior_labels += 1
            window = [p for p in observed if (r.started_at-p.finished_at).total_seconds() <= 7*86400]
            missing_history += not window
            before.append(r)
        projects[slug] = {
            "rows": len(source), "parsed_rows": len(recs), "labels": dict(labels),
            "resolved_failure_rate": labels["failure"]/(labels["failure"]+labels["success"]),
            "distinct_shas": len(by_sha),
            "duplicate_exact_rows": len(source)-len({tuple(r[k] for k in columns) for r in source}),
            "unique_authors": len({r.get("git_author_email") or r.get("git_author_name") for r in source}),
            "start_min": min(starts).isoformat() if starts else None,
            "start_max": max(starts).isoformat() if starts else None,
            "span_days": (max(starts)-min(starts)).total_seconds()/86400 if starts else None,
            "unique_start_times": len(set(starts)),
            "finish_before_start": sum(r.finished_at < r.started_at for r in recs if r.started_at and r.finished_at),
            "finish_equal_start": sum(r.finished_at == r.started_at for r in recs if r.started_at and r.finished_at),
            "same_sha_inconsistency_counts": inconsistent,
            "decisions_with_available_same_sha_resolved_label": same_sha_prior_labels,
            "decisions_with_empty_7day_resolved_history": missing_history,
            "encoded_rank": int(np.linalg.matrix_rank(matrix)),
            "context_features": {k: describe([r[k] for r in features]) for k in names},
            "raw_columns": {k: {"missing":sum(not r[k].strip() for r in source),
                                 "unique_nonempty":len({r[k] for r in source if r[k].strip()})}
                            for k in columns},
        }
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "rows": len(rows), "columns": raw, "projects": projects}


def main():
    paths = [Path("data/fixtures/github_actions_real.csv"),
             Path("data/fixtures/travistorrent_smoke.csv"),
             Path("data/raw/travistorrent.csv")]
    result = {str(p): audit(p) for p in paths if p.exists()}
    out = Path("review/evidence/research-design-data-audit.json")
    out.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
    for p, d in result.items():
        print(p, "rows", d["rows"])
        for name, group in d["projects"].items():
            print(name, {k: v for k, v in group.items() if k not in {"context_features", "raw_columns"}})
            print("features", {k:(v["unique"],v["min"],v["max"]) for k,v in group["context_features"].items()})


if __name__ == "__main__":
    main()
