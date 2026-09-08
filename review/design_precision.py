"""Analytic planning sensitivity, not policy experiments or fitted effect sizes."""
import csv
import json
import math
from pathlib import Path
from statistics import NormalDist


def required_clusters(sd, target, alpha=.05, power=None, comparisons=1):
    if sd <= 0 or target <= 0 or not 0 < alpha < 1 or comparisons < 1:
        raise ValueError("positive SD/target and valid alpha/comparisons required")
    z = NormalDist().inv_cdf(1-alpha/(2*comparisons))
    if power is not None:
        if not .5 < power < 1:
            raise ValueError("power must be between .5 and 1")
        z += NormalDist().inv_cdf(power)
    return max(2, math.ceil((z*sd/target)**2))


def build():
    root = Path(__file__).resolve().parent / 'design'
    rows = []
    for sd in (.025, .05, .10, .20):
        for target in (.01, .025, .05):
            rows.append(dict(project_difference_sd=sd, target=target,
                projects_ci_halfwidth_95=required_clusters(sd,target),
                projects_power80_seven_contrasts=required_clusters(sd,target,power=.8,comparisons=7)))
    path = root / 'precision-sensitivity.csv'
    with path.open('w', newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    label_counts = [{"decisions":n,"failure_probability":p,"expected_failures":n*p}
                    for n in (250,500,1000,2000) for p in (.02,.05,.10,.20)]
    (root/'precision-assumptions.json').write_text(json.dumps({
        "status":"illustrative_normal_approximation_not_a_sample_size_guarantee",
        "units":"project paired difference in normalized cost or Brier score",
        "normality_and_independent_project_assumptions":True,
        "confirmatory_contrasts":7,"family_alpha":.05,"planning_power":.8,
        "real_effect_sizes_estimated":False,"policy_runs_executed":False,
        "known_sigma_assumed":True,
        "small_sample_correction":"Required using justified variance bounds or external evidence for initial approval; development-only refinement after authorization and before evaluation access. These are optimistic lower planning approximations.",
        "expected_binary_events":label_counts},indent=2)+'\n')
    return rows


if __name__ == '__main__':
    for row in build():print(row)
