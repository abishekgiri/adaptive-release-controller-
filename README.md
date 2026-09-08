# Adaptive release controller research audit

This repository studies simulated deploy, canary, and block decisions with asymmetric costs. It is a research prototype, not a validated production release controller.

The September 2026 submission audit found temporal feature leakage, a defective Page-Hinkley implementation, inconsistent delayed/censored evaluation, and missing simple cost-aware baselines. The corrected results withdraw the original general operating-envelope claim. See [the full technical review](review/submission-review.md), [the corrected manuscript](paper/adaptive-deployment-control.md), and [the numerical source of truth](paper/source-of-truth.md).

## Corrected results

| Setting | Static rules | LinUCB | Simple control |
|---|---:|---:|---:|
| Synthetic fixture | 1884 | 1863 | Bayesian rate 1713.5 |
| High failure cost | 2578 | 1972 | Always block 1865 |
| Low safe-block cost | 1591 | 1072 | Always block 1005 |
| GitHub Actions proxy | 640 | 779 | Rolling cost rule 624.5 |

Real-data Thompson Sampling costs 665.7 ± 39.8 across seeds 0-29; all policies are evaluated on the same 578 resolved labels among 600 workflow rows. These are simulated CI-proxy costs, not measured deployment savings. Repeated algorithm seeds do not estimate uncertainty across repositories.

## Reproduce

Run from the repository root with Python 3.13:

```bash
python3.13 -m venv .venv
.venv/bin/python -m pip install -r requirements-audit.txt
.venv/bin/python -m pytest
.venv/bin/python -m experiments.reproduce_submission
.venv/bin/python paper/build_corrected.py
bash paper/build_documents.sh
```

The last command also needs Pandoc and a TeX installation providing `pdflatex` on PATH. The document sources compile as a readable audit revision; venue-specific submission formatting is not claimed complete. The existing Word export is generated from the same Markdown source using Pandoc.

The reproduction runs 30 seeds for main replay, cost robustness, artificial delays, five cost matrices, and two duplicate/commit sensitivity cohorts; the deterministic ablation uses seed 0. Separate synthetic drift evaluation uses 30 environment seeds, with independent stationary seeds 1000-1029 for the detector calibration screen. Every seed's summary, project action trace, and figure input is retained under `experiments/results/headline/corrected/`. `manifest.json` records code/data hashes and the tested environment. Original results and manuscript formats are archived under `review/original/` and are superseded.

Frozen data are in `data/fixtures/`; no ignored `data/raw/` file is required. Read [fixture provenance](data/fixtures/README.md): the synthetic generator and complete original GitHub collection manifest remain unavailable. The real export lacks workflow/run identifiers and contains repeated SHAs. Numerical reproduction does not solve those validity limitations.

## Validated path and limitations

The active experiment path uses `data/loaders.py`, `policies/linucb.py`, `policies/thompson.py`, `policies/cost_rules.py`, `delayed/buffer.py`, `evaluation/online_replay.py`, and the named experiment runners. Historical CI features and labels obey the chosen observation clock. Costs exclude the same unknown-label rows for all actions. Page-Hinkley follows a documented upward reference recurrence; its threshold still needs scenario-specific calibration.

Several legacy modules remain placeholders, including the offline classifier, epsilon-greedy policy, replay-environment abstraction, standalone delay sampler and imputation strategies. SNIPS/DR and production propensity logging are not implemented. The CI data cannot support causal off-policy evaluation of unlogged deployment actions. Those paths are not used as evidence for the corrected paper.

The review's readiness verdict remains **NO — significant research work remains**. The corrected contribution is a bounded simulation study and a baseline-selection finding, not proof that contextual bandits improve production releases.
