"""Reproducible matplotlib figure scripts for the paper.

Each public function generates one paper figure, saves PDF + PNG to
paper/figures/, and returns the output paths. No interactive dependencies.

Palette and style are centralised in _STYLE so all figures are consistent.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ---------------------------------------------------------------------------
# Shared style
# ---------------------------------------------------------------------------

FIGURES_DIR = Path(__file__).parent.parent / "paper" / "figures"
RESULTS_DIR = Path(__file__).parent.parent / "experiments" / "results"

_PALETTE: dict[str, str] = {
    "static_rules":               "#2c7bb6",
    "linucb":                     "#1a9641",
    "thompson":                   "#d7191c",
    "heuristic_score":            "#fdae61",
    "linucb_with_drift":          "#7b2d8b",
    "linucb_with_drift_full":     "#7b2d8b",
    "linucb_with_drift_no_reset": "#b2abd2",
    "cost_sensitive_bandit":      "#7b2d8b",
    "full":                       "#7b2d8b",
    "no_drift":                   "#1a9641",
    "no_cost":                    "#d7191c",
    "no_delay":                   "#fdae61",
}

_LABELS: dict[str, str] = {
    "static_rules":               "Static rules",
    "linucb":                     "LinUCB",
    "thompson":                   "Thompson",
    "heuristic_score":            "Heuristic score",
    "linucb_with_drift":          "LinUCB+Drift",
    "linucb_with_drift_full":     "LinUCB+Drift (full)",
    "linucb_with_drift_no_reset": "LinUCB+Drift (no reset)",
    "cost_sensitive_bandit":      "LinUCB+Drift",
    "full":                       "Full (cost+delay+drift)",
    "no_drift":                   "No drift reset",
    "no_cost":                    "No cost weighting",
    "no_delay":                   "No delay buffer",
}

_STYLE = {
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
}


def _save(fig: plt.Figure, name: str) -> tuple[Path, Path]:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    pdf = FIGURES_DIR / f"{name}.pdf"
    png = FIGURES_DIR / f"{name}.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return pdf, png


def _color(pid: str) -> str:
    return _PALETTE.get(pid, "#888888")


def _label(pid: str) -> str:
    return _LABELS.get(pid, pid)


# ---------------------------------------------------------------------------
# Figure 1 — Cumulative cost bar chart (synthetic)
# ---------------------------------------------------------------------------

def fig_cumulative_cost_synthetic() -> tuple[Path, Path]:
    """Bar chart of final cumulative cost per policy on synthetic dataset."""
    policies_order = ["static_rules", "linucb", "thompson", "heuristic_score"]
    costs_by_policy: dict[str, list[float]] = {p: [] for p in policies_order}
    for seed in range(30):
        d = json.loads((RESULTS_DIR / f"online_smoke/{seed}/online_summary.json").read_text())
        for pid in policies_order:
            costs_by_policy[pid].append(d["policies"][pid]["cumulative_cost"])
    means = {p: float(np.mean(costs_by_policy[p])) for p in policies_order}
    stds = {p: float(np.std(costs_by_policy[p], ddof=1)) for p in policies_order}

    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(5, 3.5))
        x = np.arange(len(policies_order))
        bars = ax.bar(x, [means[p] for p in policies_order],
                      yerr=[stds[p] for p in policies_order],
                      color=[_color(p) for p in policies_order],
                      capsize=4, width=0.6, error_kw={"elinewidth": 1})
        ax.set_xticks(x)
        ax.set_xticklabels([_label(p) for p in policies_order], rotation=15, ha="right")
        ax.set_ylabel("Cumulative cost (1 150 steps)")
        ax.set_title("Synthetic dataset — cumulative cost (n=30 seeds)")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
        for rect, pid in zip(bars, policies_order):
            ax.text(rect.get_x() + rect.get_width() / 2,
                    rect.get_height() + stds[pid] + 30,
                    f"{means[pid]:.0f}", ha="center", va="bottom", fontsize=8)
        fig.tight_layout()
    return _save(fig, "fig_cumulative_cost_synthetic")


# ---------------------------------------------------------------------------
# Figure 2 — Cumulative cost bar chart (real data)
# ---------------------------------------------------------------------------

def fig_cumulative_cost_real() -> tuple[Path, Path]:
    """Bar chart of final cumulative cost per policy on real GitHub Actions dataset."""
    policies_order = ["static_rules", "linucb", "thompson", "heuristic_score"]
    costs_by_policy: dict[str, list[float]] = {p: [] for p in policies_order}
    for seed in range(30):
        d = json.loads((RESULTS_DIR / f"real_github_actions/{seed}/online_summary.json").read_text())
        pol = d["policies"]
        for pid in policies_order:
            actual = "cost_sensitive_bandit" if (pid == "linucb_with_drift" and "cost_sensitive_bandit" in pol) else pid
            if actual in pol:
                costs_by_policy[pid].append(pol[actual]["cumulative_cost"])
    valid = [p for p in policies_order if costs_by_policy[p]]
    means = {p: float(np.mean(costs_by_policy[p])) for p in valid}
    stds = {p: float(np.std(costs_by_policy[p], ddof=1)) for p in valid}

    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(5, 3.5))
        x = np.arange(len(valid))
        bars = ax.bar(x, [means[p] for p in valid],
                      yerr=[stds[p] for p in valid],
                      color=[_color(p) for p in valid],
                      capsize=4, width=0.6, error_kw={"elinewidth": 1})
        ax.set_xticks(x)
        ax.set_xticklabels([_label(p) for p in valid], rotation=15, ha="right")
        ax.set_ylabel("Cumulative cost (600 steps)")
        ax.set_title("Real GitHub Actions — cumulative cost (n=30 seeds)")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
        for rect, pid in zip(bars, valid):
            ax.text(rect.get_x() + rect.get_width() / 2,
                    rect.get_height() + stds[pid] + 8,
                    f"{means[pid]:.0f}", ha="center", va="bottom", fontsize=8)
        fig.tight_layout()
    return _save(fig, "fig_cumulative_cost_real")


# ---------------------------------------------------------------------------
# Figure 3 — Action distribution stacked bars
# ---------------------------------------------------------------------------

def fig_action_distribution() -> tuple[Path, Path]:
    """Stacked bar: deploy/canary/block fractions per policy, synthetic + real side-by-side."""
    actions = ["deploy", "canary", "block"]
    action_colors = {"deploy": "#4dac26", "canary": "#f1b6da", "block": "#d01c8b"}
    datasets = [
        ("Synthetic (1 150 steps)", "online_smoke", 30,
         ["static_rules", "linucb", "thompson", "heuristic_score"]),
        ("Real GH-Actions (600 steps)", "real_github_actions", 30,
         ["static_rules", "linucb", "thompson", "heuristic_score"]),
    ]

    with plt.rc_context(_STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
        for ax, (title, config_name, n_seeds, policies_order) in zip(axes, datasets):
            fracs: dict[str, dict[str, float]] = {p: {a: 0.0 for a in actions} for p in policies_order}
            for seed in range(n_seeds):
                path = RESULTS_DIR / f"{config_name}/{seed}/online_summary.json"
                pol = json.loads(path.read_text())["policies"]
                for pid in policies_order:
                    actual = "cost_sensitive_bandit" if (pid == "linucb_with_drift" and "cost_sensitive_bandit" in pol) else pid
                    if actual in pol:
                        af = pol[actual]["action_fractions"]
                        for a in actions:
                            fracs[pid][a] += af.get(a, 0.0) / n_seeds
            x = np.arange(len(policies_order))
            bottoms = np.zeros(len(policies_order))
            for a in actions:
                vals = np.array([fracs[p][a] for p in policies_order])
                ax.bar(x, vals, bottom=bottoms, color=action_colors[a], label=a.capitalize(), width=0.6)
                bottoms += vals
            ax.set_xticks(x)
            ax.set_xticklabels([_label(p) for p in policies_order], rotation=20, ha="right")
            ax.set_ylabel("Fraction of decisions")
            ax.set_ylim(0, 1.05)
            ax.set_title(title)

        handles, labels_seen = [], {}
        for h, l in zip(*axes[0].get_legend_handles_labels()):
            if l not in labels_seen:
                labels_seen[l] = True
                handles.append(h)
        fig.legend(handles, list(labels_seen), loc="upper right", ncol=1, frameon=False)
        fig.suptitle("Action distributions per policy", y=1.02)
        fig.tight_layout()
    return _save(fig, "fig_action_distribution")


# ---------------------------------------------------------------------------
# Figure 4 — Cost-sweep monotonicity
# ---------------------------------------------------------------------------

def fig_cost_sweep() -> tuple[Path, Path]:
    """Line chart: mean cost vs cost ratio for LinUCB, static, Thompson."""
    cs = json.loads((RESULTS_DIR / "cost_sweep/cost_sweep_summary.json").read_text())
    levels_order = ["5:1", "10:1", "20:1", "40:1", "100:1"]
    ratios = [5, 10, 20, 40, 100]
    policies_to_plot = ["static_rules", "linucb", "thompson"]

    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(5.5, 3.5))
        for pid in policies_to_plot:
            vals, lo, hi = [], [], []
            for lbl in levels_order:
                p = cs["levels"][lbl]["policies"][pid]
                vals.append(p["mean_cost"]); lo.append(p["ci_lo_95"]); hi.append(p["ci_hi_95"])
            vals, lo, hi = np.array(vals), np.array(lo), np.array(hi)
            ax.plot(ratios, vals, marker="o", label=_label(pid), color=_color(pid))
            ax.fill_between(ratios, lo, hi, alpha=0.15, color=_color(pid))
        ax.set_xscale("log")
        ax.set_xticks(ratios)
        ax.set_xticklabels([f"{r}:1" for r in ratios])
        ax.set_xlabel("deploy_failure / block_bad ratio")
        ax.set_ylabel("Mean cumulative cost")
        ax.set_title("Cost-ratio sensitivity sweep (n=30 seeds)")
        ax.legend(frameon=False)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
        fig.tight_layout()
    return _save(fig, "fig_cost_sweep")


# ---------------------------------------------------------------------------
# Figure 5 — Drift mode comparison bars
# ---------------------------------------------------------------------------

def fig_drift_mode_bars() -> tuple[Path, Path]:
    """Grouped bar: mean cost per drift mode × policy subset."""
    dr = json.loads((RESULTS_DIR / "drift_eval/drift_eval_summary.json").read_text())
    modes = ["none", "abrupt", "gradual"]
    mode_labels = {"none": "Stationary", "abrupt": "Abrupt drift", "gradual": "Gradual drift"}
    policies_to_plot = ["static_rules", "linucb", "linucb_with_drift_full"]
    width = 0.22
    x = np.arange(len(modes))

    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(6, 3.5))
        for i, pid in enumerate(policies_to_plot):
            vals = [dr["conditions"][m]["policies"][pid]["mean_cost"] for m in modes]
            offset = (i - len(policies_to_plot) / 2 + 0.5) * width
            ax.bar(x + offset, vals, width=width, color=_color(pid), label=_label(pid))
        ax.set_xticks(x)
        ax.set_xticklabels([mode_labels[m] for m in modes])
        ax.set_ylabel("Mean cumulative cost (n=30 seeds, horizon=500)")
        ax.set_title("Drift mode evaluation")
        ax.legend(frameon=False, loc="upper left")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
        fig.tight_layout()
    return _save(fig, "fig_drift_mode_bars")


# ---------------------------------------------------------------------------
# Figure 6 — Ablation bar chart
# ---------------------------------------------------------------------------

def fig_ablation_bars() -> tuple[Path, Path]:
    """Bar chart of cumulative cost per ablation variant (seed 0)."""
    ab = json.loads((RESULTS_DIR / "ablation_smoke/0/ablation_summary.json").read_text())
    variants = ["no_delay", "no_drift", "full", "no_cost"]
    labels_map = {
        "full": "Full model", "no_delay": "No delay buffer",
        "no_cost": "No cost weighting", "no_drift": "No drift reset",
    }
    costs = [ab["policies"][v]["cumulative_cost"] for v in variants]

    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(5, 3.5))
        bars = ax.bar(range(len(variants)), costs,
                      color=[_color(v) for v in variants], width=0.6)
        ax.set_xticks(range(len(variants)))
        ax.set_xticklabels([labels_map[v] for v in variants], rotation=15, ha="right")
        ax.set_ylabel("Cumulative cost (1 150 steps, seed 0)")
        ax.set_title("Ablation study — component contributions")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
        for rect, cost in zip(bars, costs):
            ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + 20,
                    f"{cost:.0f}", ha="center", va="bottom", fontsize=8)
        fig.tight_layout()
    return _save(fig, "fig_ablation_bars")


# ---------------------------------------------------------------------------
# Figure 7 — Thompson seed distribution (box plots)
# ---------------------------------------------------------------------------

def fig_thompson_seed_distribution() -> tuple[Path, Path]:
    """Box plots of 30-seed cost distributions for Thompson and LinUCB."""
    smoke_t, smoke_l, real_t, real_l = [], [], [], []
    for seed in range(30):
        ds = json.loads((RESULTS_DIR / f"online_smoke/{seed}/online_summary.json").read_text())["policies"]
        smoke_t.append(ds["thompson"]["cumulative_cost"])
        smoke_l.append(ds["linucb"]["cumulative_cost"])
        dr = json.loads((RESULTS_DIR / f"real_github_actions/{seed}/online_summary.json").read_text())["policies"]
        real_t.append(dr["thompson"]["cumulative_cost"])
        real_l.append(dr["linucb"]["cumulative_cost"])

    with plt.rc_context(_STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(7, 3.5))
        for ax, t_data, l_data, title, ylabel in [
            (axes[0], smoke_t, smoke_l, "Synthetic dataset", "Cumulative cost (1 150 steps)"),
            (axes[1], real_t, real_l, "Real GH-Actions", "Cumulative cost (600 steps)"),
        ]:
            bp = ax.boxplot([t_data, l_data], labels=["Thompson", "LinUCB"],
                            patch_artist=True, notch=False, widths=0.5,
                            medianprops={"color": "black", "linewidth": 1.5})
            bp["boxes"][0].set_facecolor(_color("thompson"))
            bp["boxes"][1].set_facecolor(_color("linucb"))
            for elem in ["whiskers", "caps"]:
                for item in bp[elem]:
                    item.set_color("gray")
            ax.set_ylabel(ylabel)
            ax.set_title(title)
        fig.suptitle("Per-seed cost distribution (n=30 seeds)", y=1.02)
        fig.tight_layout()
    return _save(fig, "fig_thompson_seed_distribution")


# ---------------------------------------------------------------------------
# Figure 8 — Cumulative cost curves (per-step mean across seeds)
# ---------------------------------------------------------------------------

def fig_cumulative_cost_curves() -> tuple[Path, Path]:
    """Cumulative cost over steps, mean across 30 seeds, synthetic + real."""
    policies_to_plot = ["static_rules", "linucb", "thompson"]

    def _load_mean_cumcost(config_name: str, n_seeds: int) -> dict[str, np.ndarray]:
        arrays: dict[str, list[np.ndarray]] = {p: [] for p in policies_to_plot}
        for seed in range(n_seeds):
            for pid in policies_to_plot:
                path = RESULTS_DIR / f"{config_name}/{seed}/step_costs_{pid}.npy"
                if path.exists():
                    arrays[pid].append(np.load(path).astype(np.float64))
        result = {}
        for pid in policies_to_plot:
            if arrays[pid]:
                min_len = min(len(a) for a in arrays[pid])
                mat = np.stack([a[:min_len] for a in arrays[pid]])
                result[pid] = mat.mean(axis=0).cumsum()
        return result

    smoke = _load_mean_cumcost("online_smoke", 30)
    real = _load_mean_cumcost("real_github_actions", 30)

    with plt.rc_context(_STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
        for ax, data, title in [
            (axes[0], smoke, "Synthetic dataset (1 150 steps)"),
            (axes[1], real, "Real GH-Actions (600 steps)"),
        ]:
            for pid in policies_to_plot:
                if pid in data:
                    ax.plot(data[pid], label=_label(pid), color=_color(pid))
            ax.set_xlabel("Step")
            ax.set_ylabel("Cumulative cost")
            ax.set_title(title)
            ax.legend(frameon=False)
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
        fig.suptitle("Cumulative cost over time — mean across n=30 seeds", y=1.02)
        fig.tight_layout()
    return _save(fig, "fig_cumulative_cost_curves")


# ---------------------------------------------------------------------------
# Figure 9 — Drift recovery curves (cumulative regret per step)
# ---------------------------------------------------------------------------

def fig_drift_recovery_curves() -> tuple[Path, Path]:
    """Cumulative regret vs oracle per reveal-step, linucb vs linucb_with_drift_full."""
    modes = ["none", "abrupt", "gradual"]
    mode_labels = {
        "none": "Stationary", "abrupt": "Abrupt drift\n(midpoint)", "gradual": "Gradual drift\n(25 seg.)"
    }
    pids = ["linucb", "linucb_with_drift_full"]

    with plt.rc_context(_STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))
        for ax, mode in zip(axes, modes):
            for pid in pids:
                path = RESULTS_DIR / f"drift_eval/step_regrets_{mode}_{pid}.npy"
                if path.exists():
                    arr = np.load(path).astype(np.float64)
                    ax.plot(arr.cumsum(), label=_label(pid), color=_color(pid))
            if mode == "abrupt":
                ax.axvline(250, color="gray", linestyle="--", linewidth=0.8, label="Drift point")
            ax.set_xlabel("Reward-reveal step")
            ax.set_ylabel("Cumulative regret")
            ax.set_title(mode_labels[mode])
            ax.legend(frameon=False, fontsize=8)
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
        fig.suptitle("Drift recovery — cumulative regret vs oracle\n"
                     "(Synthetic, horizon=500, n=30 seeds mean)", y=1.04)
        fig.tight_layout()
    return _save(fig, "fig_drift_recovery_curves")


# ---------------------------------------------------------------------------
# Figure 10 — Per-step cost CDF
# ---------------------------------------------------------------------------

def fig_cost_cdf_per_step() -> tuple[Path, Path]:
    """Empirical CDF of per-step costs pooled across 30 seeds."""
    policies_to_plot = ["static_rules", "linucb", "thompson"]

    def _pooled(config_name: str, n_seeds: int) -> dict[str, np.ndarray]:
        arrays: dict[str, list[np.ndarray]] = {p: [] for p in policies_to_plot}
        for seed in range(n_seeds):
            for pid in policies_to_plot:
                path = RESULTS_DIR / f"{config_name}/{seed}/step_costs_{pid}.npy"
                if path.exists():
                    arrays[pid].append(np.load(path).astype(np.float64))
        return {pid: np.concatenate(arrs) for pid, arrs in arrays.items() if arrs}

    smoke = _pooled("online_smoke", 30)
    real = _pooled("real_github_actions", 30)

    with plt.rc_context(_STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
        for ax, costs_dict, title in [
            (axes[0], smoke, "Synthetic dataset"),
            (axes[1], real, "Real GH-Actions"),
        ]:
            for pid in policies_to_plot:
                if pid not in costs_dict:
                    continue
                data = np.sort(costs_dict[pid])
                data = data[np.isfinite(data)]
                cdf = np.arange(1, len(data) + 1) / len(data)
                ax.plot(data, cdf, label=_label(pid), color=_color(pid))
            ax.set_xlabel("Per-step cost")
            ax.set_ylabel("Cumulative fraction of steps")
            ax.set_title(title)
            ax.legend(frameon=False)
            ax.set_ylim(0, 1.05)
        fig.suptitle("Per-step cost CDF — pooled across 30 seeds × all steps", y=1.02)
        fig.tight_layout()
    return _save(fig, "fig_cost_cdf_per_step")


# ---------------------------------------------------------------------------
# Run all
# ---------------------------------------------------------------------------

FIGURE_FUNCTIONS = [
    fig_cumulative_cost_synthetic,
    fig_cumulative_cost_real,
    fig_action_distribution,
    fig_cost_sweep,
    fig_drift_mode_bars,
    fig_ablation_bars,
    fig_thompson_seed_distribution,
    fig_cumulative_cost_curves,
    fig_drift_recovery_curves,
    fig_cost_cdf_per_step,
]


def generate_all() -> list[Path]:
    """Generate all paper figures. Returns list of saved PDF paths."""
    saved = []
    for fn in FIGURE_FUNCTIONS:
        print(f"  {fn.__name__} …", flush=True)
        pdf, _ = fn()
        saved.append(pdf)
        print(f"    → {pdf.name}")
    return saved


if __name__ == "__main__":
    print(f"Generating {len(FIGURE_FUNCTIONS)} figures …")
    paths = generate_all()
    print(f"\nDone. {len(paths)} PDFs in {FIGURES_DIR}")
