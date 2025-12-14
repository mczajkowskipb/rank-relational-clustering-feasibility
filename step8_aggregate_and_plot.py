# step8_aggregate_and_plot.py
# Aggregation + boxplots + LaTeX table for step7 perturbation runs.
#
# Input:
#   results/step7_perturb_runs.csv
#
# Outputs:
#   results/step8_summary.csv
#   results/fig_step8_<dataset>.png   (one per dataset)
#   results/step8_table.tex
#
# Run:
#   python step8_aggregate_and_plot.py

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# CONFIG (common section)
# ------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")       # unused, kept for convention
RESULTS_DIR = os.path.join(BASE_DIR, "results")
SEED = 123  # unused here, kept for convention

DATASETS = ["golub", "colon", "DLBCL"]
SCENARIOS = ["baseline", "dropout", "noise", "scaling"]
METHODS = ["kmeans_euclid", "kmedoids_footrule", "kmedoids_kendall"]

IN_CSV = os.path.join(RESULTS_DIR, "step7_perturb_runs.csv")
OUT_SUMMARY = os.path.join(RESULTS_DIR, "step8_summary.csv")
OUT_TEX = os.path.join(RESULTS_DIR, "step8_table.tex")

os.makedirs(RESULTS_DIR, exist_ok=True)


# ------------------------------------------------------------
# helpers
# ------------------------------------------------------------
def q25(x): return x.quantile(0.25)
def q75(x): return x.quantile(0.75)

def fmt_cell(med, q1, q3):
    if pd.isna(med) or pd.isna(q1) or pd.isna(q3):
        return "--"
    return f"{med:.3f} [{q1:.3f}, {q3:.3f}]"

def scenario_order_key(s):
    try:
        return SCENARIOS.index(s)
    except ValueError:
        return 999

def method_order_key(m):
    try:
        return METHODS.index(m)
    except ValueError:
        return 999


# ------------------------------------------------------------
# main
# ------------------------------------------------------------
if not os.path.exists(IN_CSV):
    raise FileNotFoundError(f"Missing input: {IN_CSV}")

df = pd.read_csv(IN_CSV)

# basic column sanity
required_cols = {
    "dataset", "scenario", "rep", "method", "k", "ARI",
    "time_total_s", "time_dist_s", "time_cluster_s",
}
missing = required_cols - set(df.columns)
if missing:
    raise ValueError(f"Input CSV missing columns: {sorted(missing)}")

# enforce ordering categories (keeps script robust to extra values)
df["scenario"] = df["scenario"].astype(str)
df["method"] = df["method"].astype(str)
df["dataset"] = df["dataset"].astype(str)

# ------------------------------------------------------------
# 1) Summary table
# ------------------------------------------------------------
g = df.groupby(["dataset", "scenario", "method"], dropna=False)

summary = g.agg(
    n_runs=("ARI", "count"),
    ARI_median=("ARI", "median"),
    ARI_q25=("ARI", q25),
    ARI_q75=("ARI", q75),
    ARI_mean=("ARI", "mean"),
    ARI_std=("ARI", "std"),
    time_total_median_s=("time_total_s", "median"),
).reset_index()

# sort
summary = summary.sort_values(
    by=["dataset", "scenario", "method"],
    key=lambda s: s.map(scenario_order_key) if s.name == "scenario" else (
        s.map(method_order_key) if s.name == "method" else s
    ),
)

summary.to_csv(OUT_SUMMARY, index=False)

# ------------------------------------------------------------
# 2) Boxplots: per dataset -> one png
# ------------------------------------------------------------
saved_figs = []
method_styles = {
    "kmeans_euclid": "-",
    "kmedoids_footrule": "--",
    "kmedoids_kendall": ":",
}

for ds in DATASETS:
    dsub = df[df["dataset"] == ds].copy()
    if dsub.empty:
        continue

    base_positions = np.arange(len(SCENARIOS))  # 0..3
    offsets = np.array([-0.25, 0.0, 0.25])      # three methods
    width = 0.22

    fig, ax = plt.subplots(figsize=(10, 4))

    any_plotted = False
    for mi, m in enumerate(METHODS):
        data_per_scenario = []
        pos = []
        for si, sc in enumerate(SCENARIOS):
            vals = dsub[(dsub["scenario"] == sc) & (dsub["method"] == m)]["ARI"].dropna().values
            if len(vals) == 0:
                continue
            data_per_scenario.append(vals)
            pos.append(base_positions[si] + offsets[mi])

        if len(data_per_scenario) == 0:
            continue

        bp = ax.boxplot(
            data_per_scenario,
            positions=pos,
            widths=width,
            patch_artist=False,
            manage_ticks=False,
            showfliers=False,
        )

        # style box/whiskers/caps/medians for this method
        ls = method_styles.get(m, "-")
        for artist in bp.get("boxes", []):
            artist.set_linestyle(ls)
        for artist in bp.get("whiskers", []):
            artist.set_linestyle(ls)
        for artist in bp.get("caps", []):
            artist.set_linestyle(ls)
        for artist in bp.get("medians", []):
            artist.set_linestyle(ls)

        any_plotted = True

    if not any_plotted:
        plt.close(fig)
        continue

    ax.set_title(f"ARI robustness under perturbations — {ds}")
    ax.set_xlabel("scenario")
    ax.set_ylabel("ARI")
    ax.set_xticks(base_positions)
    ax.set_xticklabels(SCENARIOS)

    # legend with distinct linestyles (no colors)
    import matplotlib.lines as mlines
    handles = [
        mlines.Line2D([], [], color="black", linestyle=method_styles.get(m, "-"), label=m)
        for m in METHODS
    ]
    ax.legend(handles=handles, title="method", loc="best", frameon=True)

    ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.6)

    out_png = os.path.join(RESULTS_DIR, f"fig_step8_{ds}.png")
    out_svg = os.path.join(RESULTS_DIR, f"fig_step8_{ds}.svg")

    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    fig.savefig(out_svg)  # SVG (wektor)
    plt.close(fig)

    saved_figs.append(out_png)
    saved_figs.append(out_svg)

