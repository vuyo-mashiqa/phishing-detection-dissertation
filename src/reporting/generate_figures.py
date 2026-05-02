"""
generate_figures.py
===================
Generates all 4 publication-quality figures for the classical ML results.

Run from the project root:
    python src/reporting/generate_figures.py

Outputs (saved to docs/figures/classical_ml/):
    1. heatmap_cross_stratum.png
    2. bar_within_stratum_f1.png
    3. bar_fpr_operational.png
    4. scatter_offdiag_generalisation.png

Data is read directly from:
    outputs/results/classical_ml/within_stratum_results.csv
    outputs/results/classical_ml/cross_stratum_results.csv

If those CSVs are absent the script falls back to hard-coded confirmed values.
"""

import sys
from pathlib import Path
import numpy as np

try:
    import pandas as pd

    WS  = Path("outputs/results/classical_ml/within_stratum_results.csv")
    CS  = Path("outputs/results/classical_ml/cross_stratum_results.csv")

    if WS.exists() and CS.exists():
        ws_df = pd.read_csv(WS)
        cs_df = pd.read_csv(CS)

        MODELS = ["LR", "SVM", "RF", "XGB", "LGB"]
        STRATA = [1, 2, 3]

        within_f1 = {}
        within_fpr = {}
        for m in MODELS:
            row = ws_df[ws_df["model"] == m].sort_values("stratum")
            within_f1[m]  = row["f1_macro"].tolist()
            within_fpr[m] = row["fpr_at_95tpr"].tolist()

        cross_f1 = {}
        for m in MODELS:
            mat = [[0.0]*3 for _ in range(3)]
            sub = cs_df[cs_df["model"] == m]
            for _, r in sub.iterrows():
                i = int(r["train_stratum"]) - 1
                j = int(r["test_stratum"])  - 1
                mat[i][j] = float(r["f1_macro"])
            cross_f1[m] = mat

        print("Data loaded from CSV files.")
    else:
        raise FileNotFoundError("CSV files not found — using hard-coded values")

except Exception as e:
    print(f"Note: {e}")
    MODELS = ["LR", "SVM", "RF", "XGB", "LGB"]

    within_f1 = {
        "LR":  [0.9793, 0.9981, 0.9890],
        "SVM": [0.9948, 0.9991, 0.9900],
        "RF":  [0.9528, 0.9907, 0.9681],
        "XGB": [0.9905, 0.9944, 0.9855],
        "LGB": [0.9933, 0.9944, 0.9890],
    }
    within_fpr = {
        "LR":  [0.0000, 0.0000, 0.0030],
        "SVM": [0.0000, 0.0000, 0.0020],
        "RF":  [0.0018, 0.0024, 0.0202],
        "XGB": [0.0000, 0.0000, 0.0010],
        "LGB": [0.0000, 0.0024, 0.0000],
    }
    cross_f1 = {
        "LR":  [[0.9793, 0.4045, 0.5719],
                [0.0824, 0.9981, 0.5026],
                [0.7666, 0.7480, 0.9895]],
        "SVM": [[0.9948, 0.7422, 0.7099],
                [0.0953, 0.9991, 0.4887],
                [0.7466, 0.7713, 0.9900]],
        "RF":  [[0.9528, 0.2607, 0.3656],
                [0.2249, 0.9907, 0.5482],
                [0.5079, 0.6778, 0.9681]],
        "XGB": [[0.9905, 0.4229, 0.5898],
                [0.3205, 0.9944, 0.4883],
                [0.4277, 0.8076, 0.9855]],
        "LGB": [[0.9933, 0.4058, 0.5954],
                [0.2655, 0.9944, 0.5077],
                [0.4499, 0.8292, 0.9890]],
    }
    print("Using hard-coded confirmed values.")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as mpatches

OUT = Path("docs/figures/classical_ml")
OUT.mkdir(parents=True, exist_ok=True)

COLORS = ["#2166ac", "#4dac26", "#d7191c", "#ff7f00", "#984ea3"]
STYLE = {
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor":   "#cccccc", "axes.linewidth":  0.8,
    "font.family":      "DejaVu Sans",
    "axes.titlesize":   12, "axes.labelsize": 11,
    "xtick.labelsize":  10, "ytick.labelsize": 10,
    "legend.fontsize":  10, "legend.framealpha": 0.9,
    "legend.edgecolor": "#cccccc",
}
plt.rcParams.update(STYLE)

HAM_VOL  = 95_000
x        = np.arange(3)
w        = 0.15
offsets  = np.linspace(-2, 2, 5) * w
X_LABELS = ["S-I (Legacy)", "S-II (Contemp.)", "S-III (LLM-Gen.)"]

cmap_rg = LinearSegmentedColormap.from_list(
    "rg", ["#d73027","#fc8d59","#fee090","#d9ef8b","#91cf60","#1a9850"])

fig1, axes1 = plt.subplots(2, 3, figsize=(14, 8))
fig1.suptitle(
    "Cross-Stratum F1 Generalisation Matrix — Classical ML Baselines\n"
    "Rows = training stratum  |  Columns = test stratum  |  "
    "Diagonal \u25c6 = matched distribution",
    fontsize=12, fontweight="bold", y=1.01,
)
fig1.patch.set_facecolor("white")

train_lbl = ["Train I\n(Legacy)", "Train II\n(Contemp.)", "Train III\n(LLM)"]
test_lbl  = ["Test I\n(Legacy)",  "Test II\n(Contemp.)", "Test III\n(LLM)"]
positions = [(0,0),(0,1),(0,2),(1,0),(1,1)]

for idx, model in enumerate(MODELS):
    r, c = positions[idx]
    ax = axes1[r][c]
    z  = np.array(cross_f1[model])
    ax.imshow(z, cmap=cmap_rg, vmin=0, vmax=1, aspect="auto")
    ax.set_title(model, fontweight="bold", pad=6)
    ax.set_xticks([0,1,2]); ax.set_xticklabels(test_lbl,  fontsize=9)
    ax.set_yticks([0,1,2]); ax.set_yticklabels(train_lbl, fontsize=9)
    ax.set_xlabel("Test Stratum",  fontsize=9)
    ax.set_ylabel("Train Stratum", fontsize=9)
    ax.grid(False)
    for i in range(3):
        for j in range(3):
            v   = z[i, j]
            txt = "#ffffff" if v < 0.55 else "#000000"
            mrk = " \u25c6" if i == j else ""
            ax.text(j, i, f"{v:.3f}{mrk}", ha="center", va="center",
                    fontsize=10, color=txt,
                    fontweight="bold" if i == j else "normal")
    for k in range(3):
        ax.add_patch(plt.Rectangle(
            (k-0.5, k-0.5), 1, 1,
            fill=False, edgecolor="white", linewidth=2.5, zorder=5))

axes1[1][2].set_visible(False)
cbar_ax = fig1.add_axes([0.73, 0.10, 0.022, 0.36])
sm = plt.cm.ScalarMappable(cmap=cmap_rg, norm=plt.Normalize(0, 1))
cb = fig1.colorbar(sm, cax=cbar_ax)
cb.set_label("F1-Macro", fontsize=10)
cb.ax.tick_params(labelsize=9)

plt.tight_layout(rect=[0, 0, 1, 0.98])
p1 = OUT / "heatmap_cross_stratum.png"
fig1.savefig(p1, dpi=150, bbox_inches="tight", facecolor="white")
plt.close(fig1)
print(f"Saved: {p1}")

fig2, ax2 = plt.subplots(figsize=(10, 5.5))
fig2.patch.set_facecolor("white")
for i, (m, c) in enumerate(zip(MODELS, COLORS)):
    ax2.bar(x + offsets[i], within_f1[m], width=w*0.88,
            label=m, color=c, alpha=0.88, zorder=3)
ax2.set_xticks(x)
ax2.set_xticklabels(X_LABELS)
ax2.set_ylabel("F1-Macro")
ax2.set_ylim(0.90, 1.015)
ax2.set_title(
    "Within-Stratum Test F1 — Classical ML Baselines\n"
    "(y-axis starts at 0.90 to resolve inter-model differences)",
    fontweight="bold", pad=10)
ax2.legend(title="Model", loc="lower right", ncol=5)
ax2.yaxis.grid(True, color="#eeeeee", linewidth=0.6, zorder=0)
ax2.set_axisbelow(True)
ax2.spines[["top","right"]].set_visible(False)
plt.tight_layout()
p2 = OUT / "bar_within_stratum_f1.png"
fig2.savefig(p2, dpi=150, bbox_inches="tight", facecolor="white")
plt.close(fig2)
print(f"Saved: {p2}")

X_MAX = 2100
fig3, axes3 = plt.subplots(1, 3, figsize=(13, 4.8), sharey=True, sharex=False)
fig3.patch.set_facecolor("white")
strata_short = ["S-I (Legacy)", "S-II (Contemporary)", "S-III (LLM-Generated)"]
for col, (ax, stitle) in enumerate(zip(axes3, strata_short)):
    vals = [round(within_fpr[m][col] * HAM_VOL) for m in MODELS]
    bars = ax.barh(MODELS, vals, color=COLORS, alpha=0.88, zorder=3, height=0.55)
    ax.set_title(stitle, fontsize=10, fontweight="bold", pad=7)
    ax.set_xlim(0, X_MAX)
    ax.set_xticks([0, 500, 1000, 1500, 2000])
    ax.xaxis.grid(True, color="#eeeeee", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top","right"]].set_visible(False)
    for bar, val in zip(bars, vals):
        ax.text(val + 35, bar.get_y() + bar.get_height()/2,
                str(val), va="center", fontsize=9, color="#333333")
    if col == 0:
        ax.set_ylabel("Model", fontsize=10)
fig3.text(0.5, -0.01, "Estimated Daily False Alarms",
          ha="center", va="top", fontsize=11)
fig3.suptitle(
    "Daily False Alarms at FPR@95%TPR — Classical ML Baselines\n"
    "Enterprise proxy: 100K emails/day · 95% ham · 95% recall threshold",
    fontweight="bold", fontsize=11, y=1.04,
)
plt.tight_layout(rect=[0, 0.04, 1, 1])
p3 = OUT / "bar_fpr_operational.png"
fig3.savefig(p3, dpi=150, bbox_inches="tight", facecolor="white")
plt.close(fig3)
print(f"Saved: {p3}")

pairs      = [(0,1),(0,2),(1,0),(1,2),(2,0),(2,1)]
pair_lbls  = ["I\u2192II","I\u2192III","II\u2192I","II\u2192III","III\u2192I","III\u2192II"]
xp         = np.arange(6)
w4         = 0.14
offsets4   = np.linspace(-2, 2, 5) * w4

sorted_models = sorted(
    MODELS,
    key=lambda m: np.mean([cross_f1[m][i][j] for i, j in pairs]),
    reverse=True,
)
color_map = dict(zip(MODELS, COLORS))

fig4, ax4 = plt.subplots(figsize=(12, 5.5))
fig4.patch.set_facecolor("white")

for idx, model in enumerate(sorted_models):
    vals = [cross_f1[model][i][j] for i, j in pairs]
    mu   = np.mean(vals)
    ax4.bar(xp + offsets4[idx], vals, width=w4*0.88,
            label=f"{model} (mean={mu:.3f})",
            color=color_map[model], alpha=0.88, zorder=3)

ax4.axhline(0.5, color="#cc0000", linewidth=1.5, linestyle="--",
            label="Random baseline (F1=0.50)", zorder=4)

for pos in [2, 3]:
    ax4.axvspan(pos-0.45, pos+0.45, alpha=0.07, color="#cc0000", zorder=0)
ax4.text(2,  0.04, "Base-rate\ncollapse", ha="center", fontsize=8,
         color="#cc0000", style="italic")
ax4.text(3,  0.04, "Near-random\n(LLM threat)", ha="center", fontsize=8,
         color="#cc0000", style="italic")

ax4.set_xticks(xp)
ax4.set_xticklabels(pair_lbls, fontsize=12)
ax4.set_xlabel("Train \u2192 Test Stratum", fontsize=11)
ax4.set_ylabel("F1-Macro")
ax4.set_ylim(0, 1.10)
ax4.set_title(
    "Cross-Stratum Off-Diagonal F1 — Generalisation Robustness\n"
    "Shaded = critical failure zones  |  Models ranked by mean off-diagonal F1",
    fontweight="bold", pad=10,
)
ax4.legend(loc="upper right", fontsize=9, ncol=3)
ax4.yaxis.grid(True, color="#eeeeee", linewidth=0.6, zorder=0)
ax4.set_axisbelow(True)
ax4.spines[["top","right"]].set_visible(False)
plt.tight_layout()
p4 = OUT / "scatter_offdiag_generalisation.png"
fig4.savefig(p4, dpi=150, bbox_inches="tight", facecolor="white")
plt.close(fig4)
print(f"Saved: {p4}")

print("\nAll 4 figures saved to docs/figures/classical_ml/")
