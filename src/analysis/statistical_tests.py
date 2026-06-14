"""
Phase 13 -- Statistical Validity & Cross-Model Comparison
Methods §1.11.6

Tests:
  1. McNemar's test        -- paired model comparisons on matched test sets
  2. Wilcoxon signed-rank  -- AUC distributions across seeds (transformers/qlora)
  3. Mann-Whitney U        -- latency distributions, rank-biserial r
  4. Cohen's d             -- effect sizes for F1 and AUC comparisons
  5. Bonferroni correction -- familywise alpha = 0.05
  6. Generalisation heatmaps -- 3x3 F1 cross-stratum matrices per model family

Outputs:
  results/statistical_tests/
    mcnemar_results.csv
    wilcoxon_results.csv
    mannwhitney_latency.csv
    cohens_d_results.csv
    bonferroni_summary.csv
    heatmaps/  -- PNG per model family
    statistical_tests_summary.json
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import wilcoxon, mannwhitneyu
from statsmodels.stats.contingency_tables import mcnemar

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parents[2]
RES_DIR   = ROOT / "results"
SPLITS_DIR = ROOT / "data" / "processed" / "splits"
OUT_DIR   = RES_DIR / "statistical_tests"
HEAT_DIR  = OUT_DIR / "heatmaps"
OUT_DIR.mkdir(parents=True, exist_ok=True)
HEAT_DIR.mkdir(parents=True, exist_ok=True)

ALPHA     = 0.05   # familywise error rate after Bonferroni
STRATA    = ["stratum_i", "stratum_ii", "stratum_iii"]
STRATUM_LABELS = {"stratum_i": "SI", "stratum_ii": "SII", "stratum_iii": "SIII"}

# ── Load all results ──────────────────────────────────────────────────────────
def load_results():
    cml  = pd.read_csv(RES_DIR / "classical_ml"  / "all_results.csv")
    trans = pd.read_csv(RES_DIR / "transformers"  / "all_results.csv")
    qlora = pd.read_csv(RES_DIR / "qlora"         / "all_results.csv")
    front = pd.read_csv(RES_DIR / "frontier"      / "all_results.csv")

    # Normalise youden_j column name
    for df in [qlora, front]:
        if "youdens_j" in df.columns:
            df.rename(columns={"youdens_j": "youden_j"}, inplace=True)

    # Tag family
    cml["family"]   = "classical_ml"
    trans["family"] = "transformer"
    qlora["family"] = "qlora"
    front["family"] = "frontier"

    # Add seed column to non-qlora frames
    for df in [cml, trans, front]:
        if "seed" not in df.columns:
            df["seed"] = np.nan

    return cml, trans, qlora, front

# ── Helper: Cohen's d ─────────────────────────────────────────────────────────
def cohens_d(a, b):
    a, b = np.array(a, dtype=float), np.array(b, dtype=float)
    if len(a) < 2 or len(b) < 2:
        return np.nan
    pooled_sd = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2)
    return (np.mean(a) - np.mean(b)) / pooled_sd if pooled_sd > 0 else 0.0

# ── Helper: rank-biserial correlation ────────────────────────────────────────
def rank_biserial(u_stat, n1, n2):
    return 1 - (2 * u_stat) / (n1 * n2)

# ── 1. McNemar's test ─────────────────────────────────────────────────────────
def run_mcnemar(cml, trans, qlora, front):
    """
    Paired comparison: for each stratum, compare best classical ML vs best
    transformer vs best qlora vs best frontier on matched test predictions.
    Uses F1 macro as proxy for agreement rate (exact prediction files not stored).
    Falls back to approximate McNemar from confusion matrix reconstructed from
    precision/recall/n_test.
    """
    rows = []

    # Reconstruct approximate 2x2 from precision/recall/n_test
    def approx_mcnemar(row_a, row_b, n_test, n_phish):
        # TP_a, FP_a, FN_a from precision/recall
        tp_a = row_a["recall"] * n_phish
        fp_a = tp_a / row_a["precision"] - tp_a if row_a["precision"] > 0 else 0
        fn_a = n_phish - tp_a
        tp_b = row_b["recall"] * n_phish
        fp_b = tp_b / row_b["precision"] - tp_b if row_b["precision"] > 0 else 0
        fn_b = n_phish - tp_b

        # b = A correct, B wrong; c = A wrong, B correct (phishing class)
        b = max(0, round(tp_a - tp_b + fp_b - fp_a))
        c = max(0, round(tp_b - tp_a + fp_a - fp_b))
        if b + c < 1:
            return np.nan, np.nan
        table = np.array([[0, b], [c, 0]])
        try:
            result = mcnemar(table, exact=False, correction=True)
            return float(result.statistic), float(result.pvalue)
        except Exception:
            return np.nan, np.nan

    # Test sizes
    test_sizes = {
        "stratum_i":   {"n": 23288, "phish": 344},
        "stratum_ii":  {"n": 1177,  "phish": 769},
        "stratum_iii": {"n": 2004,  "phish": 1015},
    }

    # Best matched-condition row per family per stratum
    def best_matched(df, stratum):
        # Prefer matched condition; fall back to any row for this stratum
        sub = df[(df["eval_stratum"] == stratum) &
                 (df["eval_type"] == "matched")].copy()
        if sub.empty:
            sub = df[df["eval_stratum"] == stratum].copy()
        if sub.empty:
            return None
        return sub.loc[sub["f1_macro"].idxmax()]

    comparisons = [
        ("classical_ml_best", "transformer_best"),
        ("classical_ml_best", "qlora_best"),
        ("classical_ml_best", "frontier_best"),
        ("transformer_best",  "qlora_best"),
        ("transformer_best",  "frontier_best"),
        ("qlora_best",        "frontier_best"),
    ]

    family_map = {
        "classical_ml_best": cml,
        "transformer_best":  trans,
        "qlora_best":        qlora,
        "frontier_best":     front,
    }

    for stratum in STRATA:
        best = {k: best_matched(v, stratum) for k, v in family_map.items()}
        n    = test_sizes[stratum]["n"]
        np_  = test_sizes[stratum]["phish"]

        for a_key, b_key in comparisons:
            ra, rb = best[a_key], best[b_key]
            if ra is None or rb is None:
                continue
            stat, pval = approx_mcnemar(ra, rb, n, np_)
            rows.append({
                "stratum":    stratum,
                "model_a":    f"{a_key}({ra['model']})",
                "model_b":    f"{b_key}({rb['model']})",
                "f1_a":       round(ra["f1_macro"], 4),
                "f1_b":       round(rb["f1_macro"], 4),
                "mcnemar_stat": round(stat, 4) if not np.isnan(stat) else None,
                "p_value":    round(pval, 4) if not np.isnan(pval) else None,
            })

    df_out = pd.DataFrame(rows)
    # Bonferroni correction
    valid = df_out["p_value"].notna()
    n_tests = valid.sum()
    alpha_adj = ALPHA / n_tests if n_tests > 0 else ALPHA
    df_out["alpha_bonferroni"] = round(alpha_adj, 6)
    df_out["significant"]      = df_out["p_value"].apply(
        lambda p: bool(p < alpha_adj) if pd.notna(p) else None
    )
    df_out.to_csv(OUT_DIR / "mcnemar_results.csv", index=False)
    print(f"  [OK] mcnemar_results.csv  ({len(df_out)} rows, alpha_adj={alpha_adj:.4f})")
    return df_out

# ── 2. Wilcoxon signed-rank: AUC across seeds (QLoRA) ────────────────────────
def run_wilcoxon(qlora, trans):
    """
    Compare AUC across the 3 seeds for QLoRA per stratum.
    For transformers: compare per-stratum AUC across models (distilbert/roberta/deberta).
    """
    rows = []

    # QLoRA: seed variance per stratum
    for stratum in STRATA:
        sub = qlora[qlora["eval_stratum"] == stratum]
        aucs = sub["roc_auc"].dropna().values
        if len(aucs) < 3:
            continue
        # Wilcoxon against the median
        median_arr = np.full(len(aucs), np.median(aucs))
        try:
            stat, pval = wilcoxon(aucs, median_arr, zero_method="zsplit")
        except Exception:
            stat, pval = np.nan, np.nan
        rows.append({
            "comparison": f"qlora_seed_variance_{stratum}",
            "values":     aucs.tolist(),
            "mean_auc":   round(float(np.mean(aucs)), 4),
            "std_auc":    round(float(np.std(aucs)), 4),
            "wilcoxon_stat": round(float(stat), 4) if not np.isnan(stat) else None,
            "p_value":    round(float(pval), 4) if not np.isnan(pval) else None,
        })

    # Transformers: compare distilbert vs roberta vs deberta per stratum
    models_t = ["distilbert", "roberta", "deberta"]
    for stratum in STRATA:
        for i, m1 in enumerate(models_t):
            for m2 in models_t[i+1:]:
                a1 = trans[(trans["model"] == m1) & (trans["eval_stratum"] == stratum)]["roc_auc"].values
                a2 = trans[(trans["model"] == m2) & (trans["eval_stratum"] == stratum)]["roc_auc"].values
                if len(a1) == 0 or len(a2) == 0:
                    continue
                # Pad to same length
                min_len = min(len(a1), len(a2))
                try:
                    stat, pval = wilcoxon(a1[:min_len], a2[:min_len], zero_method="zsplit")
                except Exception:
                    stat, pval = np.nan, np.nan
                d = cohens_d(a1, a2)
                rows.append({
                    "comparison": f"transformer_{m1}_vs_{m2}_{stratum}",
                    "values":     [round(float(x), 4) for x in a1.tolist() + a2.tolist()],
                    "mean_auc":   round(float(np.mean(np.concatenate([a1, a2]))), 4),
                    "std_auc":    round(float(np.std(np.concatenate([a1, a2]))), 4),
                    "wilcoxon_stat": round(float(stat), 4) if not np.isnan(stat) else None,
                    "p_value":    round(float(pval), 4) if not np.isnan(pval) else None,
                    "cohens_d":   round(d, 4) if not np.isnan(d) else None,
                })

    df_out = pd.DataFrame(rows)
    valid  = df_out["p_value"].notna()
    n_tests = valid.sum()
    alpha_adj = ALPHA / n_tests if n_tests > 0 else ALPHA
    df_out["alpha_bonferroni"] = round(alpha_adj, 6)
    df_out["significant"]      = df_out["p_value"].apply(
        lambda p: bool(p < alpha_adj) if pd.notna(p) else None
    )
    df_out.drop(columns=["values"]).to_csv(OUT_DIR / "wilcoxon_results.csv", index=False)
    print(f"  [OK] wilcoxon_results.csv  ({len(df_out)} rows)")
    return df_out

# ── 3. Mann-Whitney U: latency distributions ──────────────────────────────────
def run_mannwhitney(cml, trans, qlora, front):
    """
    Compare latency (p50) distributions across model families.
    Rank-biserial correlation r as effect size.
    """
    rows = []

    def get_latencies(df, label):
        return df["latency_p50_ms"].dropna().values, label

    families = [
        get_latencies(cml,   "classical_ml"),
        get_latencies(trans, "transformer"),
        get_latencies(qlora, "qlora"),
        get_latencies(front, "frontier"),
    ]

    for i, (a, la) in enumerate(families):
        for j, (b, lb) in enumerate(families):
            if j <= i:
                continue
            if len(a) == 0 or len(b) == 0:
                continue
            try:
                stat, pval = mannwhitneyu(a, b, alternative="two-sided")
                rbc = rank_biserial(stat, len(a), len(b))
                d   = cohens_d(a, b)
            except Exception:
                stat, pval, rbc, d = np.nan, np.nan, np.nan, np.nan
            rows.append({
                "family_a":          la,
                "family_b":          lb,
                "median_latency_a":  round(float(np.median(a)), 3),
                "median_latency_b":  round(float(np.median(b)), 3),
                "mannwhitney_u":     round(float(stat), 2) if not np.isnan(stat) else None,
                "p_value":           round(float(pval), 6) if not np.isnan(pval) else None,
                "rank_biserial_r":   round(float(rbc), 4) if not np.isnan(rbc) else None,
                "cohens_d":          round(float(d), 4) if not np.isnan(d) else None,
            })

    df_out = pd.DataFrame(rows)
    valid  = df_out["p_value"].notna()
    n_tests = valid.sum()
    alpha_adj = ALPHA / n_tests if n_tests > 0 else ALPHA
    df_out["alpha_bonferroni"] = round(alpha_adj, 6)
    df_out["significant"]      = df_out["p_value"].apply(
        lambda p: bool(p < alpha_adj) if pd.notna(p) else None
    )
    df_out.to_csv(OUT_DIR / "mannwhitney_latency.csv", index=False)
    print(f"  [OK] mannwhitney_latency.csv  ({len(df_out)} rows)")
    return df_out

# ── 4. Cohen's d: F1 pairwise across families ─────────────────────────────────
def run_cohens_d(cml, trans, qlora, front):
    rows = []
    families = {
        "classical_ml": cml,
        "transformer":  trans,
        "qlora":        qlora,
        "frontier":     front,
    }
    keys = list(families.keys())
    for stratum in STRATA:
        for i, k1 in enumerate(keys):
            for k2 in keys[i+1:]:
                a = families[k1][families[k1]["eval_stratum"] == stratum]["f1_macro"].values
                b = families[k2][families[k2]["eval_stratum"] == stratum]["f1_macro"].values
                if len(a) == 0 or len(b) == 0:
                    continue
                d = cohens_d(a, b)
                rows.append({
                    "stratum":   stratum,
                    "family_a":  k1,
                    "family_b":  k2,
                    "mean_f1_a": round(float(np.mean(a)), 4),
                    "mean_f1_b": round(float(np.mean(b)), 4),
                    "cohens_d":  round(d, 4) if not np.isnan(d) else None,
                    "magnitude": (
                        "negligible" if abs(d) < 0.2 else
                        "small"      if abs(d) < 0.5 else
                        "medium"     if abs(d) < 0.8 else
                        "large"
                    ) if not np.isnan(d) else None,
                })
    df_out = pd.DataFrame(rows)
    df_out.to_csv(OUT_DIR / "cohens_d_results.csv", index=False)
    print(f"  [OK] cohens_d_results.csv  ({len(df_out)} rows)")
    return df_out

# ── 5. Generalisation heatmaps ────────────────────────────────────────────────
def plot_heatmaps(cml, trans, qlora, front):
    """
    3x3 F1 heatmap: rows = train stratum, cols = eval stratum.
    One heatmap per model family. Best model per family shown.
    """
    family_configs = [
        ("Classical ML\n(best per config)", cml,   "classical_ml"),
        ("Transformer\n(best per config)",  trans,  "transformer"),
        ("QLoRA Llama-3\n(best seed)",      qlora,  "qlora"),
        ("Frontier LLM\n(direct zero-shot)",front,  "frontier"),
    ]

    for title, df, family in family_configs:
        # Build 3x3 matrix
        matrix = pd.DataFrame(np.nan,
                               index=STRATA,
                               columns=STRATA)

        for train_s in STRATA:
            for eval_s in STRATA:
                sub = df[df["eval_stratum"] == eval_s]
                # Filter by train config where possible
                if "train_config" in sub.columns:
                    tc_sub = sub[sub["train_config"] == train_s]
                    if tc_sub.empty:
                        tc_sub = sub
                else:
                    tc_sub = sub
                if tc_sub.empty:
                    continue
                matrix.loc[train_s, eval_s] = tc_sub["f1_macro"].max()

        matrix = matrix.astype(float)
        labels = [STRATUM_LABELS[s] for s in STRATA]
        matrix.index   = labels
        matrix.columns = labels

        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(
            matrix,
            annot=True,
            fmt=".3f",
            cmap="RdYlGn",
            vmin=0.0,
            vmax=1.0,
            linewidths=0.5,
            ax=ax,
            cbar_kws={"label": "F1 Macro"},
        )
        ax.set_title(f"{title}", fontsize=11, pad=10)
        ax.set_xlabel("Evaluation Stratum", fontsize=9)
        ax.set_ylabel("Training Stratum",   fontsize=9)
        plt.tight_layout()

        fname = f"heatmap_{family}.png"
        plt.savefig(str(HEAT_DIR / fname), dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  [OK] heatmaps/{fname}")

# ── 6. Bonferroni summary ─────────────────────────────────────────────────────
def write_bonferroni_summary(mcn_df, wil_df, mwu_df):
    total_tests = (
        int(mcn_df["p_value"].notna().sum()) +
        int(wil_df["p_value"].notna().sum()) +
        int(mwu_df["p_value"].notna().sum())
    )
    alpha_adj = ALPHA / total_tests if total_tests > 0 else ALPHA

    sig_mcn = int(mcn_df["significant"].sum()) if "significant" in mcn_df else 0
    sig_wil = int(wil_df["significant"].sum()) if "significant" in wil_df else 0
    sig_mwu = int(mwu_df["significant"].sum()) if "significant" in mwu_df else 0

    summary = {
        "familywise_alpha":        ALPHA,
        "total_tests":             total_tests,
        "bonferroni_alpha":        round(alpha_adj, 6),
        "significant_mcnemar":     sig_mcn,
        "significant_wilcoxon":    sig_wil,
        "significant_mannwhitney": sig_mwu,
        "total_significant":       sig_mcn + sig_wil + sig_mwu,
    }
    with open(OUT_DIR / "bonferroni_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"  [OK] bonferroni_summary.json  "
          f"(total_tests={total_tests}, alpha_adj={alpha_adj:.5f}, "
          f"sig={sig_mcn + sig_wil + sig_mwu})")
    return summary

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("Phase 13 -- Statistical Validity & Cross-Model Comparison")
    print("=" * 60)

    cml, trans, qlora, front = load_results()
    print(f"  Loaded: CML={len(cml)}, Trans={len(trans)}, "
          f"QLoRA={len(qlora)}, Frontier={len(front)}")
    print()

    print("  Running McNemar's test...")
    mcn_df = run_mcnemar(cml, trans, qlora, front)

    print("  Running Wilcoxon signed-rank test...")
    wil_df = run_wilcoxon(qlora, trans)

    print("  Running Mann-Whitney U test (latency)...")
    mwu_df = run_mannwhitney(cml, trans, qlora, front)

    print("  Computing Cohen's d (F1 pairwise)...")
    run_cohens_d(cml, trans, qlora, front)

    print("  Plotting generalisation heatmaps...")
    plot_heatmaps(cml, trans, qlora, front)

    print("  Writing Bonferroni summary...")
    summary = write_bonferroni_summary(mcn_df, wil_df, mwu_df)

    print()
    print("=" * 60)
    print("PHASE 13 STATISTICAL TESTS COMPLETE.")
    print(f"  Total tests       : {summary['total_tests']}")
    print(f"  Bonferroni alpha  : {summary['bonferroni_alpha']}")
    print(f"  Significant tests : {summary['total_significant']}")
    print("=" * 60)
    sys.exit(0)

if __name__ == "__main__":
    main()
