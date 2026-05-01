"""
generate_results_doc.py
=======================
Reads all results CSVs and writes a consolidated, human-readable markdown
results record to docs/results/CLASSICAL_ML_RESULTS.md.
"""

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

RESULTS_DIR = Path("outputs/results/classical_ml")
OUT_PATH    = Path("docs/results/CLASSICAL_ML_RESULTS.md")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

STRATUM_LABELS = {
    "I":   "Stratum I — Backward Comparability (Enron + Nazario)",
    "II":  "Stratum II — Contemporary General (phishing_pot + CSDMC2010)",
    "III": "Stratum III — LLM-Generated (PhishFuzzer entity_rephrased_v1)",
}

MODEL_ORDER  = ["LR", "SVM", "RF", "XGB", "LGB"]
METRIC_COLS  = ["f1_macro", "precision", "recall", "roc_auc",
                "pr_auc", "balanced_acc", "mcc", "fpr_at_95tpr"]
METRIC_NAMES = ["F1-macro", "Precision", "Recall", "ROC-AUC",
                "PR-AUC", "Bal-Acc", "MCC", "FPR@95TPR"]

SPLIT_NOTES = {
    "I":   "n_test=23,288 | Ham=22,947 (98.5%) | Phishing=341 (1.5%)",
    "II":  "n_test=1,177  | Ham=415 (35.3%)    | Phishing=762 (64.7%)",
    "III": "n_test=2,004  | Ham=990 (49.4%)    | Phishing=1,014 (50.6%)",
}

MODEL_DESCRIPTIONS = {
    "LR":  "Logistic Regression (lbfgs, C=1.0, max_iter=500, class_weight=balanced)",
    "SVM": "Linear SVM (C=0.1, calibrated via Platt scaling, class_weight=balanced)",
    "RF":  "Random Forest (100 trees, min_samples_leaf=2, class_weight=balanced)",
    "XGB": "XGBoost histogram (200 estimators, lr=0.1, depth=6, scale_pos_weight)",
    "LGB": "LightGBM histogram (200 estimators, lr=0.1, scale_pos_weight)",
}


def fmt(v):
    """Format a metric value for display."""
    try:
        return f"{float(v):.4f}"
    except (TypeError, ValueError):
        return str(v)


def table_row(model, row, metrics):
    cells = [f"**{model}**"] + [fmt(row.get(m, "—")) for m in metrics]
    return "| " + " | ".join(cells) + " |"


def build_within_stratum_section(df_all: pd.DataFrame) -> str:
    lines = []
    lines.append("## 1. Within-Stratum Test Results\n")
    lines.append(
        "Each model is trained on the training split of a stratum and "
        "evaluated on the held-out test split of the **same stratum**. "
        "These are the matched-distribution baselines against which "
        "cross-stratum generalisation is measured.\n"
    )

    for stratum_key in ["I", "II", "III"]:
        lines.append(f"### {STRATUM_LABELS[stratum_key]}\n")
        lines.append(f"_{SPLIT_NOTES[stratum_key]}_\n")

        df_s = df_all[
            (df_all["stratum"] == stratum_key) &
            (df_all["split"] == "test")
        ].copy()

        if df_s.empty:
            lines.append("_Results not yet available._\n")
            continue

        # Header
        header = "| Model | " + " | ".join(METRIC_NAMES) + " |"
        sep    = "|-------|" + "|".join(["-------"] * len(METRIC_NAMES)) + "|"
        lines.append(header)
        lines.append(sep)

        for model in MODEL_ORDER:
            row_df = df_s[df_s["model"] == model]
            if row_df.empty:
                continue
            row = row_df.iloc[0]
            lines.append(table_row(model, row, METRIC_COLS))

        lines.append("")

        # Best model callout
        best_row = df_s.loc[df_s["f1_macro"].astype(float).idxmax()]
        best_model = best_row["model"]
        best_f1    = fmt(best_row["f1_macro"])
        best_fpr   = fmt(best_row["fpr_at_95tpr"])
        lines.append(
            f"> **Best model (F1-macro):** {best_model} "
            f"(F1={best_f1}, FPR@95TPR={best_fpr})\n"
        )

    return "\n".join(lines)


def build_cross_stratum_section(df_cross: pd.DataFrame) -> str:
    lines = []
    lines.append("## 2. Cross-Stratum Generalisation Matrix\n")
    lines.append(
        "Each model is trained on one stratum and evaluated on the test split "
        "of every stratum. Off-diagonal cells measure out-of-distribution "
        "generalisation — the central empirical contribution of this study. "
        "Diagonal cells reproduce the within-stratum test results above.\n"
    )

    if df_cross.empty:
        lines.append("_Results not yet available (Step 7.3 pending)._\n")
        return "\n".join(lines)

    for model in MODEL_ORDER:
        lines.append(f"### {model} — {MODEL_DESCRIPTIONS[model]}\n")
        df_m = df_cross[df_cross["model"] == model]

        header = "| Train \\ Test | Test I | Test II | Test III |"
        sep    = "|-------------|--------|---------|----------|"
        lines.append(header)
        lines.append(sep)

        for train_s in ["I", "II", "III"]:
            cells = []
            for test_s in ["I", "II", "III"]:
                match = df_m[
                    (df_m["train_stratum"] == train_s) &
                    (df_m["test_stratum"]  == test_s)
                ]
                if match.empty:
                    cells.append("—")
                else:
                    f1  = fmt(match.iloc[0]["f1_macro"])
                    fpr = fmt(match.iloc[0]["fpr_at_95tpr"])
                    diag = " ◀" if train_s == test_s else ""
                    cells.append(f"F1={f1} / FPR={fpr}{diag}")
            lines.append(f"| **Train {train_s}** | {' | '.join(cells)} |")

        lines.append("")

    lines.append(
        "> ◀ Diagonal = within-stratum (matched distribution). "
        "Off-diagonal = cross-stratum (out-of-distribution).\n"
    )
    return "\n".join(lines)


def build_interpretation_section(df_within: pd.DataFrame) -> str:
    lines = []
    lines.append("## 3. Key Observations\n")

    if df_within.empty:
        lines.append("_Pending results._\n")
        return "\n".join(lines)

    test_df = df_within[df_within["split"] == "test"].copy()
    test_df["f1_macro"] = test_df["f1_macro"].astype(float)
    test_df["fpr_at_95tpr"] = test_df["fpr_at_95tpr"].astype(float)

    # Stratum II vs III degradation
    lines.append("### Stratum II → III F1 Degradation (within-stratum)\n")
    lines.append(
        "The table below quantifies how much each model's F1 drops from "
        "Stratum II (contemporary phishing) to Stratum III (LLM-generated "
        "phishing) under matched training conditions.\n"
    )
    lines.append("| Model | Stratum II F1 | Stratum III F1 | Δ F1 |")
    lines.append("|-------|--------------|----------------|------|")

    for model in MODEL_ORDER:
        s2 = test_df[(test_df["stratum"] == "II") & (test_df["model"] == model)]
        s3 = test_df[(test_df["stratum"] == "III") & (test_df["model"] == model)]
        if s2.empty or s3.empty:
            continue
        f2   = float(s2.iloc[0]["f1_macro"])
        f3   = float(s3.iloc[0]["f1_macro"])
        delta = f3 - f2
        sign  = "+" if delta >= 0 else ""
        lines.append(
            f"| {model} | {f2:.4f} | {f3:.4f} | {sign}{delta:.4f} |"
        )

    lines.append("")
    lines.append("### FPR@95TPR Comparison (operational deployment metric)\n")
    lines.append(
        "FPR@95TPR is the false positive rate when the detector's recall is "
        "held at 95%. For an enterprise processing 100,000 emails/day, "
        "FPR@95TPR × 95,000 (ham volume) gives the expected daily false alarm count.\n"
    )
    lines.append("| Model | S-I FPR | S-II FPR | S-III FPR | S-III daily FAs* |")
    lines.append("|-------|---------|----------|-----------|-----------------|")

    for model in MODEL_ORDER:
        fprs = {}
        for s in ["I", "II", "III"]:
            row = test_df[(test_df["stratum"] == s) & (test_df["model"] == model)]
            fprs[s] = float(row.iloc[0]["fpr_at_95tpr"]) if not row.empty else float("nan")
        daily_fa = int(fprs.get("III", 0) * 95_000)
        lines.append(
            f"| {model} | {fprs.get('I', 0):.4f} | "
            f"{fprs.get('II', 0):.4f} | {fprs.get('III', 0):.4f} | "
            f"~{daily_fa:,} |"
        )

    lines.append("")
    lines.append("_\\* Estimated daily false alarms at 100K emails/day, 95% ham volume._\n")
    return "\n".join(lines)


def main():
    print("Generating consolidated results document...")

    # Load within-stratum results
    within_frames = []
    for s in ["i", "ii", "iii"]:
        p = RESULTS_DIR / f"stratum_{s}_results.csv"
        if p.exists():
            df = pd.read_csv(p)
            df["stratum"] = s.upper()
            within_frames.append(df)
        else:
            print(f"  WARNING: {p} not found — skipping Stratum {s.upper()}")

    df_within = pd.concat(within_frames, ignore_index=True) if within_frames else pd.DataFrame()

    # Load cross-stratum results
    cross_path = RESULTS_DIR / "cross_stratum_matrix.csv"
    df_cross   = pd.read_csv(cross_path) if cross_path.exists() else pd.DataFrame()

    # Build document
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M SAST")
    lines = [
        "# Classical ML Baseline Results",
        "",
        f"**Project:** Defending Against AI-Generated Phishing Emails  ",
        f"**Author:** Vuyo Mashiqa  ",
        f"**Generated:** {timestamp}  ",
        f"**Dataset version:** v1.0-data-frozen  ",
        f"**Commit:** see `git log` for hash at time of generation  ",
        "",
        "This document is auto-generated by `src/reporting/generate_results_doc.py` "
        "from the canonical results CSVs in `outputs/results/classical_ml/`. "
        "Re-run that script after any experiment to keep it current. "
        "Do not edit this file manually.",
        "",
        "---",
        "",
        "## Feature Engineering Summary",
        "",
        "| Component | Configuration | Max Features |",
        "|-----------|--------------|-------------|",
        "| Body TF-IDF (word) | Unigram/bigram, sublinear_tf, min_df=2, max_df=0.95 | 50,000 |",
        "| Body TF-IDF (char) | 3–5gram (char_wb), sublinear_tf, min_df=2 | 30,000 |",
        "| Subject TF-IDF (word) | Unigram/bigram, sublinear_tf, min_df=2 | 10,000 |",
        "| URL features | Count, suspicious-TLD fraction, IP-URL count, @-in-URL, max URL length | 5 |",
        "| Sender features | Domain-mismatch proxy, urgency marker in subject | 2 |",
        "| **Total (Stratum I)** | | **90,007** |",
        "",
        "Feature pipelines are fitted on the **training split only** and serialised "
        "to `outputs/features/{stratum}_pipeline.joblib`. Cross-stratum experiments "
        "apply the train-stratum pipeline to test-stratum data without re-fitting, "
        "correctly simulating deployment on an unseen distribution.",
        "",
        "---",
        "",
        "## Model Hyperparameters",
        "",
        "| Model | Key Hyperparameters | Imbalance Handling |",
        "|-------|--------------------|--------------------|",
    ]
    for m, desc in MODEL_DESCRIPTIONS.items():
        imb = "scale_pos_weight=n_ham/n_phish" if m in ("XGB", "LGB") else "class_weight=balanced"
        lines.append(f"| {m} | {desc} | {imb} |")

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append(build_within_stratum_section(df_within))
    lines.append("---")
    lines.append("")
    lines.append(build_cross_stratum_section(df_cross))
    lines.append("---")
    lines.append("")
    lines.append(build_interpretation_section(df_within))
    lines.append("---")
    lines.append("")
    lines.append("## 4. Reproducibility")
    lines.append("")
    lines.append(
        "All results are reproducible from the frozen dataset (tag `v1.0-data-frozen`) "
        "by running the following sequence:"
    )
    lines.append("")
    lines.append("```bash")
    lines.append("python src/models/train_classical.py --stratum i")
    lines.append("python src/models/train_classical.py --stratum ii")
    lines.append("python src/models/train_classical.py --stratum iii")
    lines.append("python src/models/cross_stratum_classical.py")
    lines.append("python src/reporting/generate_results_doc.py")
    lines.append("```")
    lines.append("")
    lines.append(
        "Each training run produces a JSON manifest in `outputs/manifests/` "
        "recording the Git commit SHA, input/output file hashes, record counts, "
        "and random seed at execution time."
    )

    OUT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Written -> {OUT_PATH}")
    print(f"  Lines: {len(lines):,}")
    print("Done.")


if __name__ == "__main__":
    main()
