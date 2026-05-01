"""
train_classical.py
==================
Trains classical ML baselines on one stratum and evaluates on val + test.

Models:
    LR   - Logistic Regression (L2, liblinear)
    SVM  - Linear SVM (LinearSVC + Platt calibration for probabilities)
    RF   - Random Forest (100 trees)
    XGB  - XGBoost (histogram)
    LGB  - LightGBM (histogram)

All models use class_weight="balanced" (or equivalent) to handle imbalance.

Usage:
    python src/models/train_classical.py --stratum i
    python src/models/train_classical.py --stratum ii
    python src/models/train_classical.py --stratum iii

Outputs:
    outputs/results/classical_ml/stratum_{i/ii/iii}_results.csv
    outputs/results/classical_ml/stratum_{i/ii/iii}_results.json
    outputs/features/{stratum}_pipeline.joblib   (fitted feature pipeline)
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.features.build_features import FeaturePipeline
from src.utils.manifest_utils import ManifestWriter

from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.svm import LinearSVC

try:
    import xgboost as xgb
    import lightgbm as lgb
except ImportError as e:
    print(f"ERROR: {e}. Run: pip install xgboost lightgbm")
    sys.exit(1)

csv.field_size_limit(10_000_000)

# ------------------------------------------------------------------
# PATHS
# ------------------------------------------------------------------
SPLITS_BASE   = Path("data/splits")
FEATURES_DIR  = Path("outputs/features")
RESULTS_DIR   = Path("outputs/results/classical_ml")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FEATURES_DIR.mkdir(parents=True, exist_ok=True)

STRATUM_MAP = {"i": "stratum_i", "ii": "stratum_ii", "iii": "stratum_iii"}

RANDOM_STATE = 42


# ------------------------------------------------------------------
# METRICS
# ------------------------------------------------------------------
def fpr_at_95_tpr(y_true, y_prob) -> float:
    """False positive rate when TPR >= 0.95."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    idx = np.searchsorted(tpr, 0.95)
    if idx >= len(fpr):
        return float(fpr[-1])
    return float(fpr[idx])


def compute_metrics(y_true, y_pred, y_prob, split: str, model: str,
                    stratum: str, elapsed_s: float, n_train: int) -> dict:
    n = len(y_true)
    return {
        "stratum":           stratum,
        "model":             model,
        "split":             split,
        "n_samples":         n,
        "n_train":           n_train,
        "elapsed_train_s":   round(elapsed_s, 2),
        "latency_ms_per_email": round(elapsed_s / n_train * 1000, 4)
                               if split == "train" else None,
        "f1_macro":          round(f1_score(y_true, y_pred, average="macro",
                                            zero_division=0), 4),
        "precision":         round(precision_score(y_true, y_pred,
                                                   zero_division=0), 4),
        "recall":            round(recall_score(y_true, y_pred,
                                                zero_division=0), 4),
        "roc_auc":           round(roc_auc_score(y_true, y_prob), 4),
        "pr_auc":            round(average_precision_score(y_true, y_prob), 4),
        "balanced_acc":      round(balanced_accuracy_score(y_true, y_pred), 4),
        "mcc":               round(matthews_corrcoef(y_true, y_pred), 4),
        "fpr_at_95tpr":      round(fpr_at_95_tpr(y_true, y_prob), 4),
    }


# ------------------------------------------------------------------
# MODEL DEFINITIONS
# ------------------------------------------------------------------
def get_models(n_phish: int, n_ham: int) -> dict:
    """Return model dict. scale_pos_weight for XGB/LGB = n_ham / n_phish."""
    spw = n_ham / max(n_phish, 1)
    return {
        "LR": LogisticRegression(
            C=1.0, max_iter=500, solver="lbfgs",
            class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1,
        ),
        "SVM": CalibratedClassifierCV(
            LinearSVC(C=0.1, max_iter=2000, class_weight="balanced",
                      random_state=RANDOM_STATE),
            method="sigmoid", cv=3,
        ),
        "RF": RandomForestClassifier(
            n_estimators=100, max_depth=None, min_samples_leaf=2,
            class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1,
        ),
        "XGB": xgb.XGBClassifier(
            n_estimators=200, learning_rate=0.1, max_depth=6,
            scale_pos_weight=spw, tree_method="hist",
            random_state=RANDOM_STATE, n_jobs=-1,
            eval_metric="logloss", verbosity=0,
        ),
        "LGB": lgb.LGBMClassifier(
            n_estimators=200, learning_rate=0.1, max_depth=-1,
            scale_pos_weight=spw, random_state=RANDOM_STATE,
            n_jobs=-1, verbosity=-1,
        ),
    }


# ------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------
def main(stratum_key: str):
    stratum_dir  = SPLITS_BASE / STRATUM_MAP[stratum_key]
    stratum_name = STRATUM_MAP[stratum_key].replace("_", " ").upper()

    print("=" * 65)
    print(f"CLASSICAL ML TRAINING — {stratum_name}")
    print("=" * 65)

    mw = ManifestWriter(
        script_name=f"train_classical_{stratum_key}",
        random_seed=RANDOM_STATE,
        parameters={"stratum": stratum_key, "models": ["LR", "SVM", "RF", "XGB", "LGB"]},
        notes=f"Classical ML baselines on {stratum_name}. "
              "class_weight=balanced (LR/SVM/RF); scale_pos_weight (XGB/LGB).",
    )

    # ---- Load splits ------------------------------------------------
    print("\nLoading splits...")
    dfs = {}
    for split in ("train", "val", "test"):
        path = stratum_dir / f"{split}.csv"
        mw.add_input(str(path))
        df = pd.read_csv(path, encoding="utf-8", encoding_errors="replace",
                         keep_default_na=False)
        dfs[split] = df
        n0 = int((df["label"] == 0).sum())
        n1 = int((df["label"] == 1).sum())
        print(f"  {split:6s}: {len(df):>8,} rows  (ham={n0:,}  phish={n1:,})")

    # ---- Feature engineering ----------------------------------------
    print("\nBuilding features (fitting on train)...")
    fp = FeaturePipeline()
    X = {}
    X["train"] = fp.fit_transform(dfs["train"])
    X["val"]   = fp.transform(dfs["val"])
    X["test"]  = fp.transform(dfs["test"])

    pipeline_path = str(FEATURES_DIR / f"{STRATUM_MAP[stratum_key]}_pipeline.joblib")
    fp.save(pipeline_path)
    mw.add_output(pipeline_path)

    y = {s: dfs[s]["label"].values for s in ("train", "val", "test")}
    n_train_phish = int((y["train"] == 1).sum())
    n_train_ham   = int((y["train"] == 0).sum())

    # ---- Train and evaluate -----------------------------------------
    models = get_models(n_train_phish, n_train_ham)
    all_results = []

    for name, model in models.items():
        print(f"\n  [{name}] Training...")
        t0 = time.perf_counter()
        model.fit(X["train"], y["train"])
        elapsed = time.perf_counter() - t0
        print(f"    Trained in {elapsed:.1f}s")

        for split in ("val", "test"):
            y_pred = model.predict(X[split])
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X[split])[:, 1]
            else:
                y_prob = model.decision_function(X[split])

            metrics = compute_metrics(
                y[split], y_pred, y_prob,
                split=split, model=name,
                stratum=stratum_key.upper(),
                elapsed_s=elapsed,
                n_train=len(y["train"]),
            )
            all_results.append(metrics)

            f1  = metrics["f1_macro"]
            auc = metrics["roc_auc"]
            fpr = metrics["fpr_at_95tpr"]
            print(f"    {split:6s}: F1={f1:.4f}  ROC-AUC={auc:.4f}  FPR@95TPR={fpr:.4f}")

    # ---- Save results -----------------------------------------------
    csv_path  = RESULTS_DIR / f"stratum_{stratum_key}_results.csv"
    json_path = RESULTS_DIR / f"stratum_{stratum_key}_results.json"

    pd.DataFrame(all_results).to_csv(csv_path, index=False)
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)

    mw.add_output(str(csv_path))
    mw.add_output(str(json_path))
    mw.set_counts({
        "n_models":  len(models),
        "n_results": len(all_results),
        "train_rows": len(y["train"]),
    })
    mw.write()

    print(f"\nResults saved -> {csv_path}")
    print("=" * 65)
    print("DONE.")
    print("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stratum", required=True, choices=["i", "ii", "iii"],
        help="Which stratum to train on: i, ii, or iii",
    )
    args = parser.parse_args()
    main(args.stratum)
    