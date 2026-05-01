"""
cross_stratum_classical.py
==========================
Cross-stratum generalisation experiment for classical ML models.

For each (train_stratum, test_stratum) pair in {I, II, III}^2:
  - Loads the fitted feature pipeline from the train stratum
  - Transforms the test stratum's test split using that pipeline
  - Evaluates all 5 trained models (LR, SVM, RF, XGB, LGB)

The diagonal (train==test) reuses within-stratum test results from
stratum_X_results.csv rather than re-running inference, for consistency.

Output:
    outputs/results/classical_ml/cross_stratum_matrix.csv
    outputs/results/classical_ml/cross_stratum_matrix.json
"""

import csv
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.features.build_features import FeaturePipeline
from src.utils.manifest_utils import ManifestWriter
from sklearn.metrics import (
    average_precision_score, balanced_accuracy_score,
    f1_score, matthews_corrcoef, precision_score,
    recall_score, roc_auc_score, roc_curve,
)

csv.field_size_limit(10_000_000)

SPLITS_BASE  = Path("data/splits")
FEATURES_DIR = Path("outputs/features")
RESULTS_DIR  = Path("outputs/results/classical_ml")
MANIFESTS    = Path("outputs/manifests")

STRATA = ["i", "ii", "iii"]
STRATUM_DIRS = {"i": "stratum_i", "ii": "stratum_ii", "iii": "stratum_iii"}
MODELS = ["LR", "SVM", "RF", "XGB", "LGB"]

RANDOM_STATE = 42


def fpr_at_95_tpr(y_true, y_prob) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    idx = np.searchsorted(tpr, 0.95)
    return float(fpr[min(idx, len(fpr) - 1)])


def compute_metrics(y_true, y_pred, y_prob,
                    train_stratum, test_stratum, model) -> dict:
    return {
        "train_stratum":  train_stratum.upper(),
        "test_stratum":   test_stratum.upper(),
        "model":          model,
        "n_test":         len(y_true),
        "is_diagonal":    train_stratum == test_stratum,
        "f1_macro":       round(f1_score(y_true, y_pred, average="macro",
                                         zero_division=0), 4),
        "precision":      round(precision_score(y_true, y_pred,
                                                zero_division=0), 4),
        "recall":         round(recall_score(y_true, y_pred,
                                             zero_division=0), 4),
        "roc_auc":        round(roc_auc_score(y_true, y_prob), 4),
        "pr_auc":         round(average_precision_score(y_true, y_prob), 4),
        "balanced_acc":   round(balanced_accuracy_score(y_true, y_pred), 4),
        "mcc":            round(matthews_corrcoef(y_true, y_pred), 4),
        "fpr_at_95tpr":   round(fpr_at_95_tpr(y_true, y_prob), 4),
    }


def load_test_split(stratum_key: str) -> pd.DataFrame:
    path = SPLITS_BASE / STRATUM_DIRS[stratum_key] / "test.csv"
    return pd.read_csv(path, encoding="utf-8", encoding_errors="replace",
                       keep_default_na=False)


def load_trained_models(train_stratum: str) -> dict:
    """
    Re-trains all models on the training stratum to get fitted model objects.
    Uses the same hyperparameters as train_classical.py.
    """
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import LinearSVC
    import xgboost as xgb
    import lightgbm as lgb

    save_dir = Path("outputs/models/classical_ml") / f"stratum_{train_stratum}"
    if save_dir.exists() and all((save_dir / f"{m}.joblib").exists() for m in ["LR","SVM","RF","XGB","LGB"]):
        print(f"    Loading cached models from {save_dir} ...")
        fp = FeaturePipeline.load(
            str(FEATURES_DIR / f"{STRATUM_DIRS[train_stratum]}_pipeline.joblib")
        )
        fitted = {m: joblib.load(save_dir / f"{m}.joblib") for m in ["LR","SVM","RF","XGB","LGB"]}
        return fitted, fp
    fp = FeaturePipeline.load(
        str(FEATURES_DIR / f"{STRATUM_DIRS[train_stratum]}_pipeline.joblib")
    )

    train_path = SPLITS_BASE / STRATUM_DIRS[train_stratum] / "train.csv"
    df_train = pd.read_csv(train_path, encoding="utf-8",
                           encoding_errors="replace", keep_default_na=False)
    X_train = fp.transform(df_train)
    y_train = df_train["label"].values

    n_phish = int((y_train == 1).sum())
    n_ham   = int((y_train == 0).sum())
    spw     = n_ham / max(n_phish, 1)

    print(f"    Re-fitting models on Stratum {train_stratum.upper()} train "
          f"({len(y_train):,} rows, spw={spw:.1f})...")

    model_defs = {
        "LR":  LogisticRegression(C=1.0, max_iter=500, solver="lbfgs",
                                  class_weight="balanced",
                                  random_state=RANDOM_STATE, n_jobs=-1),
        "SVM": CalibratedClassifierCV(
                   LinearSVC(C=0.1, max_iter=2000, class_weight="balanced",
                             random_state=RANDOM_STATE),
                   method="sigmoid", cv=3),
        "RF":  RandomForestClassifier(n_estimators=100, min_samples_leaf=2,
                                      class_weight="balanced",
                                      random_state=RANDOM_STATE, n_jobs=-1),
        "XGB": xgb.XGBClassifier(n_estimators=200, learning_rate=0.1,
                                  max_depth=6, scale_pos_weight=spw,
                                  tree_method="hist",
                                  random_state=RANDOM_STATE, n_jobs=-1,
                                  eval_metric="logloss", verbosity=0),
        "LGB": lgb.LGBMClassifier(n_estimators=200, learning_rate=0.1,
                                  scale_pos_weight=spw,
                                  random_state=RANDOM_STATE,
                                  n_jobs=-1, verbosity=-1),
    }

    fitted = {}
    for name, m in model_defs.items():
        t0 = time.perf_counter()
        m.fit(X_train, y_train)
        elapsed = time.perf_counter() - t0
        fitted[name] = m
        print(f"      {name}: {elapsed:.1f}s")

    # Save fitted models so a crash does not lose the training run
    save_dir = Path("outputs/models/classical_ml") / f"stratum_{train_stratum}"
    save_dir.mkdir(parents=True, exist_ok=True)
    for mname, mobj in fitted.items():
        joblib.dump(mobj, save_dir / f"{mname}.joblib")
    print(f"      Models saved -> {save_dir}")
    return fitted, fp


def main():
    print("=" * 65)
    print("CROSS-STRATUM GENERALISATION — CLASSICAL ML")
    print("=" * 65)

    mw = ManifestWriter(
        script_name="cross_stratum_classical",
        random_seed=RANDOM_STATE,
        parameters={"strata": STRATA, "models": MODELS,
                    "n_pairs": len(STRATA) ** 2},
        notes="9-pair cross-stratum generalisation matrix for classical ML. "
              "Models fitted on train split of train_stratum; evaluated on "
              "test split of test_stratum using train_stratum's feature pipeline.",
    )

    all_results = []

    for train_s in STRATA:
        print(f"\n{'-'*50}")
        print(f"  Training stratum: {train_s.upper()}")
        print(f"{'-'*50}")

        fitted_models, fp = load_trained_models(train_s)

        for test_s in STRATA:
            label = f"  Train={train_s.upper()} -> Test={test_s.upper()}"
            is_diag = (train_s == test_s)
            marker = " [diagonal]" if is_diag else ""
            print(f"\n  {label}{marker}")

            df_test = load_test_split(test_s)
            X_test  = fp.transform(df_test)
            y_true  = df_test["label"].values

            mw.add_input(
                str(SPLITS_BASE / STRATUM_DIRS[test_s] / "test.csv")
            )

            for model_name, model in fitted_models.items():
                y_pred = model.predict(X_test)
                y_prob = (model.predict_proba(X_test)[:, 1]
                          if hasattr(model, "predict_proba")
                          else model.decision_function(X_test))

                row = compute_metrics(y_true, y_pred, y_prob,
                                      train_s, test_s, model_name)
                all_results.append(row)

                f1  = row["f1_macro"]
                auc = row["roc_auc"]
                fpr = row["fpr_at_95tpr"]
                print(f"    {model_name}: F1={f1:.4f}  "
                      f"ROC-AUC={auc:.4f}  FPR@95TPR={fpr:.4f}")

    # ---- Save -------------------------------------------------------
    csv_path  = RESULTS_DIR / "cross_stratum_matrix.csv"
    json_path = RESULTS_DIR / "cross_stratum_matrix.json"

    df_out = pd.DataFrame(all_results)
    df_out.to_csv(csv_path, index=False)
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)

    mw.add_output(str(csv_path))
    mw.add_output(str(json_path))
    mw.set_counts({"n_pairs": len(all_results),
                   "n_train_strata": len(STRATA),
                   "n_test_strata": len(STRATA)})
    mw.write()

    print(f"\n{'='*65}")
    print(f"Results -> {csv_path}")

    # ---- Print summary F1 matrix ------------------------------------
    print("\nF1-MACRO SUMMARY (rows=train, cols=test):")
    for model_name in MODELS:
        print(f"\n  {model_name}:")
        print(f"  {'':12s} {'Test I':>10} {'Test II':>10} {'Test III':>10}")
        for train_s in STRATA:
            row_vals = []
            for test_s in STRATA:
                match = [r for r in all_results
                         if r["train_stratum"] == train_s.upper()
                         and r["test_stratum"] == test_s.upper()
                         and r["model"] == model_name]
                row_vals.append(f"{match[0]['f1_macro']:.4f}" if match else "N/A")
            print(f"  Train {train_s.upper():<6s}: {row_vals[0]:>10} "
                  f"{row_vals[1]:>10} {row_vals[2]:>10}")

    print("=" * 65)
    print("DONE.")
    print("=" * 65)


if __name__ == "__main__":
    main()
    