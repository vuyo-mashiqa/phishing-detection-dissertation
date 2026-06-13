"""
Phase 12 -- Explainability: Classical ML (SHAP)
Methods §1.11.5

For each (train_stratum, model_family, config) combination:
  - TreeSHAP  : Random Forest, XGBoost, LightGBM
  - LinearSHAP: Logistic Regression, Linear SVM

Outputs per (train_stratum):
  results/explanations/classical_ml/{train_stratum}/{model_family}/
    top10_aggregate.json      -- top-10 features by mean |SHAP| across 500 samples
    shap_values_sample.npz    -- SHAP values for the 500-sample subset (cols = features)
    beeswarm.png              -- beeswarm summary plot
    bar.png                   -- bar summary plot

Methods compliance:
  - 500 randomly sampled test instances per stratum  (§1.11.5)
  - Top-10 contributing features by absolute SHAP value per prediction (§1.11.5)
  - TreeSHAP for tree models; LinearSHAP for linear models  (§1.11.5)
  - Results saved to results/explanations/ (§1.11.5)
"""

import json
import sys
import random
from pathlib import Path

import joblib
import numpy as np
import scipy.sparse as sp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parents[2]
MODELS_DIR  = ROOT / "models" / "classical_ml"
REPR_DIR    = ROOT / "data" / "processed" / "representations"
SPLITS_DIR  = ROOT / "data" / "processed" / "splits"
OUT_DIR     = ROOT / "results" / "explanations" / "classical_ml"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Constants ────────────────────────────────────────────────────────────────
STRATA          = ["stratum_i", "stratum_ii", "stratum_iii"]
MODEL_FAMILIES  = ["logistic_regression", "linear_svm",
                   "random_forest", "xgboost", "lightgbm"]
TREE_FAMILIES   = {"random_forest", "xgboost", "lightgbm"}
LINEAR_FAMILIES = {"logistic_regression", "linear_svm"}
N_SAMPLE        = 500   # Methods §1.11.5
RANDOM_SEED     = 42

# ── Helpers ──────────────────────────────────────────────────────────────────

def load_feature_names(train_strat: str) -> list:
    path = REPR_DIR / train_strat / "feature_names.json"
    if not path.exists():
        print(f"  [XX] feature_names.json not found: {path}")
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as f:
        names = json.load(f)
    # Guarantee a flat Python list of strings
    if isinstance(names[0], list):
        names = [item for sublist in names for item in sublist]
    return [str(n) for n in names]


def load_test_matrix(train_strat: str) -> sp.csr_matrix:
    """Load the test-set feature matrix for a given training stratum."""
    npz_path = REPR_DIR / train_strat / f"test_{train_strat}_X.npz"
    if not npz_path.exists():
        print(f"  [XX] Test NPZ not found: {npz_path}")
        sys.exit(1)
    X = sp.load_npz(str(npz_path)).astype(np.float64)
    return X


def sample_rows(X: sp.csr_matrix, n: int, seed: int) -> sp.csr_matrix:
    rng = np.random.default_rng(seed)
    idx = rng.choice(X.shape[0], size=min(n, X.shape[0]), replace=False)
    return X[idx]


def align_features(X: sp.csr_matrix, n_feat: int) -> np.ndarray:
    """Truncate or pad X to match the model's n_features_in_, return dense float64."""
    if X.shape[1] > n_feat:
        X = X[:, :n_feat]
    elif X.shape[1] < n_feat:
        pad = sp.csr_matrix((X.shape[0], n_feat - X.shape[1]), dtype=np.float64)
        X = sp.hstack([X, pad])
    return X.toarray().astype(np.float64)


def get_model_n_features(model) -> int:
    """Extract n_features_in_ from the model or its pipeline steps."""
    if hasattr(model, "n_features_in_"):
        return int(model.n_features_in_)
    # Pipeline: check each step
    if hasattr(model, "steps"):
        for _, step in model.steps:
            if hasattr(step, "n_features_in_"):
                return int(step.n_features_in_)
    # XGBoost / LightGBM wrapped in Pipeline or SelectKBest
    if hasattr(model, "named_steps"):
        for name, step in model.named_steps.items():
            if hasattr(step, "n_features_in_"):
                return int(step.n_features_in_)
    raise ValueError(f"Cannot determine n_features_in_ from {type(model)}")


def get_base_estimator(model):
    """Unwrap Pipeline to get the final estimator for SHAP."""
    if hasattr(model, "steps"):
        return model.steps[-1][1]
    if hasattr(model, "named_steps"):
        steps = list(model.named_steps.values())
        return steps[-1]
    return model


def apply_pipeline_transform(model, X_dense: np.ndarray) -> np.ndarray:
    """
    If the model is a Pipeline, apply all steps EXCEPT the final estimator
    to transform X before passing to SHAP.
    Returns the transformed dense array.
    """
    if not (hasattr(model, "steps") or hasattr(model, "named_steps")):
        return X_dense

    steps = list(model.steps) if hasattr(model, "steps") else list(model.named_steps.items())

    X_t = X_dense
    for name, step in steps[:-1]:   # all but the final estimator
        X_t = step.transform(X_t)
        if sp.issparse(X_t):
            X_t = X_t.toarray().astype(np.float64)
    return X_t


def run_shap(model, family: str, X_sample: sp.csr_matrix,
             feat_names: list, out_dir: Path):
    """
    Compute SHAP values and write outputs for one (model, stratum) pair.
    Returns True on success, False on failure.
    """
    n_feat   = get_model_n_features(model)
    X_dense  = align_features(X_sample, n_feat)

    estimator = get_base_estimator(model)
    X_for_shap = apply_pipeline_transform(model, X_dense)

    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        if family in TREE_FAMILIES:
            explainer = shap.TreeExplainer(
                estimator,
                feature_perturbation="interventional",
                data=X_for_shap[:100],
            )
            shap_vals = explainer.shap_values(
                X_for_shap, check_additivity=False
            )
            # Normalise to 2D (samples × features) — phishing class only
            if isinstance(shap_vals, list):
                # list of arrays: [class0, class1]
                shap_arr = np.array(shap_vals[1], dtype=np.float64)
            else:
                shap_arr = np.array(shap_vals, dtype=np.float64)
            # 3D array (samples × features × classes) → take class 1
            if shap_arr.ndim == 3:
                shap_arr = shap_arr[:, :, 1]

        else:  # linear
            explainer = shap.LinearExplainer(estimator, X_for_shap)
            shap_vals = explainer.shap_values(X_for_shap)
            if isinstance(shap_vals, list):
                shap_arr = np.array(shap_vals[1], dtype=np.float64)
            else:
                shap_arr = np.array(shap_vals, dtype=np.float64)

    except Exception as e:
        print(f"    [XX] SHAP failed: {e}")
        return False

    # ── Align feature names to shap_arr width ────────────────────────────────
    n_shap_feats = int(shap_arr.shape[1])
    feat_names_aligned = [str(feat_names[i]) if i < len(feat_names) else f"feat_{i}"
                          for i in range(n_shap_feats)]

    # ── Top-10 aggregate (mean |SHAP|) ────────────────────────────────────────
    abs_mean = np.abs(shap_arr).mean(axis=0).flatten()
    top10_idx = [int(i) for i in np.argsort(abs_mean)[::-1][:10]]
    top10 = [
        {
            "rank": rank + 1,
            "feature_idx": idx,
            "feature_name": feat_names_aligned[idx],
            "mean_abs_shap": float(abs_mean[idx]),
        }
        for rank, idx in enumerate(top10_idx)
    ]

    with open(out_dir / "top10_aggregate.json", "w", encoding="utf-8") as f:
        json.dump(top10, f, indent=2, ensure_ascii=False)
    print(f"    [OK] top10_aggregate.json written")

    # ── Save raw SHAP array ───────────────────────────────────────────────────
    np.savez_compressed(
        str(out_dir / "shap_values_sample.npz"),
        shap_values=shap_arr.astype(np.float32),   # float32 saves ~50% space
    )
    print(f"    [OK] shap_values_sample.npz written  shape={shap_arr.shape}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    shap_exp = shap.Explanation(
        values=shap_arr,
        feature_names=feat_names_aligned,
    )

    # Beeswarm
    plt.figure(figsize=(10, 6))
    shap.plots.beeswarm(shap_exp, max_display=10, show=False)
    plt.tight_layout()
    plt.savefig(str(out_dir / "beeswarm.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    [OK] beeswarm.png written")

    # Bar
    plt.figure(figsize=(10, 6))
    shap.plots.bar(shap_exp, max_display=10, show=False)
    plt.tight_layout()
    plt.savefig(str(out_dir / "bar.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    [OK] bar.png written")

    return True


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Phase 12 -- Explainability: Classical ML (SHAP)")
    print("=" * 60)
    print(f"  shap version : {shap.__version__}")
    print(f"  N_SAMPLE     : {N_SAMPLE}")
    print()

    all_ok   = True
    summary  = []

    for train_strat in STRATA:
        print(f"\n{'-' * 60}")
        print(f"  Training stratum : {train_strat}")
        print(f"{'-' * 60}")

        feat_names = load_feature_names(train_strat)
        print(f"  Feature names loaded : {len(feat_names):,}")

        X_test = load_test_matrix(train_strat)
        print(f"  Test matrix shape    : {X_test.shape}")

        X_sample = sample_rows(X_test, N_SAMPLE, RANDOM_SEED)
        print(f"  Sample shape         : {X_sample.shape}")

        for family in MODEL_FAMILIES:
            model_dir = MODELS_DIR / train_strat / family
            model_path = model_dir / "model.pkl"

            if not model_path.exists():
                print(f"\n  [--] {family}: model.pkl not found at {model_path} — skipping")
                continue

            print(f"\n  [{family}]")
            model = joblib.load(str(model_path))
            print(f"    Loaded : {type(model).__name__}")

            out_dir = OUT_DIR / train_strat / family

            ok = run_shap(model, family, X_sample, feat_names, out_dir)
            summary.append({
                "train_stratum": train_strat,
                "model_family":  family,
                "status":        "OK" if ok else "FAILED",
            })
            if not ok:
                all_ok = False

    # ── Final summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for row in summary:
        mark = "[OK]" if row["status"] == "OK" else "[XX]"
        print(f"  {mark}  {row['train_stratum']:12s}  {row['model_family']}")

    print()
    if all_ok:
        print("PHASE 12A CLASSICAL ML SHAP COMPLETE.")
    else:
        print("PHASE 12A FAILED. Fix XX entries above before proceeding.")

    print("=" * 60)
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
