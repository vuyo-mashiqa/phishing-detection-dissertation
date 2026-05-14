"""
test_classical_ml.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT        = Path(__file__).resolve().parents[1]
MODELS_DIR  = ROOT / "models" / "classical_ml"
RESULTS_DIR = ROOT / "results" / "classical_ml"

CONFIGS     = ["stratum_i", "stratum_ii", "stratum_iii", "pooled"]
STRATA      = ["stratum_i", "stratum_ii", "stratum_iii"]
MODEL_NAMES = ["logistic_regression", "linear_svm", "random_forest", "xgboost", "lightgbm"]

REQUIRED_METRICS = [
    "f1_macro", "precision", "recall", "roc_auc", "pr_auc",
    "balanced_accuracy", "mcc", "fpr_at_95tpr", "youden_j", "threshold",
]
REQUIRED_LATENCY = ["latency_p50_ms", "latency_p95_ms", "latency_p99_ms"]

# Sanity floors — deliberately conservative to accommodate known edge cases
# (e.g. LinearSVM Stratum I degenerate threshold)
F1_FLOOR     = 0.50
ROCAUC_FLOOR = 0.70


@pytest.fixture(scope="module", params=[
    (c, m) for c in CONFIGS for m in MODEL_NAMES
])
def run(request):
    return request.param  # (config, model_name)


class TestArtefactsExist:
    def test_manifest_exists(self, run):
        config, model = run
        assert (MODELS_DIR / config / model / "manifest.json").exists()

    def test_model_pkl_exists(self, run):
        config, model = run
        assert (MODELS_DIR / config / model / "model.pkl").exists()

    def test_threshold_json_exists(self, run):
        config, model = run
        assert (MODELS_DIR / config / model / "threshold.json").exists()

    def test_results_json_exists(self, run):
        config, model = run
        assert (MODELS_DIR / config / model / "results.json").exists()


class TestResultsContent:
    def test_all_metrics_present(self, run):
        config, model = run
        with open(MODELS_DIR / config / model / "results.json") as f:
            r = json.load(f)
        for metric in REQUIRED_METRICS:
            assert metric in r["test_metrics"], (
                f"{config}/{model}: missing metric '{metric}' in test_metrics"
            )

    def test_latency_fields_present(self, run):
        config, model = run
        with open(MODELS_DIR / config / model / "results.json") as f:
            r = json.load(f)
        for field in REQUIRED_LATENCY:
            assert field in r["latency"], (
                f"{config}/{model}: missing latency field '{field}'"
            )

    def test_cross_stratum_all_strata_present(self, run):
        config, model = run
        with open(MODELS_DIR / config / model / "results.json") as f:
            r = json.load(f)
        for stratum in STRATA:
            assert stratum in r["cross_stratum"], (
                f"{config}/{model}: missing cross-stratum eval for '{stratum}'"
            )

    def test_cross_stratum_metrics_complete(self, run):
        config, model = run
        with open(MODELS_DIR / config / model / "results.json") as f:
            r = json.load(f)
        for stratum in STRATA:
            for metric in REQUIRED_METRICS:
                assert metric in r["cross_stratum"][stratum], (
                    f"{config}/{model}/cross[{stratum}]: missing '{metric}'"
                )


class TestSanityBounds:
    def test_matched_f1_above_floor(self, run):
        config, model = run
        with open(MODELS_DIR / config / model / "results.json") as f:
            r = json.load(f)
        f1 = r["test_metrics"]["f1_macro"]
        assert f1 >= F1_FLOOR, (
            f"{config}/{model}: matched test F1={f1:.4f} below floor {F1_FLOOR}"
        )

    def test_matched_rocauc_above_floor(self, run):
        config, model = run
        with open(MODELS_DIR / config / model / "results.json") as f:
            r = json.load(f)
        auc = r["test_metrics"]["roc_auc"]
        assert auc >= ROCAUC_FLOOR, (
            f"{config}/{model}: matched ROC-AUC={auc:.4f} below floor {ROCAUC_FLOOR}"
        )

    def test_threshold_in_unit_interval(self, run):
        config, model = run
        with open(MODELS_DIR / config / model / "threshold.json") as f:
            t = json.load(f)
        assert 0.0 <= t["threshold"] <= 1.0, (
            f"{config}/{model}: threshold {t['threshold']} outside [0,1]"
        )

    def test_latency_p50_under_100ms(self, run):
        config, model = run
        with open(MODELS_DIR / config / model / "results.json") as f:
            r = json.load(f)
        p50 = r["latency"]["latency_p50_ms"]
        assert p50 < 100.0, (
            f"{config}/{model}: p50 latency={p50:.2f}ms exceeds 100ms"
        )


class TestManifestContent:
    def test_manifest_has_required_fields(self, run):
        config, model = run
        with open(MODELS_DIR / config / model / "manifest.json") as f:
            m = json.load(f)
        for field in ["config", "model", "git_sha", "timestamp",
                      "train_time_s", "n_train", "scale_pos_weight",
                      "t_star", "train_npz_sha256", "hyperparameters"]:
            assert field in m, f"{config}/{model}: manifest missing '{field}'"

    def test_manifest_scale_pos_weight_positive(self, run):
        config, model = run
        with open(MODELS_DIR / config / model / "manifest.json") as f:
            m = json.load(f)
        assert m["scale_pos_weight"] > 0.0


class TestAllResultsCSV:
    def test_csv_exists(self):
        assert (RESULTS_DIR / "all_results.csv").exists()

    def test_csv_row_count(self):
        df = pd.read_csv(RESULTS_DIR / "all_results.csv")
        # 20 models × (1 matched + 3 cross-stratum) = 80 rows
        assert len(df) == 80, f"Expected 80 rows, got {len(df)}"

    def test_csv_required_columns(self):
        df = pd.read_csv(RESULTS_DIR / "all_results.csv")
        for col in ["train_config", "model", "eval_stratum",
                    "f1_macro", "roc_auc", "pr_auc", "fpr_at_95tpr",
                    "latency_p50_ms"]:
            assert col in df.columns, f"all_results.csv missing column '{col}'"

    def test_csv_no_null_f1(self):
        df = pd.read_csv(RESULTS_DIR / "all_results.csv")
        assert df["f1_macro"].notna().all(), "Null F1 values in all_results.csv"

    def test_csv_all_models_present(self):
        df = pd.read_csv(RESULTS_DIR / "all_results.csv")
        for model in MODEL_NAMES:
            assert model in df["model"].values, f"Model '{model}' missing from CSV"

    def test_csv_all_configs_present(self):
        df = pd.read_csv(RESULTS_DIR / "all_results.csv")
        for config in CONFIGS:
            assert config in df["train_config"].values, (
                f"Config '{config}' missing from CSV"
            )
