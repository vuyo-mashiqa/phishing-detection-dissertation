"""test_representation.py

Validates the hybrid representation matrices for all 4 configs.
Tests enforce:
  - Correct feature dimensionality (word + char + 7 structural)
  - No data leakage (val/test transformed with training vocabulary only)
  - Label alignment with split files
  - Manifest integrity
"""

import json
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[1]
FEATURES_DIR = ROOT / "outputs" / "features"
SPLITS_DIR = ROOT / "data" / "processed" / "splits"

CONFIGS = ["stratum_i", "stratum_ii", "stratum_iii", "pooled"]
PARTITIONS = ["train", "val", "test"]

# Expected split row counts (Methods §1.9)
EXPECTED_COUNTS = {
    "stratum_i":  {"train": 108_675, "val": 23_287, "test": 23_288},
    "stratum_ii": {"train":   5_492, "val":  1_177, "test":  1_177},
    "stratum_iii":{"train":   9_349, "val":  2_003, "test":  2_004},
    "pooled":     {"train": 123_516, "val": 26_468, "test": 26_468},
}

N_STRUCTURAL = 7


@pytest.fixture(scope="module", params=CONFIGS)
def config(request):
    return request.param


class TestManifest:
    def test_manifest_exists(self, config):
        assert (FEATURES_DIR / config / "manifest.json").exists()

    def test_manifest_fields(self, config):
        with open(FEATURES_DIR / config / "manifest.json") as f:
            m = json.load(f)
        for field in ["config", "git_sha", "n_train", "n_val", "n_test",
                      "n_features", "n_word_features", "n_char_features",
                      "n_structural_features", "structural_col_names"]:
            assert field in m, f"Missing field: {field}"
        assert m["n_structural_features"] == N_STRUCTURAL
        assert len(m["structural_col_names"]) == N_STRUCTURAL

    def test_manifest_feature_dim_consistency(self, config):
        with open(FEATURES_DIR / config / "manifest.json") as f:
            m = json.load(f)
        assert m["n_features"] == m["n_word_features"] + m["n_char_features"] + N_STRUCTURAL


class TestMatrixShapes:
    @pytest.mark.parametrize("partition", PARTITIONS)
    def test_matrix_exists(self, config, partition):
        assert (FEATURES_DIR / config / f"{partition}.npz").exists()

    @pytest.mark.parametrize("partition", PARTITIONS)
    def test_row_counts(self, config, partition):
        X = sp.load_npz(FEATURES_DIR / config / f"{partition}.npz")
        expected = EXPECTED_COUNTS[config][partition]
        assert X.shape[0] == expected, (
            f"{config}/{partition}: expected {expected} rows, got {X.shape[0]}"
        )

    def test_feature_dim_consistent_across_partitions(self, config):
        dims = set()
        for partition in PARTITIONS:
            X = sp.load_npz(FEATURES_DIR / config / f"{partition}.npz")
            dims.add(X.shape[1])
        assert len(dims) == 1, f"{config}: inconsistent feature dims across partitions: {dims}"

    def test_char_features_present(self, config):
        with open(FEATURES_DIR / config / "manifest.json") as f:
            m = json.load(f)
        assert m["n_char_features"] > 0, "Character n-gram TF-IDF produced zero features"

    def test_word_features_present(self, config):
        with open(FEATURES_DIR / config / "manifest.json") as f:
            m = json.load(f)
        assert m["n_word_features"] > 0, "Word TF-IDF produced zero features"


class TestLabelAlignment:
    @pytest.mark.parametrize("partition", PARTITIONS)
    def test_label_count_matches_matrix_rows(self, config, partition):
        X = sp.load_npz(FEATURES_DIR / config / f"{partition}.npz")
        y = np.load(FEATURES_DIR / config / f"y_{partition}.npy")
        assert X.shape[0] == len(y), (
            f"{config}/{partition}: matrix rows={X.shape[0]} != label count={len(y)}"
        )

    @pytest.mark.parametrize("partition", PARTITIONS)
    def test_labels_are_binary(self, config, partition):
        y = np.load(FEATURES_DIR / config / f"y_{partition}.npy")
        assert set(np.unique(y)).issubset({0, 1}), f"{config}/{partition}: non-binary labels"


class TestNoDataLeakage:
    def test_vocab_size_word_consistent(self, config):
        """Val and test word features must match training vocabulary size."""
        with open(FEATURES_DIR / config / "manifest.json") as f:
            m = json.load(f)
        n_word = m["n_word_features"]
        for partition in ["val", "test"]:
            X = sp.load_npz(FEATURES_DIR / config / f"{partition}.npz")
            # Word features occupy columns 0..n_word-1
            # Confirm val/test have same total dim as train (no re-fitting)
            assert X.shape[1] == m["n_features"], (
                f"{config}/{partition}: feature dim mismatch indicates possible re-fit"
            )