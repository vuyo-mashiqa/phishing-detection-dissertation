"""
tests/test_features.py
======================
Schema and integrity tests for the engineered feature files produced by
build_features.py (Step 7.2).

Run:
    pytest tests/test_features.py -v
"""

import pandas as pd
import pytest
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────
FEAT_I        = Path("data/processed/features/features_stratum_i.csv")
FEAT_II       = Path("data/processed/features/features_stratum_ii.csv")
FEAT_III      = Path("data/processed/features/features_stratum_iii.csv")
FEAT_COMBINED = Path("data/processed/features/features_combined.csv")

CANON_I       = Path("data/processed/stratum_i/stratum_i_combined.csv")
CANON_II      = Path("data/processed/stratum_ii/stratum_ii_combined.csv")
CANON_III     = Path("data/processed/stratum_iii/stratum_iii_combined.csv")

EXPECTED_COLS = [
    "message_id", "label", "stratum", "source",
    "html_char_ratio", "reply_to_mismatch",
    "body_length", "subject_length",
    "body_unique_word_ratio", "body_capitalisation_ratio",
    "body_digit_ratio", "body_special_char_ratio",
    "url_count", "unique_domain_count",
    "has_ip_url", "url_domain_mismatch",
    "urgency_word_count",
    "exclamation_count", "question_count",
]

FLOAT_RATIO_COLS = [
    "html_char_ratio",
    "body_unique_word_ratio",
    "body_capitalisation_ratio",
    "body_digit_ratio",
    "body_special_char_ratio",
]

BINARY_COLS = [
    "reply_to_mismatch",
    "has_ip_url",
    "url_domain_mismatch",
]

NON_NEGATIVE_INT_COLS = [
    "body_length", "subject_length",
    "url_count", "unique_domain_count",
    "urgency_word_count",
    "exclamation_count", "question_count",
]


# ── Fixtures ──────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def feat_i():
    return pd.read_csv(FEAT_I, low_memory=False)

@pytest.fixture(scope="module")
def feat_ii():
    return pd.read_csv(FEAT_II, low_memory=False)

@pytest.fixture(scope="module")
def feat_iii():
    return pd.read_csv(FEAT_III, low_memory=False)

@pytest.fixture(scope="module")
def combined():
    return pd.read_csv(FEAT_COMBINED, low_memory=False)

@pytest.fixture(scope="module")
def canon_i():
    return pd.read_csv(CANON_I, dtype=str, low_memory=False)

@pytest.fixture(scope="module")
def canon_ii():
    return pd.read_csv(CANON_II, dtype=str, low_memory=False)

@pytest.fixture(scope="module")
def canon_iii():
    return pd.read_csv(CANON_III, dtype=str, low_memory=False)


# ── T1: Files exist ───────────────────────────────────────────────────
@pytest.mark.parametrize("path", [FEAT_I, FEAT_II, FEAT_III, FEAT_COMBINED])
def test_file_exists(path):
    assert path.exists(), f"Missing: {path}"


# ── T2: Column set and order ──────────────────────────────────────────
@pytest.mark.parametrize("df_fixture", ["feat_i", "feat_ii", "feat_iii", "combined"])
def test_columns(df_fixture, request):
    df = request.getfixturevalue(df_fixture)
    assert list(df.columns) == EXPECTED_COLS, (
        f"{df_fixture}: columns mismatch\n"
        f"  expected : {EXPECTED_COLS}\n"
        f"  got      : {list(df.columns)}"
    )


# ── T3: Row counts match canonical CSVs ──────────────────────────────
def test_row_count_stratum_i(feat_i, canon_i):
    assert len(feat_i) == len(canon_i), (
        f"Stratum I row count: features={len(feat_i)} canonical={len(canon_i)}"
    )

def test_row_count_stratum_ii(feat_ii, canon_ii):
    assert len(feat_ii) == len(canon_ii), (
        f"Stratum II row count: features={len(feat_ii)} canonical={len(canon_ii)}"
    )

def test_row_count_stratum_iii(feat_iii, canon_iii):
    assert len(feat_iii) == len(canon_iii), (
        f"Stratum III row count: features={len(feat_iii)} canonical={len(canon_iii)}"
    )

def test_combined_row_count(combined, feat_i, feat_ii, feat_iii):
    expected = len(feat_i) + len(feat_ii) + len(feat_iii)
    assert len(combined) == expected, (
        f"Combined rows: {len(combined)} != {expected} (sum of strata)"
    )


# ── T4: No nulls in any column ────────────────────────────────────────
@pytest.mark.parametrize("df_fixture", ["feat_i", "feat_ii", "feat_iii", "combined"])
def test_no_nulls(df_fixture, request):
    df = request.getfixturevalue(df_fixture)
    null_counts = df.isnull().sum()
    assert not null_counts.any(), (
        f"{df_fixture}: null values found:\n{null_counts[null_counts > 0]}"
    )


# ── T5: message_id is unique within each stratum ──────────────────────
@pytest.mark.parametrize("df_fixture", ["feat_i", "feat_ii", "feat_iii"])
def test_message_id_unique_per_stratum(df_fixture, request):
    df = request.getfixturevalue(df_fixture)
    dupes = df["message_id"].duplicated().sum()
    assert dupes == 0, f"{df_fixture}: {dupes} duplicate message_ids"


# ── T6: message_id is unique in combined ──────────────────────────────
def test_message_id_unique_combined(combined):
    dupes = combined["message_id"].duplicated().sum()
    assert dupes == 0, f"combined: {dupes} duplicate message_ids"


# ── T7: label is binary {0, 1} ───────────────────────────────────────
@pytest.mark.parametrize("df_fixture", ["feat_i", "feat_ii", "feat_iii", "combined"])
def test_label_binary(df_fixture, request):
    df = request.getfixturevalue(df_fixture)
    invalid = set(df["label"].unique()) - {0, 1}
    assert not invalid, f"{df_fixture}: unexpected label values: {invalid}"


# ── T8: float ratio columns in [0.0, 1.0] ────────────────────────────
@pytest.mark.parametrize("col", FLOAT_RATIO_COLS)
@pytest.mark.parametrize("df_fixture", ["combined"])
def test_float_ratios_bounded(col, df_fixture, request):
    df = request.getfixturevalue(df_fixture)
    out_of_range = df[(df[col] < 0.0) | (df[col] > 1.0)]
    assert len(out_of_range) == 0, (
        f"{col}: {len(out_of_range)} values outside [0, 1]"
    )


# ── T9: binary columns contain only {0, 1} ───────────────────────────
@pytest.mark.parametrize("col", BINARY_COLS)
def test_binary_columns(col, combined):
    invalid = set(combined[col].unique()) - {0, 1}
    assert not invalid, f"{col}: unexpected values: {invalid}"


# ── T10: non-negative integer columns ────────────────────────────────
@pytest.mark.parametrize("col", NON_NEGATIVE_INT_COLS)
def test_non_negative_int_cols(col, combined):
    negative = (combined[col] < 0).sum()
    assert negative == 0, f"{col}: {negative} negative values"


# ── T11: label distribution sanity checks ────────────────────────────
def test_label_dist_stratum_i(feat_i):
    dist = feat_i["label"].value_counts().to_dict()
    assert dist.get(0, 0) == 152_977, f"Stratum I ham count: {dist.get(0)}"
    assert dist.get(1, 0) == 2_273,   f"Stratum I phishing count: {dist.get(1)}"

def test_label_dist_stratum_ii(feat_ii):
    dist = feat_ii["label"].value_counts().to_dict()
    assert dist.get(0, 0) == 2_763, f"Stratum II ham count: {dist.get(0)}"
    assert dist.get(1, 0) == 5_083, f"Stratum II phishing count: {dist.get(1)}"

def test_label_dist_stratum_iii(feat_iii):
    dist = feat_iii["label"].value_counts().to_dict()
    assert dist.get(0, 0) == 6_600, f"Stratum III ham count: {dist.get(0)}"
    assert dist.get(1, 0) == 6_756, f"Stratum III phishing count: {dist.get(1)}"

def test_label_dist_combined(combined):
    dist = combined["label"].value_counts().to_dict()
    assert dist.get(0, 0) == 162_340, f"Combined ham: {dist.get(0)}"
    assert dist.get(1, 0) == 14_112,  f"Combined phishing: {dist.get(1)}"


# ── T12: stratum values are valid ────────────────────────────────────
def test_stratum_values(combined):
    invalid = set(combined["stratum"].unique()) - {"I", "II", "III"}
    assert not invalid, f"Unexpected stratum values: {invalid}"


# ── T13: message_id sets match canonical CSVs ────────────────────────
def test_message_ids_match_canonical_i(feat_i, canon_i):
    assert set(feat_i["message_id"]) == set(canon_i["message_id"]), (
        "Stratum I: message_id set mismatch between features and canonical CSV"
    )

def test_message_ids_match_canonical_ii(feat_ii, canon_ii):
    assert set(feat_ii["message_id"]) == set(canon_ii["message_id"]), (
        "Stratum II: message_id set mismatch between features and canonical CSV"
    )

def test_message_ids_match_canonical_iii(feat_iii, canon_iii):
    assert set(feat_iii["message_id"]) == set(canon_iii["message_id"]), (
        "Stratum III: message_id set mismatch between features and canonical CSV"
    )


# ── T14: structural feature defaults are not universal ───────────────
# If ALL html_char_ratio values are 0.0 the join silently failed.
def test_html_char_ratio_not_all_zero(combined):
    nonzero = (combined["html_char_ratio"] > 0).sum()
    assert nonzero > 0, (
        "html_char_ratio is 0 for every row — structural join likely failed"
    )

def test_reply_to_mismatch_not_all_zero(combined):
    nonzero = (combined["reply_to_mismatch"] == 1).sum()
    assert nonzero > 0, (
        "reply_to_mismatch is 0 for every row — structural join likely failed"
    )
    