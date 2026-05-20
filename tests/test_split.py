"""
tests/test_split.py
===================
Integrity tests for the 70/15/15 splits produced by split_dataset.py.
  (i)  zero message_id overlap across partitions
  (ii) row counts sum to the source stratum total
  (iii) canonical schema conformance in all split files

Run:
    pytest tests/test_split.py -v
"""

import hashlib
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.stdout.reconfigure(encoding="utf-8")

# ── Paths ──────────────────────────────────────────────────────────────
OUT_DIR    = Path("data/processed/splits")
FEAT_I     = Path("data/processed/features/features_stratum_i.csv")
FEAT_II    = Path("data/processed/features/features_stratum_ii.csv")
FEAT_III   = Path("data/processed/features/features_stratum_iii.csv")
FEAT_COMB  = Path("data/processed/features/features_combined.csv")

STRATA = [
    ("I",   "stratum_i",   FEAT_I,   155_250, 108_675, 23_287, 23_288),
    ("II",  "stratum_ii",  FEAT_II,    7_846,   5_492,  1_177,  1_177),
    ("III", "stratum_iii", FEAT_III,  13_356,   9_349,  2_003,  2_004),
]
POOLED_COUNTS = (176_452, 123_516, 26_468, 26_468)

EXPECTED_COLS = [
    "message_id", "label", "stratum", "source",
    "html_char_ratio", "reply_to_mismatch",
    "body_length", "subject_length",
    "body_unique_word_ratio", "body_capitalisation_ratio",
    "body_digit_ratio", "body_special_char_ratio",
    "url_count", "unique_domain_count",
    "has_ip_url", "url_domain_mismatch",
    "urgency_word_count", "exclamation_count", "question_count",
]


# ── Fixtures ───────────────────────────────────────────────────────────
def load_split(suffix: str, part: str) -> pd.DataFrame:
    return pd.read_csv(OUT_DIR / f"{part}_{suffix}.csv", low_memory=False)

@pytest.fixture(scope="module")
def splits_i():
    return (load_split("stratum_i", "train"),
            load_split("stratum_i", "val"),
            load_split("stratum_i", "test"))

@pytest.fixture(scope="module")
def splits_ii():
    return (load_split("stratum_ii", "train"),
            load_split("stratum_ii", "val"),
            load_split("stratum_ii", "test"))

@pytest.fixture(scope="module")
def splits_iii():
    return (load_split("stratum_iii", "train"),
            load_split("stratum_iii", "val"),
            load_split("stratum_iii", "test"))

@pytest.fixture(scope="module")
def splits_pooled():
    return (load_split("pooled", "train"),
            load_split("pooled", "val"),
            load_split("pooled", "test"))

@pytest.fixture(scope="module")
def feat_i():   return pd.read_csv(FEAT_I,    low_memory=False)

@pytest.fixture(scope="module")
def feat_ii():  return pd.read_csv(FEAT_II,   low_memory=False)

@pytest.fixture(scope="module")
def feat_iii(): return pd.read_csv(FEAT_III,  low_memory=False)

@pytest.fixture(scope="module")
def feat_comb(): return pd.read_csv(FEAT_COMB, low_memory=False)


# ── T1: All 12 split files exist ──────────────────────────────────────
@pytest.mark.parametrize("fname", [
    "train_stratum_i.csv",   "val_stratum_i.csv",   "test_stratum_i.csv",
    "train_stratum_ii.csv",  "val_stratum_ii.csv",  "test_stratum_ii.csv",
    "train_stratum_iii.csv", "val_stratum_iii.csv", "test_stratum_iii.csv",
    "train_pooled.csv",      "val_pooled.csv",      "test_pooled.csv",
    "split_report.txt",
])
def test_file_exists(fname):
    assert (OUT_DIR / fname).exists(), f"Missing: {OUT_DIR / fname}"


# ── T2: Column schema in every split file ─────────────────────────────
@pytest.mark.parametrize("suffix,part", [
    ("stratum_i",   "train"), ("stratum_i",   "val"), ("stratum_i",   "test"),
    ("stratum_ii",  "train"), ("stratum_ii",  "val"), ("stratum_ii",  "test"),
    ("stratum_iii", "train"), ("stratum_iii", "val"), ("stratum_iii", "test"),
    ("pooled",      "train"), ("pooled",      "val"), ("pooled",      "test"),
])
def test_column_schema(suffix, part):
    df = load_split(suffix, part)
    assert list(df.columns) == EXPECTED_COLS, (
        f"{part}_{suffix}: columns mismatch\n"
        f"  expected: {EXPECTED_COLS}\n"
        f"  got:      {list(df.columns)}"
    )


# ── T3: Row counts match Methods §1.9 table ───────────────────────────
def test_row_counts_stratum_i(splits_i):
    tr, va, te = splits_i
    assert len(tr) == 108_675, f"Stratum I train: {len(tr)}"
    assert len(va) ==  23_287, f"Stratum I val:   {len(va)}"
    assert len(te) ==  23_288, f"Stratum I test:  {len(te)}"

def test_row_counts_stratum_ii(splits_ii):
    tr, va, te = splits_ii
    assert len(tr) == 5_492, f"Stratum II train: {len(tr)}"
    assert len(va) == 1_177, f"Stratum II val:   {len(va)}"
    assert len(te) == 1_177, f"Stratum II test:  {len(te)}"

def test_row_counts_stratum_iii(splits_iii):
    tr, va, te = splits_iii
    assert len(tr) == 9_349, f"Stratum III train: {len(tr)}"
    assert len(va) == 2_003, f"Stratum III val:   {len(va)}"
    assert len(te) == 2_004, f"Stratum III test:  {len(te)}"

def test_row_counts_pooled(splits_pooled):
    tr, va, te = splits_pooled
    assert len(tr) == 123_516, f"Pooled train: {len(tr)}"
    assert len(va) ==  26_468, f"Pooled val:   {len(va)}"
    assert len(te) ==  26_468, f"Pooled test:  {len(te)}"


# ── T4: Row counts sum to source total (Methods §1.9 condition ii) ────
@pytest.mark.parametrize("fixture,total", [
    ("splits_i",      155_250),
    ("splits_ii",       7_846),
    ("splits_iii",     13_356),
    ("splits_pooled", 176_452),
])
def test_row_counts_sum_to_total(fixture, total, request):
    tr, va, te = request.getfixturevalue(fixture)
    got = len(tr) + len(va) + len(te)
    assert got == total, f"{fixture}: sum {got} != expected {total}"


# ── T5: Zero message_id overlap across all three partitions ───────────
#    (Methods §1.9 condition i)
@pytest.mark.parametrize("fixture", [
    "splits_i", "splits_ii", "splits_iii", "splits_pooled"
])
def test_no_train_val_overlap(fixture, request):
    tr, va, _ = request.getfixturevalue(fixture)
    overlap = set(tr["message_id"]) & set(va["message_id"])
    assert len(overlap) == 0, f"{fixture}: {len(overlap)} train/val overlaps"

@pytest.mark.parametrize("fixture", [
    "splits_i", "splits_ii", "splits_iii", "splits_pooled"
])
def test_no_train_test_overlap(fixture, request):
    tr, _, te = request.getfixturevalue(fixture)
    overlap = set(tr["message_id"]) & set(te["message_id"])
    assert len(overlap) == 0, f"{fixture}: {len(overlap)} train/test overlaps"

@pytest.mark.parametrize("fixture", [
    "splits_i", "splits_ii", "splits_iii", "splits_pooled"
])
def test_no_val_test_overlap(fixture, request):
    _, va, te = request.getfixturevalue(fixture)
    overlap = set(va["message_id"]) & set(te["message_id"])
    assert len(overlap) == 0, f"{fixture}: {len(overlap)} val/test overlaps"


# ── T6: message_id coverage — union equals source feature file ─────────
def test_coverage_stratum_i(splits_i, feat_i):
    tr, va, te = splits_i
    split_ids  = set(tr["message_id"]) | set(va["message_id"]) | set(te["message_id"])
    source_ids = set(feat_i["message_id"])
    assert split_ids == source_ids, (
        f"Stratum I: split covers {len(split_ids)} IDs, source has {len(source_ids)}"
    )

def test_coverage_stratum_ii(splits_ii, feat_ii):
    tr, va, te = splits_ii
    split_ids  = set(tr["message_id"]) | set(va["message_id"]) | set(te["message_id"])
    source_ids = set(feat_ii["message_id"])
    assert split_ids == source_ids

def test_coverage_stratum_iii(splits_iii, feat_iii):
    tr, va, te = splits_iii
    split_ids  = set(tr["message_id"]) | set(va["message_id"]) | set(te["message_id"])
    source_ids = set(feat_iii["message_id"])
    assert split_ids == source_ids

def test_coverage_pooled(splits_pooled, feat_comb):
    tr, va, te = splits_pooled
    split_ids  = set(tr["message_id"]) | set(va["message_id"]) | set(te["message_id"])
    source_ids = set(feat_comb["message_id"])
    assert split_ids == source_ids


# ── T7: No nulls in any split file ────────────────────────────────────
@pytest.mark.parametrize("suffix,part", [
    ("stratum_i",   "train"), ("stratum_i",   "val"), ("stratum_i",   "test"),
    ("stratum_ii",  "train"), ("stratum_ii",  "val"), ("stratum_ii",  "test"),
    ("stratum_iii", "train"), ("stratum_iii", "val"), ("stratum_iii", "test"),
    ("pooled",      "train"), ("pooled",      "val"), ("pooled",      "test"),
])
def test_no_nulls(suffix, part):
    df = load_split(suffix, part)
    nulls = df.isnull().sum()
    assert not nulls.any(), (
        f"{part}_{suffix}: nulls found:\n{nulls[nulls > 0]}"
    )


# ── T8: Label is binary {0, 1} in every split ─────────────────────────
@pytest.mark.parametrize("suffix,part", [
    ("stratum_i",   "train"), ("stratum_i",   "val"), ("stratum_i",   "test"),
    ("stratum_ii",  "train"), ("stratum_ii",  "val"), ("stratum_ii",  "test"),
    ("stratum_iii", "train"), ("stratum_iii", "val"), ("stratum_iii", "test"),
    ("pooled",      "train"), ("pooled",      "val"), ("pooled",      "test"),
])
def test_label_binary(suffix, part):
    df = load_split(suffix, part)
    invalid = set(df["label"].unique()) - {0, 1}
    assert not invalid, f"{part}_{suffix}: unexpected labels: {invalid}"


# ── T9: Label distribution preserved within 1% per stratum ────────────
@pytest.mark.parametrize("fixture,label", [
    ("splits_i",   "Stratum I"),
    ("splits_ii",  "Stratum II"),
    ("splits_iii", "Stratum III"),
    ("splits_pooled", "Pooled"),
])
def test_label_stratification(fixture, label, request):
    tr, va, te = request.getfixturevalue(fixture)
    full = pd.concat([tr, va, te])
    for lbl in [0, 1]:
        base_rate = (full["label"] == lbl).mean()
        for part_name, part in [("train", tr), ("val", va), ("test", te)]:
            rate  = (part["label"] == lbl).mean()
            drift = abs(rate - base_rate)
            assert drift < 0.01, (
                f"{label} {part_name}: label {lbl} drift {drift:.4f} > 1%"
            )


# ── T10: Stale 80/20 files must not exist ─────────────────────────────
@pytest.mark.parametrize("stale_file", ["train.csv", "test.csv"])
def test_no_stale_8020_files(stale_file):
    stale = OUT_DIR / stale_file
    assert not stale.exists(), (
        f"Stale 80/20 split file still present: {stale}. "
        "Delete before proceeding."
    )


# ── T11: split_report.txt exists and mentions all 12 split files ──────
def test_split_report_content():
    report = (OUT_DIR / "split_report.txt").read_text(encoding="utf-8")
    for fname in [
        "train_stratum_i",   "val_stratum_i",   "test_stratum_i",
        "train_stratum_ii",  "val_stratum_ii",  "test_stratum_ii",
        "train_stratum_iii", "val_stratum_iii", "test_stratum_iii",
        "train_pooled",      "val_pooled",      "test_pooled",
    ]:
        assert fname in report, f"split_report.txt missing: {fname}"
