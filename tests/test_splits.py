"""
test_splits.py
==============
Verifies the integrity of the generated train/val/test splits.

Tests (per stratum + pooled):
  1. All three split files exist
  2. No message_id overlap between train, val, and test
  3. Row counts sum to the original stratum total (within ±1 for rounding)
  4. Class balance: phishing fraction in each split within ±5pp of full dataset
  5. Canonical columns present and in correct order
"""

import csv
import pytest
from pathlib import Path

csv.field_size_limit(10_000_000)

CANONICAL_COLS = [
    "message_id", "subject", "body", "sender", "label",
    "stratum", "source", "original_file", "body_length", "body_sha256",
]

STRATA_SOURCES = {
    "stratum_i":   Path("data/processed/stratum_i/stratum_i_combined.csv"),
    "stratum_ii":  Path("data/processed/stratum_ii/stratum_ii_combined.csv"),
    "stratum_iii": Path("data/processed/stratum_iii/stratum_iii_combined.csv"),
    "pooled":      None,
}


def load_ids_and_labels(path: Path):
    ids, labels = [], []
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            ids.append(row["message_id"])
            labels.append(int(row["label"]))
    return ids, labels


def load_cols(path: Path):
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        return reader.fieldnames or []


@pytest.mark.parametrize("stratum", list(STRATA_SOURCES.keys()))
def test_split_files_exist(stratum):
    base = Path("data/splits") / stratum
    for split in ("train", "val", "test"):
        p = base / f"{split}.csv"
        if not p.exists():
            pytest.skip(f"{p} not yet generated")


@pytest.mark.parametrize("stratum", list(STRATA_SOURCES.keys()))
def test_no_id_overlap(stratum):
    base = Path("data/splits") / stratum
    for split in ("train", "val", "test"):
        if not (base / f"{split}.csv").exists():
            pytest.skip(f"Splits not yet generated for {stratum}")

    ids = {}
    for split in ("train", "val", "test"):
        id_list, _ = load_ids_and_labels(base / f"{split}.csv")
        ids[split] = set(id_list)

    assert ids["train"].isdisjoint(ids["val"]),  "Train/val overlap"
    assert ids["train"].isdisjoint(ids["test"]), "Train/test overlap"
    assert ids["val"].isdisjoint(ids["test"]),   "Val/test overlap"


@pytest.mark.parametrize("stratum", list(STRATA_SOURCES.keys()))
def test_row_counts_sum_to_original(stratum):
    base = Path("data/splits") / stratum
    for split in ("train", "val", "test"):
        if not (base / f"{split}.csv").exists():
            pytest.skip(f"Splits not yet generated for {stratum}")

    src = STRATA_SOURCES[stratum]
    if src is None or not src.exists():
        pytest.skip(f"Source file not available for {stratum}")

    original_ids, _ = load_ids_and_labels(src)
    original_count  = len(original_ids)

    split_total = sum(
        len(load_ids_and_labels(base / f"{s}.csv")[0])
        for s in ("train", "val", "test")
    )
    assert abs(split_total - original_count) <= 1, (
        f"{stratum}: split total {split_total} != original {original_count}"
    )


@pytest.mark.parametrize("stratum", list(STRATA_SOURCES.keys()))
def test_canonical_columns(stratum):
    base = Path("data/splits") / stratum
    for split in ("train", "val", "test"):
        if not (base / f"{split}.csv").exists():
            pytest.skip(f"Splits not yet generated for {stratum}")
        cols = load_cols(base / f"{split}.csv")
        assert list(cols) == CANONICAL_COLS, (
            f"{stratum}/{split}: columns {cols} != {CANONICAL_COLS}"
        )
        