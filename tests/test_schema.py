"""
tests/test_schema.py

Automated schema validation for all canonical processed CSV files.
Enforces SCHEMA_CANONICAL.md

Usage:
    pytest tests/test_schema.py -v

All tests skip gracefully if a canonical file does not yet exist.
"""

import pytest
import pandas as pd
from pathlib import Path

# ------------------------------------------------------------------ #
#  SCHEMA CONSTANTS — must exactly match SCHEMA_CANONICAL.md          #
# ------------------------------------------------------------------ #

CANONICAL_COLS = [
    "message_id",
    "subject",
    "body",
    "sender",
    "label",
    "stratum",
    "source",
    "original_file",
    "body_length",
    "body_sha256",
]

VALID_LABELS   = {0, 1}
VALID_STRATA   = {"I", "II", "III"}
VALID_SOURCES  = {"enron", "nazario", "phishing_pot", "csdmc2010", "phishfuzzer"}

CANONICAL_FILES = [
    "data/processed/stratum_i/stratum_i_combined.csv",
    "data/processed/stratum_ii/stratum_ii_combined.csv",
    "data/processed/stratum_iii/stratum_iii_combined.csv",
]

# ------------------------------------------------------------------ #
#  HELPER                                                             #
# ------------------------------------------------------------------ #

def load_or_skip(filepath):
    """Load CSV or skip test if file does not yet exist."""
    if not Path(filepath).exists():
        pytest.skip(f"File not yet produced: {filepath}")
    return pd.read_csv(filepath, low_memory=False)


# ------------------------------------------------------------------ #
#  TEST 1 — File exists                                               #
# ------------------------------------------------------------------ #

@pytest.mark.parametrize("filepath", CANONICAL_FILES)
def test_file_exists(filepath):
    """Canonical file must exist before downstream steps can proceed."""
    if not Path(filepath).exists():
        pytest.skip(f"Not yet produced: {filepath}")
    assert Path(filepath).is_file()


# ------------------------------------------------------------------ #
#  TEST 2 — Column names and order                                    #
# ------------------------------------------------------------------ #

@pytest.mark.parametrize("filepath", CANONICAL_FILES)
def test_columns_names_and_order(filepath):
    """
    Column names must exactly match CANONICAL_COLS in exactly that order.
    A file with correct names but wrong order still fails — downstream
    code depends on positional consistency.
    """
    df = load_or_skip(filepath)
    actual = list(df.columns)
    assert actual == CANONICAL_COLS, (
        f"\nColumn mismatch in {filepath}"
        f"\n  Expected: {CANONICAL_COLS}"
        f"\n  Got:      {actual}"
    )


# ------------------------------------------------------------------ #
#  TEST 3 — No nulls in required columns                              #
# ------------------------------------------------------------------ #

@pytest.mark.parametrize("filepath", CANONICAL_FILES)
def test_no_nulls_in_required_columns(filepath):
    """
    These columns must never contain null (NaN) values.
    subject and sender may be empty strings but not null — tested separately.
    """
    df = load_or_skip(filepath)
    required_never_null = [
        "message_id", "body", "label", "stratum",
        "source", "original_file", "body_length", "body_sha256"
    ]
    for col in required_never_null:
        null_count = df[col].isnull().sum()
        assert null_count == 0, (
            f"\nNull values found in required column '{col}' in {filepath}: "
            f"{null_count} rows affected."
        )


# ------------------------------------------------------------------ #
#  TEST 4 — subject and sender: no nulls (empty string is OK)         #
# ------------------------------------------------------------------ #

@pytest.mark.parametrize("filepath", CANONICAL_FILES)
def test_subject_sender_no_nulls_empty_string_ok(filepath):
    """
    subject and sender must not contain null values.
    Empty string "" is acceptable when the field is absent in source email.
    """
    df = load_or_skip(filepath)
    for col in ["subject", "sender"]:
        null_count = df[col].isnull().sum()
        assert null_count == 0, (
            f"\nNull values found in '{col}' in {filepath}: {null_count} rows. "
            f"Replace nulls with empty string '' — not NaN."
        )


# ------------------------------------------------------------------ #
#  TEST 5 — No empty body strings                                     #
# ------------------------------------------------------------------ #

@pytest.mark.parametrize("filepath", CANONICAL_FILES)
def test_no_empty_body(filepath):
    """
    Body must contain at least one non-whitespace character.
    Emails with empty or whitespace-only bodies must be dropped at parse time.
    """
    df = load_or_skip(filepath)
    empty_count = (df["body"].str.strip() == "").sum()
    assert empty_count == 0, (
        f"\n{empty_count} rows have empty or whitespace-only body in {filepath}. "
        f"These must be dropped during parsing."
    )


# ------------------------------------------------------------------ #
#  TEST 6 — Label values restricted to 0 and 1                        #
# ------------------------------------------------------------------ #

@pytest.mark.parametrize("filepath", CANONICAL_FILES)
def test_label_domain(filepath):
    """label column must contain only 0 (legitimate) or 1 (phishing)."""
    df = load_or_skip(filepath)
    invalid = set(df["label"].unique()) - VALID_LABELS
    assert not invalid, (
        f"\nInvalid label values in {filepath}: {invalid}. "
        f"Only 0 and 1 are permitted."
    )


# ------------------------------------------------------------------ #
#  TEST 7 — Stratum values restricted to I, II, III                  #
# ------------------------------------------------------------------ #

@pytest.mark.parametrize("filepath", CANONICAL_FILES)
def test_stratum_domain(filepath):
    """stratum column must contain only 'I', 'II', or 'III'."""
    df = load_or_skip(filepath)
    invalid = set(df["stratum"].unique()) - VALID_STRATA
    assert not invalid, (
        f"\nInvalid stratum values in {filepath}: {invalid}. "
        f"Only 'I', 'II', 'III' are permitted."
    )


# ------------------------------------------------------------------ #
#  TEST 8 — Source values restricted to known corpus names            #
# ------------------------------------------------------------------ #

@pytest.mark.parametrize("filepath", CANONICAL_FILES)
def test_source_domain(filepath):
    """source column must contain only known corpus names."""
    df = load_or_skip(filepath)
    invalid = set(df["source"].unique()) - VALID_SOURCES
    assert not invalid, (
        f"\nInvalid source values in {filepath}: {invalid}. "
        f"Allowed values: {VALID_SOURCES}"
    )


# ------------------------------------------------------------------ #
#  TEST 9 — No duplicate message_ids                                  #
# ------------------------------------------------------------------ #

@pytest.mark.parametrize("filepath", CANONICAL_FILES)
def test_no_duplicate_message_ids(filepath):
    """Every row must have a unique message_id within the file."""
    df = load_or_skip(filepath)
    dupe_count = df["message_id"].duplicated().sum()
    assert dupe_count == 0, (
        f"\n{dupe_count} duplicate message_id values in {filepath}. "
        f"The message_id must be unique per row."
    )


# ------------------------------------------------------------------ #
#  TEST 10 — body_length matches actual body length                   #
# ------------------------------------------------------------------ #

@pytest.mark.parametrize("filepath", CANONICAL_FILES)
def test_body_length_matches_body(filepath):
    """
    body_length must equal len(body) for every row.
    Sample 200 rows for performance — if any mismatch exists, it will
    be caught here without reading the entire file.
    """
    df = load_or_skip(filepath)
    sample = df.sample(min(200, len(df)), random_state=42)
    computed = sample["body"].str.len()
    mismatches = (computed != sample["body_length"]).sum()
    assert mismatches == 0, (
        f"\n{mismatches} rows in {filepath} have body_length != len(body). "
        f"body_length must be computed as len(body) after cleaning."
    )


# ------------------------------------------------------------------ #
#  TEST 11 — body_sha256 is 64 hex characters                         #
# ------------------------------------------------------------------ #

@pytest.mark.parametrize("filepath", CANONICAL_FILES)
def test_body_sha256_format(filepath):
    """body_sha256 must be exactly 64 lowercase hexadecimal characters."""
    df = load_or_skip(filepath)
    sample = df.sample(min(200, len(df)), random_state=42)
    valid_pattern = sample["body_sha256"].str.match(r"^[a-f0-9]{64}$")
    invalid_count = (~valid_pattern).sum()
    assert invalid_count == 0, (
        f"\n{invalid_count} rows in {filepath} have malformed body_sha256 values. "
        f"Must be exactly 64 lowercase hex characters."
    )


# ------------------------------------------------------------------ #
#  TEST 12 — Both label classes present in each file                  #
# ------------------------------------------------------------------ #

@pytest.mark.parametrize("filepath", CANONICAL_FILES)
def test_both_classes_present(filepath):
    """
    Each canonical file must contain both phishing (1) and
    legitimate (0) examples. A file with only one class indicates
    a parser or combine script error.
    """
    df = load_or_skip(filepath)
    labels_present = set(df["label"].unique())
    assert 0 in labels_present, (
        f"\nNo legitimate (label=0) rows found in {filepath}."
    )
    assert 1 in labels_present, (
        f"\nNo phishing (label=1) rows found in {filepath}."
    )
