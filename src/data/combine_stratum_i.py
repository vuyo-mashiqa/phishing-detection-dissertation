"""
combine_stratum_i.py
====================
Combines Enron ham (label=0) and Nazario phishing (label=1) into the single
canonical Stratum I combined file.

Inputs:
    data/processed/stratum_i/enron_ham.csv
    data/processed/stratum_i/nazario_phishing.csv

Output:
    data/processed/stratum_i/stratum_i_combined.csv  (176K rows, 10 cols)
"""

import csv
import sys
import uuid
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.manifest_utils import ManifestWriter

csv.field_size_limit(10_000_000)

# ------------------------------------------------------------------
# PATHS
# ------------------------------------------------------------------
HAM_PATH   = Path("data/processed/stratum_i/enron_ham.csv")
PHISH_PATH = Path("data/processed/stratum_i/nazario_phishing.csv")
OUTPUT_PATH = Path("data/processed/stratum_i/stratum_i_combined.csv")

# ------------------------------------------------------------------
# CANONICAL SCHEMA — must match SCHEMA_CANONICAL.md exactly
# ------------------------------------------------------------------
CANONICAL_COLS = [
    "message_id", "subject", "body", "sender", "label",
    "stratum", "source", "original_file", "body_length", "body_sha256",
]

VALID_LABELS  = {0, 1}
VALID_SOURCES = {"enron", "nazario", "phishing_pot", "csdmc2010", "phishfuzzer"}

LABEL_MAP = {"ham": 0, "phishing": 1}
SOURCE_MAP = {
    "Enron Email Corpus (CMU 2015)": "enron",
    "Nazario Phishing Corpus":       "nazario",
}


def load_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def normalise_row(row: dict) -> dict:
    """Apply all four normalisations to a single row dict in place."""
    # 1. Label encoding
    row["label"] = LABEL_MAP.get(str(row.get("label", "")), row.get("label", ""))
    try:
        row["label"] = int(row["label"])
    except (ValueError, TypeError):
        pass  # will be caught by validation

    # 2. Source normalisation
    row["source"] = SOURCE_MAP.get(row.get("source", ""), row.get("source", ""))

    # 3. body_length — compute from body regardless of whether field exists
    row["body_length"] = len(row.get("body", ""))

    # 4. Null -> empty string for nullable fields
    for field in ("subject", "sender"):
        if row.get(field) is None or row.get(field) == "None":
            row[field] = ""

    return row


def fix_duplicate_message_ids(rows: list[dict]) -> tuple[list[dict], int]:
    """Reassign UUID to any message_id that appears more than once."""
    id_counter = Counter(r["message_id"] for r in rows)
    seen = set()
    reassigned = 0
    for row in rows:
        mid = row["message_id"]
        if id_counter[mid] > 1:
            if mid in seen:
                row["message_id"] = str(uuid.uuid4())
                reassigned += 1
            else:
                seen.add(mid)
        else:
            seen.add(mid)
    return rows, reassigned


def validate(rows: list[dict]) -> list[str]:
    """Return a list of violation strings. Empty list = clean."""
    errors = []
    actual_cols = list(rows[0].keys()) if rows else []
    if actual_cols != CANONICAL_COLS:
        errors.append(f"Column order mismatch.\n  Got:      {actual_cols}\n  Expected: {CANONICAL_COLS}")
    for i, row in enumerate(rows):
        if row.get("label") not in VALID_LABELS:
            errors.append(f"Row {i}: invalid label '{row.get('label')}'")
        if row.get("source") not in VALID_SOURCES:
            errors.append(f"Row {i}: invalid source '{row.get('source')}'")
        for col in ("message_id", "body", "stratum", "source", "original_file", "body_sha256"):
            if not row.get(col):
                errors.append(f"Row {i}: null/empty in required field '{col}'")
        if int(row.get("body_length", -1)) != len(row.get("body", "")):
            errors.append(f"Row {i}: body_length mismatch")
        if len(errors) > 20:
            errors.append("... (truncated after 20 errors)")
            break
    return errors


def main():
    print("=" * 65)
    print("STRATUM I COMBINE SCRIPT")
    print("=" * 65)

    mw = ManifestWriter(
        script_name="combine_stratum_i",
        random_seed=None,
        parameters={
            "ham_source": str(HAM_PATH),
            "phish_source": str(PHISH_PATH),
            "imbalance_policy": "natural -- handled at modelling time",
            "label_map": str(LABEL_MAP),
            "source_map": str(SOURCE_MAP),
        },
        notes=(
            "Canonical Stratum I combined file. Combines Enron ham (label=0) and "
            "Nazario phishing (label=1). Applies label encoding, source normalisation, "
            "body_length computation, column ordering, and duplicate message_id resolution "
            "in a single reproducible pass. Natural class imbalance (~67:1) preserved."
        ),
    )
    mw.add_input(str(HAM_PATH))
    mw.add_input(str(PHISH_PATH))

    # ---- Load ---------------------------------------------------
    print(f"Loading {HAM_PATH.name}...")
    ham_rows = load_csv(HAM_PATH)
    print(f"  Rows loaded: {len(ham_rows):,}")

    print(f"Loading {PHISH_PATH.name}...")
    phish_rows = load_csv(PHISH_PATH)
    print(f"  Rows loaded: {len(phish_rows):,}")

    # ---- Cross-corpus deduplication -----------------------------
    ham_hashes = {r["body_sha256"] for r in ham_rows}
    cross_dupes = [r for r in phish_rows if r["body_sha256"] in ham_hashes]
    phish_clean = [r for r in phish_rows if r["body_sha256"] not in ham_hashes]
    if cross_dupes:
        print(f"WARNING: {len(cross_dupes)} cross-corpus duplicates removed from phishing side")
    else:
        print("Cross-corpus deduplication: 0 duplicates — clean")

    # ---- Normalise ----------------------------------------------
    all_rows = [normalise_row(r) for r in ham_rows] + \
               [normalise_row(r) for r in phish_clean]

    # ---- Fix duplicate message_ids ------------------------------
    all_rows, reassigned = fix_duplicate_message_ids(all_rows)
    print(f"Duplicate message_ids fixed: {reassigned} reassigned to UUID")

    # ---- Enforce column order -----------------------------------
    all_rows = [{col: row[col] for col in CANONICAL_COLS} for row in all_rows]

    # ---- Validate -----------------------------------------------
    errors = validate(all_rows)
    if errors:
        print("\nVALIDATION FAILURES — fix before writing:")
        for e in errors:
            print(f"  {e}")
        sys.exit(1)

    # ---- Write --------------------------------------------------
    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CANONICAL_COLS)
        w.writeheader()
        w.writerows(all_rows)

    label_counts = Counter(r["label"] for r in all_rows)
    source_counts = Counter(r["source"] for r in all_rows)
    n_ham   = label_counts[0]
    n_phish = label_counts[1]
    ratio   = n_phish / n_ham if n_ham else float("inf")

    print(f"\nOutput written -> {OUTPUT_PATH}")
    print(f"  Total rows:          {len(all_rows):,}")
    print(f"  Ham   (label=0):     {n_ham:,}  ({source_counts['enron']:,} enron)")
    print(f"  Phish (label=1):     {n_phish:,}  ({source_counts['nazario']:,} nazario)")
    print(f"  Phish/Ham ratio:     {ratio:.4f}  (severe imbalance — handled at modelling time)")
    print(f"  Column order:        {CANONICAL_COLS}")
    print(f"  Columns match:       {list(all_rows[0].keys()) == CANONICAL_COLS}")

    mw.add_output(str(OUTPUT_PATH))
    mw.set_counts({
        "ham_rows_loaded":            len(ham_rows),
        "phishing_rows_loaded":       len(phish_rows),
        "cross_corpus_dupes_removed": len(cross_dupes),
        "duplicate_message_ids_fixed": reassigned,
        "ham_retained":               n_ham,
        "phishing_retained":          n_phish,
        "total_retained":             len(all_rows),
        "phish_ham_ratio":            round(ratio, 4),
    })
    mw.write()
    print("=" * 65)
    print("DONE. Run: pytest tests/test_schema.py -v")
    print("=" * 65)


if __name__ == "__main__":
    main()
