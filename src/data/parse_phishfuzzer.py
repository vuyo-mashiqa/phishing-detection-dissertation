"""
src/data/parse_phishfuzzer.py

Parser for the PhishFuzzer LLM-generated corpus -- Stratum III.

Input:  data/raw/stratum_iii/PhishFuzzer/PhishFuzzer_emails_entity_rephrased_v1.json
        (19,800 records total)

Inclusion rules:
  - Type == 'Phishing'  -> label = 1 (expected ~6,756 pre-dedup)
  - Type == 'Valid'     -> label = 0 (expected ~6,600 pre-dedup)
  - Type == 'Spam'      -> EXCLUDED (not phishing, not legitimate enterprise mail)
  - Created by == 'LLM' -> all records pass (verified at acquisition)

Outputs:
  data/processed/stratum_iii/stratum_iii_combined.csv
      Canonical schema: 10 columns (message_id, subject, body, sender,
      label, stratum, source, original_file, body_length, body_sha256)

  data/processed/stratum_iii/stratum_iii_metadata.csv
      PhishFuzzer-specific fields joined by message_id:
      message_id, entity_type, length_type, motivation, urls, attachment

Design decision:
  PhishFuzzer metadata fields (entity_type, length_type, motivation,
  urls, attachment) are stored in a SEPARATE metadata CSV, not in the
  canonical combined CSV. This preserves the unified three-stratum schema
  while retaining all research-relevant metadata for analysis.
  Join key: message_id (one-to-one relationship).
"""

import hashlib
import json
import re
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.manifest_utils import ManifestWriter

# ------------------------------------------------------------------ #
#  PATHS                                                              #
# ------------------------------------------------------------------ #
INPUT_FILE      = Path("data/raw/stratum_iii/PhishFuzzer/PhishFuzzer_emails_entity_rephrased_v1.json")
OUTPUT_CANON    = Path("data/processed/stratum_iii/stratum_iii_combined.csv")
OUTPUT_METADATA = Path("data/processed/stratum_iii/stratum_iii_metadata.csv")
OUTPUT_CANON.parent.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------ #
#  SCHEMA DEFINITIONS                                                 #
# ------------------------------------------------------------------ #
CANONICAL_COLS = [
    "message_id", "subject", "body", "sender",
    "label", "stratum", "source", "original_file",
    "body_length", "body_sha256",
]

METADATA_COLS = [
    "message_id", "entity_type", "length_type",
    "motivation", "urls", "attachment",
]

# Label mapping — Spam is excluded
TYPE_TO_LABEL = {
    "Phishing": 1,
    "Valid":    0,
}

# ------------------------------------------------------------------ #
#  HELPERS                                                            #
# ------------------------------------------------------------------ #

def clean_text(text: str) -> str:
    """Collapse whitespace. PhishFuzzer bodies are pre-generated plain text."""
    if not text or not isinstance(text, str):
        return ""
    return re.sub(r"\s+", " ", text).strip()


def compute_body_sha256(text: str) -> str:
    """SHA-256 of lowercased body for case-insensitive deduplication."""
    return hashlib.sha256(text.lower().encode("utf-8", errors="replace")).hexdigest()


def make_message_id(sha256_hex: str) -> str:
    """Deterministic message_id per SCHEMA_CANONICAL.md Section 6."""
    return f"SIII_pf_{sha256_hex[:16]}"


def safe_str(value) -> str:
    """Convert any value to string, treating None and 'None' as empty string."""
    if value is None:
        return ""
    s = str(value).strip()
    return "" if s.lower() == "none" else s


# ------------------------------------------------------------------ #
#  MAIN                                                               #
# ------------------------------------------------------------------ #

def main():
    print("=" * 65)
    print("  PHISHFUZZER PARSER -- Stratum III LLM-Generated Corpus")
    print("=" * 65)

    mw = ManifestWriter(
        script_name="parse_phishfuzzer",
        random_seed=None,
        parameters={
            "input_file":        str(INPUT_FILE),
            "included_types":    ["Phishing", "Valid"],
            "excluded_types":    ["Spam"],
            "created_by_filter": "LLM",
            "seed_file":         "PhishFuzzer_emails_original_seed_v1.json -- excluded",
        },
        notes=(
            "PhishFuzzer LLM-generated corpus. Includes Type=Phishing and "
            "Type=Valid only. Excludes Type=Spam entirely. "
            "PhishFuzzer metadata fields stored in separate stratum_iii_metadata.csv "
            "joined by message_id. Canonical combined CSV contains 10-column schema only."
        )
    )
    mw.add_input(str(INPUT_FILE))

    # ---- Load JSON ---------------------------------------------------
    print(f"  Loading {INPUT_FILE.name} ({INPUT_FILE.stat().st_size:,} bytes)...")
    with open(INPUT_FILE, encoding="utf-8", errors="replace") as f:
        records = json.load(f)
    print(f"  Total records in file: {len(records):,}")

    # ---- Count by type before filtering ------------------------------
    type_counts = {}
    for r in records:
        t = r.get("Type", "UNKNOWN")
        type_counts[t] = type_counts.get(t, 0) + 1
    print(f"  Type distribution: {type_counts}")

    # ---- Parse -------------------------------------------------------
    canon_rows    = []
    metadata_rows = []
    seen_hashes   = set()

    n_excluded_spam  = 0
    n_dropped_empty  = 0
    n_dropped_dupe   = 0

    for record in tqdm(records, desc="  Parsing", unit="record"):
        email_type = record.get("Type", "")

        # Exclude Spam
        if email_type == "Spam":
            n_excluded_spam += 1
            continue

        # Skip unknown types
        if email_type not in TYPE_TO_LABEL:
            continue

        label = TYPE_TO_LABEL[email_type]

        body = clean_text(record.get("Body", ""))
        if not body:
            n_dropped_empty += 1
            continue

        sha = compute_body_sha256(body)
        if sha in seen_hashes:
            n_dropped_dupe += 1
            continue
        seen_hashes.add(sha)

        message_id = make_message_id(sha)
        subject    = clean_text(record.get("Subject", ""))
        sender     = clean_text(record.get("Sender",  ""))

        canon_rows.append({
            "message_id":    message_id,
            "subject":       subject,
            "body":          body,
            "sender":        sender,
            "label":         label,
            "stratum":       "III",
            "source":        "phishfuzzer",
            "original_file": INPUT_FILE.name,
            "body_length":   len(body),
            "body_sha256":   sha,
        })

        metadata_rows.append({
            "message_id":  message_id,
            "entity_type": safe_str(record.get("Entity_Type")),
            "length_type": safe_str(record.get("Length_Type")),
            "motivation":  safe_str(record.get("Motivation")),
            "urls":        safe_str(record.get("URL")),
            "attachment":  safe_str(record.get("File")),
        })

    n_retained = len(canon_rows)
    n_phishing = sum(1 for r in canon_rows if r["label"] == 1)
    n_valid    = sum(1 for r in canon_rows if r["label"] == 0)

    print(f"\n  Total records:         {len(records):,}")
    print(f"  Excluded (Spam):       {n_excluded_spam:,}")
    print(f"  Dropped (empty body):  {n_dropped_empty:,}")
    print(f"  Dropped (duplicate):   {n_dropped_dupe:,}")
    print(f"  Retained:              {n_retained:,}")
    print(f"    Phishing (label=1):  {n_phishing:,}")
    print(f"    Valid    (label=0):  {n_valid:,}")

    if not canon_rows:
        print("  ERROR: No records retained. Aborting.")
        sys.exit(1)

    # ---- Build DataFrames --------------------------------------------
    df_canon = pd.DataFrame(canon_rows)[CANONICAL_COLS]
    df_meta  = pd.DataFrame(metadata_rows)[METADATA_COLS]

    # ---- Null policy -------------------------------------------------
    df_canon["subject"] = df_canon["subject"].where(df_canon["subject"].notna(), other="")
    df_canon["sender"]  = df_canon["sender"].where(df_canon["sender"].notna(), other="")
    df_canon["label"]       = df_canon["label"].astype(int)
    df_canon["body_length"] = df_canon["body_length"].astype(int)

    for col in METADATA_COLS[1:]:
        df_meta[col] = df_meta[col].where(df_meta[col].notna(), other="")

    # ---- Null assertions ---------------------------------------------
    canon_nulls = df_canon.isnull().sum()
    if canon_nulls.any():
        raise ValueError(f"Nulls in canonical CSV:\n{canon_nulls[canon_nulls > 0]}")

    meta_nulls = df_meta.isnull().sum()
    if meta_nulls.any():
        raise ValueError(f"Nulls in metadata CSV:\n{meta_nulls[meta_nulls > 0]}")

    # ---- Write -------------------------------------------------------
    with open(OUTPUT_CANON, "w", encoding="utf-8", errors="replace", newline="\n") as fh:
        df_canon.to_csv(fh, index=False, lineterminator="\n")

    with open(OUTPUT_METADATA, "w", encoding="utf-8", errors="replace", newline="\n") as fh:
        df_meta.to_csv(fh, index=False, lineterminator="\n")

    print(f"\n  Canonical CSV  -> {OUTPUT_CANON}")
    print(f"  Shape: {df_canon.shape} | Columns: {list(df_canon.columns)}")
    print(f"\n  Metadata CSV   -> {OUTPUT_METADATA}")
    print(f"  Shape: {df_meta.shape} | Columns: {list(df_meta.columns)}")

    # ---- Verify metadata join integrity ------------------------------
    assert set(df_canon["message_id"]) == set(df_meta["message_id"]), \
        "message_id mismatch between canonical and metadata CSVs"
    print(f"\n  Join integrity: PASSED (message_id sets are identical)")

    mw.add_output(str(OUTPUT_CANON))
    mw.add_output(str(OUTPUT_METADATA))
    mw.set_counts({
        "total_records":      len(records),
        "excluded_spam":      n_excluded_spam,
        "dropped_empty":      n_dropped_empty,
        "dropped_duplicate":  n_dropped_dupe,
        "retained":           n_retained,
        "label_0_valid":      n_valid,
        "label_1_phishing":   n_phishing,
    })
    mw.write()

    print("=" * 65)
    print("  DONE. Run: pytest tests/test_schema.py -v")
    print("=" * 65)


if __name__ == "__main__":
    main()
