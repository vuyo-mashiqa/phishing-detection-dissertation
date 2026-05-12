"""
build_features.py
=================
Merges canonical CSVs with structural companion files, then engineers
all features defined in SCHEMA_FEATURES.md for the unified dataset.

Feature groups
--------------
  Text features   (derived from body/subject columns in canonical CSV)
  Structural      (html_char_ratio, reply_to_mismatch — from companion CSVs)
  Metadata        (stratum, source, label)

Input files
-----------
  data/processed/stratum_i/stratum_i_combined.csv
  data/processed/stratum_i/stratum_i_structural.csv
  data/processed/stratum_ii/stratum_ii_combined.csv
  data/processed/stratum_ii/stratum_ii_structural.csv
  data/processed/stratum_iii/stratum_iii_combined.csv
  data/processed/stratum_iii/stratum_iii_structural.csv

Output files
------------
  data/processed/features/features_stratum_i.csv
  data/processed/features/features_stratum_ii.csv
  data/processed/features/features_stratum_iii.csv
  data/processed/features/features_combined.csv

Run
---
  python src/data/build_features.py
"""

import csv
import re
import sys
import unicodedata
from pathlib import Path

import pandas as pd

csv.field_size_limit(10_000_000)
sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.manifest_utils import ManifestWriter


# ==============================================================================
# PATHS
# ==============================================================================
STRATUM_I_CANON      = Path("data/processed/stratum_i/stratum_i_combined.csv")
STRATUM_I_STRUCT     = Path("data/processed/stratum_i/stratum_i_structural.csv")
STRATUM_II_CANON     = Path("data/processed/stratum_ii/stratum_ii_combined.csv")
STRATUM_II_STRUCT    = Path("data/processed/stratum_ii/stratum_ii_structural.csv")
STRATUM_III_CANON    = Path("data/processed/stratum_iii/stratum_iii_combined.csv")
STRATUM_III_STRUCT   = Path("data/processed/stratum_iii/stratum_iii_structural.csv")

OUT_DIR              = Path("data/processed/features")
OUT_I                = OUT_DIR / "features_stratum_i.csv"
OUT_II               = OUT_DIR / "features_stratum_ii.csv"
OUT_III              = OUT_DIR / "features_stratum_iii.csv"
OUT_COMBINED         = OUT_DIR / "features_combined.csv"

FEATURE_COLS = [
    # Identity
    "message_id",
    # Label / metadata
    "label", "stratum", "source",
    # Structural (from companion CSVs)
    "html_char_ratio",
    "reply_to_mismatch",
    # Text length
    "body_length",
    "subject_length",
    # Lexical
    "body_unique_word_ratio",
    "body_capitalisation_ratio",
    "body_digit_ratio",
    "body_special_char_ratio",
    # URL / link signals
    "url_count",
    "unique_domain_count",
    "has_ip_url",
    "url_domain_mismatch",
    # Urgency / social engineering
    "urgency_word_count",
    # Punctuation density
    "exclamation_count",
    "question_count",
]


# ==============================================================================
# VOCABULARY LISTS
# ==============================================================================
URGENCY_WORDS = frozenset([
    "urgent", "immediately", "action required", "verify", "confirm",
    "suspend", "suspended", "expire", "expired", "limited time",
    "act now", "click here", "login", "password", "account",
    "security", "alert", "warning", "update", "validate",
    "congratulations", "winner", "prize", "claim", "free",
    "offer", "dear", "customer", "bank", "paypal", "ebay",
])

URL_RE    = re.compile(r"https?://[^\s<>\"']+", re.IGNORECASE)
DOMAIN_RE = re.compile(r"https?://(?:www\.)?([^/\s<>\"']+)", re.IGNORECASE)
IP_URL_RE = re.compile(r"https?://\d{1,3}(?:\.\d{1,3}){3}", re.IGNORECASE)


# ==============================================================================
# FEATURE EXTRACTION
# ==============================================================================
def extract_features(body: str, subject: str) -> dict:
    """
    Compute all text-derived features from a single email's body and subject.
    All features are deterministic and operate on pre-cleaned text.
    """
    body    = str(body    or "")
    subject = str(subject or "")

    # --- length ---
    body_length    = len(body)
    subject_length = len(subject)

    # --- lexical ratios ---
    words = body.split()
    n_words = len(words)
    if n_words > 0:
        unique_words            = len(set(w.lower() for w in words))
        body_unique_word_ratio  = round(unique_words / n_words, 6)
    else:
        body_unique_word_ratio  = 0.0

    n_chars = len(body)
    if n_chars > 0:
        upper_chars              = sum(1 for c in body if c.isupper())
        digit_chars              = sum(1 for c in body if c.isdigit())
        special_chars            = sum(
            1 for c in body
            if not c.isalnum() and not c.isspace()
            and unicodedata.category(c) not in ("Zs",)
        )
        body_capitalisation_ratio = round(upper_chars  / n_chars, 6)
        body_digit_ratio          = round(digit_chars  / n_chars, 6)
        body_special_char_ratio   = round(special_chars / n_chars, 6)
    else:
        body_capitalisation_ratio = 0.0
        body_digit_ratio          = 0.0
        body_special_char_ratio   = 0.0

    # --- URL features ---
    urls    = URL_RE.findall(body)
    domains = DOMAIN_RE.findall(body)

    url_count           = len(urls)
    unique_domain_count = len(set(d.lower() for d in domains))
    has_ip_url          = int(bool(IP_URL_RE.search(body)))

    # url_domain_mismatch: display text contains a domain that differs from
    # the href domain (anchor text spoofing). Detected via <a href> pattern.
    href_re  = re.compile(r'href=["\']?(https?://[^"\'>\s]+)', re.IGNORECASE)
    text_url = re.compile(r'https?://(?:www\.)?([^/\s<>\"\']+)', re.IGNORECASE)
    hrefs    = href_re.findall(body)
    mismatch = 0
    for href in hrefs:
        href_domain = DOMAIN_RE.search(href)
        text_domain = text_url.search(body[body.find(href) + len(href):body.find(href) + len(href) + 200])
        if href_domain and text_domain:
            if href_domain.group(1).lower() != text_domain.group(1).lower():
                mismatch = 1
                break
    url_domain_mismatch = mismatch

    # --- urgency words ---
    body_lower = body.lower()
    urgency_word_count = sum(
        1 for w in URGENCY_WORDS if w in body_lower
    )

    # --- punctuation density ---
    exclamation_count = body.count("!")
    question_count    = body.count("?")

    return {
        "body_length":              body_length,
        "subject_length":           subject_length,
        "body_unique_word_ratio":   body_unique_word_ratio,
        "body_capitalisation_ratio": body_capitalisation_ratio,
        "body_digit_ratio":         body_digit_ratio,
        "body_special_char_ratio":  body_special_char_ratio,
        "url_count":                url_count,
        "unique_domain_count":      unique_domain_count,
        "has_ip_url":               has_ip_url,
        "url_domain_mismatch":      url_domain_mismatch,
        "urgency_word_count":       urgency_word_count,
        "exclamation_count":        exclamation_count,
        "question_count":           question_count,
    }


# ==============================================================================
# STRATUM PROCESSOR
# ==============================================================================
def process_stratum(
    canon_path: Path,
    struct_path: Path,
    out_path: Path,
    label: str,
) -> pd.DataFrame:
    """
    Loads canonical CSV, left-joins structural companion, engineers features,
    writes per-stratum feature CSV, and returns the DataFrame.
    """
    print(f"  Loading {canon_path.name} ...")
    canon = pd.read_csv(canon_path, dtype=str, low_memory=False)
    print(f"    {len(canon):,} rows")

    print(f"  Loading {struct_path.name} ...")
    struct = pd.read_csv(struct_path, dtype=str, low_memory=False)
    print(f"    {len(struct):,} rows")

    # Left join on message_id — all canonical rows kept; structural fills in features.
    # Unmatched structural rows are impossible (companion file mirrors canonical exactly).
    merged = canon.merge(struct, on="message_id", how="left")
    assert len(merged) == len(canon), (
        f"Row count changed after merge: {len(canon)} -> {len(merged)}"
    )

    # Fill structural defaults for any missing join (should be 0 after Step 7.1)
    merged["html_char_ratio"]    = pd.to_numeric(merged["html_char_ratio"],    errors="coerce").fillna(0.0)
    merged["reply_to_mismatch"]  = pd.to_numeric(merged["reply_to_mismatch"], errors="coerce").fillna(0).astype(int)

    # --- Engineer text features row-by-row ---
    print(f"  Engineering features ...")
    feature_rows = []
    for _, row in merged.iterrows():
        feats = extract_features(row.get("body", ""), row.get("subject", ""))
        feature_rows.append(feats)

    feat_df = pd.DataFrame(feature_rows)

    # --- Assemble final DataFrame ---
    result = pd.DataFrame()
    result["message_id"]        = merged["message_id"]
    result["label"]             = pd.to_numeric(merged["label"], errors="coerce").astype(int)
    result["stratum"]           = merged["stratum"]
    result["source"]            = merged["source"]
    result["html_char_ratio"]   = merged["html_char_ratio"]
    result["reply_to_mismatch"] = merged["reply_to_mismatch"]

    for col in feat_df.columns:
        result[col] = feat_df[col]

    # Enforce column order
    result = result[[c for c in FEATURE_COLS if c in result.columns]]

    # Validate nulls
    null_counts = result.isnull().sum()
    if null_counts.any():
        print(f"  WARNING: nulls found before write:")
        print(null_counts[null_counts > 0].to_string())

    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_path, index=False, encoding="utf-8", lineterminator="\n")

    label_dist = result["label"].value_counts().to_dict()
    print(f"  {label}: {len(result):,} rows written  |  label dist: {label_dist}")
    return result


# ==============================================================================
# MAIN
# ==============================================================================
def main():
    print("=" * 65)
    print("FEATURE ENGINEERING")
    print("=" * 65)

    # Validate inputs
    inputs = [
        STRATUM_I_CANON,  STRATUM_I_STRUCT,
        STRATUM_II_CANON, STRATUM_II_STRUCT,
        STRATUM_III_CANON, STRATUM_III_STRUCT,
    ]
    missing = [str(p) for p in inputs if not p.exists()]
    if missing:
        print("ERROR — missing input files:")
        for m in missing:
            print(f"  {m}")
        sys.exit(1)
    print("All inputs validated.\n")

    mw = ManifestWriter(
        script_name="build_features",
        random_seed=None,
        parameters={
            "join_key":       "message_id",
            "join_type":      "left (canonical is authoritative)",
            "feature_groups": "text, structural, metadata",
            "url_regex":      URL_RE.pattern,
            "ip_url_regex":   IP_URL_RE.pattern,
            "urgency_vocab":  f"{len(URGENCY_WORDS)} terms",
        },
        notes=(
            "Structural features joined from companion CSVs produced by "
            "extract_structural_features.py (Step 7.1). "
            "All 176,452 records confirmed 0 unmatched before this step."
        ),
    )
    for p in inputs:
        mw.add_input(str(p))

    # --- Per-stratum processing ---
    print("--- STRATUM I ---")
    df_i   = process_stratum(STRATUM_I_CANON,   STRATUM_I_STRUCT,   OUT_I,   "Stratum I")

    print("\n--- STRATUM II ---")
    df_ii  = process_stratum(STRATUM_II_CANON,  STRATUM_II_STRUCT,  OUT_II,  "Stratum II")

    print("\n--- STRATUM III ---")
    df_iii = process_stratum(STRATUM_III_CANON, STRATUM_III_STRUCT, OUT_III, "Stratum III")

    # --- Combined ---
    print("\n--- COMBINED ---")
    combined = pd.concat([df_i, df_ii, df_iii], ignore_index=True)
    combined.to_csv(OUT_COMBINED, index=False, encoding="utf-8", lineterminator="\n")

    label_dist = combined["label"].value_counts().to_dict()
    print(f"  Combined: {len(combined):,} rows  |  label dist: {label_dist}")

    # --- Summary ---
    print("\n" + "=" * 65)
    print("FEATURE ENGINEERING COMPLETE")
    print("=" * 65)
    print(f"  Stratum I    : {len(df_i):,} rows   | {len(df_i.columns)} features")
    print(f"  Stratum II   : {len(df_ii):,} rows    | {len(df_ii.columns)} features")
    print(f"  Stratum III  : {len(df_iii):,} rows   | {len(df_iii.columns)} features")
    print(f"  Combined     : {len(combined):,} rows  | {len(combined.columns)} features")
    print(f"  Feature cols : {list(combined.columns)}")

    for out in [OUT_I, OUT_II, OUT_III, OUT_COMBINED]:
        mw.add_output(str(out))

    mw.set_counts({
        "stratum_i_rows":   len(df_i),
        "stratum_ii_rows":  len(df_ii),
        "stratum_iii_rows": len(df_iii),
        "combined_rows":    len(combined),
        "n_features":       len(FEATURE_COLS),
        "label_0_ham":      int(label_dist.get(0, 0)),
        "label_1_phishing": int(label_dist.get(1, 0)),
    })
    mw.write()

    print("\nReady for Step 7.3: run pytest tests/test_features.py -v")


if __name__ == "__main__":
    main()
    