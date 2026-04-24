"""
src/data/parse_csdmc2010.py

Parser for the CSDMC2010 ham corpus — Stratum II legitimate class.

Input:  data/raw/stratum_ii/csdmc2010/ham/ham/
        Files are named TRAIN_XXXXX.eml.txt (RFC 2822 format)
Output: data/processed/stratum_ii/csdmc2010_ham.csv
        (canonical schema: 10 columns as per SCHEMA_CANONICAL.md)

Drop conditions:
  - Body is empty or whitespace-only after cleaning
  - Exact duplicate (same body_sha256 already seen in this corpus)

Label: 0 (legitimate) | Stratum: II | Source: csdmc2010
"""

import email as email_lib
import hashlib
import re
import sys
from email.header import decode_header, make_header
from pathlib import Path

import chardet
import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.manifest_utils import ManifestWriter

# ------------------------------------------------------------------ #
#  PATHS                                                              #
# ------------------------------------------------------------------ #
INPUT_DIR   = Path("data/raw/stratum_ii/csdmc2010")
OUTPUT_PATH = Path("data/processed/stratum_ii/csdmc2010_ham.csv")
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------ #
#  CANONICAL SCHEMA COLUMNS                                           #
# ------------------------------------------------------------------ #
CANONICAL_COLS = [
    "message_id", "subject", "body", "sender",
    "label", "stratum", "source", "original_file",
    "body_length", "body_sha256",
]

# ------------------------------------------------------------------ #
#  HELPERS                                                            #
# ------------------------------------------------------------------ #

def decode_payload(part) -> str:
    """Decode a MIME part to string, trying declared charset then chardet."""
    payload = part.get_payload(decode=True)
    if payload is None:
        return ""
    declared = part.get_content_charset()
    if declared:
        try:
            return payload.decode(declared, errors="replace")
        except (LookupError, UnicodeDecodeError):
            pass
    detected = chardet.detect(payload)
    enc = detected.get("encoding") or "utf-8"
    try:
        return payload.decode(enc, errors="replace")
    except (LookupError, UnicodeDecodeError):
        return payload.decode("utf-8", errors="replace")


def strip_html(text: str) -> str:
    """
    Strip HTML tags only if the text actually contains HTML markup.
    Avoids the BeautifulSoup MarkupResemblesLocatorWarning on plain text.
    """
    if "<" in text and ">" in text:
        soup = BeautifulSoup(text, "lxml")
        return soup.get_text(separator=" ")
    return text


def extract_body(msg) -> str:
    """
    Extract and clean plain-text body from a parsed email message.
    Preference: text/plain -> text/html (stripped) -> empty string.
    Applies all cleaning rules from SCHEMA_CANONICAL.md Section 5.
    """
    plain_parts = []
    html_parts  = []

    if msg.is_multipart():
        for part in msg.walk():
            ct = part.get_content_type()
            if ct == "text/plain":
                plain_parts.append(decode_payload(part))
            elif ct == "text/html":
                html_parts.append(decode_payload(part))
    else:
        ct = msg.get_content_type()
        if ct == "text/plain":
            plain_parts.append(decode_payload(msg))
        elif ct == "text/html":
            html_parts.append(decode_payload(msg))

    raw = " ".join(plain_parts) if plain_parts else " ".join(html_parts)

    # Strip HTML only when markup is detected
    text = strip_html(raw)

    # Collapse all whitespace to single spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text


def compute_body_sha256(text: str) -> str:
    """SHA-256 of lowercased body for case-insensitive deduplication."""
    return hashlib.sha256(text.lower().encode("utf-8")).hexdigest()


def make_message_id(sha256_hex: str) -> str:
    """Deterministic message_id per SCHEMA_CANONICAL.md Section 6."""
    return f"SII_csdmc_{sha256_hex[:16]}"


def decode_header_field(value: str) -> str:
    """Decode encoded-word MIME headers (e.g. =?utf-8?b?...?=)."""
    if not value:
        return ""
    try:
        return str(make_header(decode_header(value)))
    except Exception:
        return value


# ------------------------------------------------------------------ #
#  MAIN                                                               #
# ------------------------------------------------------------------ #

def main():
    print("=" * 65)
    print("  CSDMC2010 HAM PARSER -- Stratum II Legitimate Class")
    print("=" * 65)

    mw = ManifestWriter(
        script_name="parse_csdmc2010",
        random_seed=None,
        parameters={
            "input_dir":      str(INPUT_DIR),
            "file_pattern":   "*.eml.txt",
            "label":          0,
            "stratum":        "II",
            "source":         "csdmc2010",
            "exclude_macosx": True,
        },
        notes=(
            "CSDMC2010 ham corpus. Files are TRAIN_XXXXX.eml.txt format. "
            "Excludes __MACOSX artefact directory. "
            "Drops emails with empty body after cleaning. "
            "Drops exact duplicates by body_sha256."
        )
    )
    mw.add_input(str(INPUT_DIR))

    # Discover all .eml.txt files, excluding __MACOSX artefacts
    all_files = [
        f for f in INPUT_DIR.rglob("*.eml.txt")
        if "__MACOSX" not in str(f)
        and not f.name.startswith(".")
    ]
    print(f"  Files discovered (.eml.txt, excl __MACOSX): {len(all_files):,}")

    if len(all_files) == 0:
        print("  ERROR: No .eml.txt files found. Check INPUT_DIR path.")
        sys.exit(1)

    rows            = []
    seen_hashes     = set()
    n_dropped_empty = 0
    n_dropped_dupe  = 0
    n_error         = 0

    for filepath in tqdm(all_files, desc="  Parsing", unit="email"):
        try:
            raw_bytes = filepath.read_bytes()
            msg = email_lib.message_from_bytes(raw_bytes)
        except Exception:
            n_error += 1
            continue

        body = extract_body(msg)

        if not body:
            n_dropped_empty += 1
            continue

        sha = compute_body_sha256(body)

        if sha in seen_hashes:
            n_dropped_dupe += 1
            continue
        seen_hashes.add(sha)

        subject = decode_header_field(msg.get("Subject", "") or "")
        sender  = decode_header_field(msg.get("From",    "") or "")

        rows.append({
            "message_id":    make_message_id(sha),
            "subject":       subject,
            "body":          body,
            "sender":        sender,
            "label":         0,
            "stratum":       "II",
            "source":        "csdmc2010",
            "original_file": filepath.name,
            "body_length":   len(body),
            "body_sha256":   sha,
        })

    print(f"\n  Files discovered:     {len(all_files):,}")
    print(f"  Parse errors:         {n_error:,}")
    print(f"  Dropped (empty body): {n_dropped_empty:,}")
    print(f"  Dropped (duplicate):  {n_dropped_dupe:,}")
    print(f"  Retained:             {len(rows):,}")

    if not rows:
        print("  ERROR: No emails retained. Aborting.")
        sys.exit(1)

    df = pd.DataFrame(rows)
    df = df[CANONICAL_COLS]

    # Drop phantom rows: any row missing message_id or with empty/whitespace body
    df = df[df["message_id"].notna() & (df["message_id"] != "")]
    df = df[df["body"].notna() & (df["body"].str.strip() != "")]

    # subject and sender: write empty string for absent fields
    df["subject"] = df["subject"].where(df["subject"].notna(), other="")
    df["sender"]  = df["sender"].where(df["sender"].notna(), other="")

    # Enforce correct dtypes before writing
    df["label"]       = df["label"].astype(int)
    df["body_length"] = df["body_length"].astype(int)

    # Final assertion: zero nulls before writing to disk
    null_counts = df.isnull().sum()
    if null_counts.any():
        raise ValueError(f"Null values remain before write:\n{null_counts[null_counts > 0]}")

    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8", lineterminator="\n")

    print(f"\n  Output written -> {OUTPUT_PATH}")
    print(f"  Shape: {df.shape}  |  Columns: {list(df.columns)}")

    mw.add_output(str(OUTPUT_PATH))
    mw.set_counts({
        "files_discovered":  len(all_files),
        "parse_errors":      n_error,
        "dropped_empty":     n_dropped_empty,
        "dropped_duplicate": n_dropped_dupe,
        "retained":          len(rows),
        "label_0_count":     len(rows),
        "label_1_count":     0,
    })
    mw.write()

    print("=" * 65)
    print("  DONE. Run: pytest tests/test_schema.py -v")
    print("=" * 65)


if __name__ == "__main__":
    main()
