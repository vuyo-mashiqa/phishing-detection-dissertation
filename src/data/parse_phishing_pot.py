"""
src/data/parse_phishing_pot.py

Parser for the phishing_pot corpus -- Stratum II phishing class.

Input:  data/raw/stratum_ii/phishing_pot/  (recursively finds all .eml files)
Output: data/processed/stratum_ii/phishing_pot_phishing.csv
        (canonical schema: 10 columns as per SCHEMA_CANONICAL.md)

Drop conditions:
  - Body is empty or whitespace-only after cleaning
  - Exact duplicate (same body_sha256 already seen in this corpus)

Label: 1 (phishing) | Stratum: II | Source: phishing_pot

Run:
    python src/data/parse_phishing_pot.py

Then validate:
    pytest tests/test_schema.py -v
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
INPUT_DIR   = Path("data/raw/stratum_ii/phishing_pot")
OUTPUT_PATH = Path("data/processed/stratum_ii/phishing_pot_phishing.csv")
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
    """Strip HTML tags only when markup is detected -- avoids BeautifulSoup warning."""
    if "<" in text and ">" in text:
        soup = BeautifulSoup(text, "lxml")
        return soup.get_text(separator=" ")
    return text


def extract_body(msg) -> str:
    """
    Extract and clean plain-text body.
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
    text = strip_html(raw)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def compute_body_sha256(text: str) -> str:
    """SHA-256 of lowercased body for case-insensitive deduplication."""
    return hashlib.sha256(text.lower().encode("utf-8")).hexdigest()


def make_message_id(sha256_hex: str) -> str:
    """Deterministic message_id per SCHEMA_CANONICAL.md Section 6."""
    return f"SII_pp_{sha256_hex[:16]}"


def decode_header_field(value: str) -> str:
    """Decode encoded-word MIME headers."""
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
    print("  PHISHING_POT PARSER -- Stratum II Phishing Class")
    print("=" * 65)

    mw = ManifestWriter(
        script_name="parse_phishing_pot",
        random_seed=None,
        parameters={
            "input_dir":    str(INPUT_DIR),
            "file_pattern": "**/*.eml",
            "label":        1,
            "stratum":      "II",
            "source":       "phishing_pot",
        },
        notes=(
            "phishing_pot corpus. Recursively discovers all .eml files. "
            "Drops emails with empty body after HTML stripping and cleaning. "
            "Drops exact duplicates by body_sha256."
        )
    )
    mw.add_input(str(INPUT_DIR))

    # Recursively discover all .eml files
    all_files = list(INPUT_DIR.rglob("*.eml"))
    print(f"  Files discovered (.eml, recursive): {len(all_files):,}")

    if len(all_files) == 0:
        print("  ERROR: No .eml files found. Check INPUT_DIR path.")
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
            "label":         1,
            "stratum":       "II",
            "source":        "phishing_pot",
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

    # Drop phantom rows
    df = df[df["message_id"].notna() & (df["message_id"] != "")]
    df = df[df["body"].notna() & (df["body"].str.strip() != "")]

    # subject and sender: empty string for absent fields
    df["subject"] = df["subject"].where(df["subject"].notna(), other="")
    df["sender"]  = df["sender"].where(df["sender"].notna(), other="")

    # Enforce correct dtypes
    df["label"]       = df["label"].astype(int)
    df["body_length"] = df["body_length"].astype(int)

    # Final assertion: zero nulls
    null_counts = df.isnull().sum()
    if null_counts.any():
        raise ValueError(
            f"Null values remain before write:\n{null_counts[null_counts > 0]}"
        )

    # Sanitise all string columns to clean UTF-8 before writing.
    # Some email bodies contain bytes that decoded to non-UTF-8 characters.
    # Round-tripping through UTF-8 with replace removes any such characters.
    for col in ["message_id", "subject", "body", "sender", "source",
                "original_file", "stratum", "body_sha256"]:
        df[col] = df[col].apply(
            lambda x: x.encode("utf-8", errors="replace").decode("utf-8", errors="replace")
            if isinstance(x, str) else x
        )

    # Write with errors=replace to guarantee clean UTF-8 output
    # even if any residual non-UTF-8 bytes survived sanitisation.
    with open(OUTPUT_PATH, "w", encoding="utf-8", errors="replace", newline="\n") as fh:
        df.to_csv(fh, index=False, lineterminator="\n")

    print(f"\n  Output written -> {OUTPUT_PATH}")
    print(f"  Shape: {df.shape}  |  Columns: {list(df.columns)}")

    mw.add_output(str(OUTPUT_PATH))
    mw.set_counts({
        "files_discovered":  len(all_files),
        "parse_errors":      n_error,
        "dropped_empty":     n_dropped_empty,
        "dropped_duplicate": n_dropped_dupe,
        "retained":          len(rows),
        "label_0_count":     0,
        "label_1_count":     len(rows),
    })
    mw.write()

    print("=" * 65)
    print("  DONE. Run: pytest tests/test_schema.py -v")
    print("=" * 65)


if __name__ == "__main__":
    main()
