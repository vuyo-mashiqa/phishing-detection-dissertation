"""
pii_scrub_enron.py
==================
Produces a PII-scrubbed version of stratum_i_combined.csv for use in
published artefacts (dissertation examples, supplementary material).

The UNSCRUBBED file remains on local disk for model training.
Scrubbing is applied to the Enron ham rows ONLY (source == "enron").
Nazario phishing rows (source == "nazario") are passed through unchanged.

Scrubbing pipeline (regex-only, no spaCy):
    1. Email addresses       -> [EMAIL]
    2. Phone numbers         -> [PHONE]
    3. US SSN patterns       -> [SSN]
    4. Credit card patterns  -> [CARD]
    5. Simple PERSON-like patterns (capitalised word sequences) -> [PERSON]

Design decisions:
    - Only Enron rows are scrubbed (Nazario is phishing bait, low PII risk).
    - Regex-only for performance and determinism.
    - The scrub map (original -> placeholder) is NOT saved to the repo.
    - body_sha256 is recomputed on the scrubbed body.
    - body_length is recomputed on the scrubbed body.
    - message_id is preserved (links scrubbed to unscrubbed row).

Output:
    data/processed/stratum_i/stratum_i_scrubbed.csv
"""

import csv
import hashlib
import re
from pathlib import Path

csv.field_size_limit(10_000_000)

INPUT_PATH  = Path("data/processed/stratum_i/stratum_i_combined.csv")
OUTPUT_PATH = Path("data/processed/stratum_i/stratum_i_scrubbed.csv")
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

CANONICAL_COLS = [
    "message_id", "subject", "body", "sender", "label",
    "stratum", "source", "original_file", "body_length", "body_sha256",
]

# Regex patterns
PATTERNS = [
    # Email addresses
    (re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}"), "[EMAIL]"),
    # Phone numbers
    (re.compile(r"\b(\+?1[\s\-.]?)?\(?\d{3}\)?[\s\-.]?\d{3}[\s\-.]?\d{4}\b"), "[PHONE]"),
    # US SSN
    (re.compile(r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b"), "[SSN]"),
    # Credit cards
    (re.compile(r"\b(?:\d{4}[\s\-]?){3}\d{4}\b"), "[CARD]"),
]

# Simple PERSON-like pattern: 2–4 capitalised words in a row (e.g., "John A Smith")
PERSON_PATTERN = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\b")


def regex_scrub_body(text: str) -> str:
    for pattern, placeholder in PATTERNS:
        text = pattern.sub(placeholder, text)
    # Replace name-like sequences
    text = PERSON_PATTERN.sub("[PERSON]", text)
    return text


def regex_scrub_subject(text: str) -> str:
    # Subject: email + simple name patterns
    text = PATTERNS[0][0].sub("[EMAIL]", text)  # email
    text = PERSON_PATTERN.sub("[PERSON]", text)
    return text


def sha256(text: str) -> str:
    return hashlib.sha256(text.lower().encode("utf-8")).hexdigest()


def main():
    print("=" * 65)
    print("PII SCRUBBER (regex-only) — Enron rows only")
    print("=" * 65)

    with open(INPUT_PATH, newline="", encoding="utf-8", errors="replace") as f:
        rows = list(csv.DictReader(f))

    total     = len(rows)
    n_enron   = sum(1 for r in rows if r["source"] == "enron")
    n_nazario = sum(1 for r in rows if r["source"] == "nazario")

    print(f"Total rows:   {total:,}")
    print(f"Enron rows:   {n_enron:,}  (will be scrubbed)")
    print(f"Nazario rows: {n_nazario:,}  (pass-through)")
    print("\nScrubbing Enron rows...")

    scrubbed_rows = []
    for i, row in enumerate(rows):
        if i > 0 and i % 5000 == 0:
            print(f"  Processed {i:,}/{total:,} ({i/total*100:.1f}%)")

        if row["source"] == "enron":
            body    = row["body"]
            subject = row.get("subject", "")
            sender  = row.get("sender", "")

            scrubbed_body    = regex_scrub_body(body)
            scrubbed_subject = regex_scrub_subject(subject)
            scrubbed_sender  = "[EMAIL]" if sender else ""

            new_row = dict(row)
            new_row["body"]        = scrubbed_body
            new_row["subject"]     = scrubbed_subject
            new_row["sender"]      = scrubbed_sender
            new_row["body_length"] = len(scrubbed_body)
            new_row["body_sha256"] = sha256(scrubbed_body)
        else:
            new_row = dict(row)

        scrubbed_rows.append({col: new_row[col] for col in CANONICAL_COLS})

    print(f"  Processed {total:,}/{total:,} (100.0%)")

    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CANONICAL_COLS)
        w.writeheader()
        w.writerows(scrubbed_rows)

    print(f"\nOutput written -> {OUTPUT_PATH}")
    print(f"  Rows: {len(scrubbed_rows):,}")
    print("\nScrubbed file is for publication artefacts only.")
    print("Unscrubbed stratum_i_combined.csv is retained for training.")

if __name__ == "__main__":
    main()
    