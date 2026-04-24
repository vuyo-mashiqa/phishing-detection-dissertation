# SCHEMA_CANONICAL.md
## Canonical Processed Email Schema
**Project:** Phishing Detection Dissertation\
**Version:** 1.0\
**Date:** 2026-04-23\
**Status:** LOCKED

---

## 1. The Rule

All canonical processed CSV files across all three strata must conform
to exactly this schema. There are no exceptions and no stratum-specific
variations in the canonical files.

Stratum-specific metadata — such as PhishFuzzer fields entity_type,
length_type, motivation, url, and attachment — are stored in a SEPARATE
metadata CSV file alongside the canonical file. They are joined to the
canonical file by message_id when needed. They never appear inside the
canonical file itself.

---

## 2. Column Definitions

The canonical file has exactly 10 columns in exactly this order:

| Position | Column Name   | Type    | Null Policy              | Description |
|----------|---------------|---------|--------------------------|-------------|
| 1        | message_id    | string  | NEVER null               | Unique deterministic identifier. Format: S{stratum}_{source_abbrev}_{sha256_body_first_16_chars}. Example: SII_pp_a3f1b2c4d5e6f7a8 |
| 2        | subject       | string  | Empty string "" if absent — NEVER null | Email subject line, decoded from MIME encoding |
| 3        | body          | string  | NEVER null, NEVER empty  | Plain text body. HTML stripped with BeautifulSoup. Whitespace normalised (consecutive spaces/newlines collapsed to single space). Must contain at least one non-whitespace character. |
| 4        | sender        | string  | Empty string "" if absent — NEVER null | From header value, decoded from MIME encoding |
| 5        | label         | integer | NEVER null               | Classification target. 1 = phishing. 0 = legitimate. No other values permitted. |
| 6        | stratum       | string  | NEVER null               | Which stratum. Allowed values: "I", "II", "III" only. No other values permitted. |
| 7        | source        | string  | NEVER null               | Human-readable corpus name. Allowed values: "enron", "nazario", "phishing_pot", "csdmc2010", "phishfuzzer" only. No other values permitted. |
| 8        | original_file | string  | NEVER null               | The specific source filename this email came from. Examples: "phishing2.mbox", "00001.eml", "PhishFuzzer_emails_entity_rephrased_v1.json" |
| 9        | body_length   | integer | NEVER null, always > 0   | Character count of the cleaned body field. Computed as len(body) after cleaning. |
| 10       | body_sha256   | string  | NEVER null               | SHA-256 hash of the normalised body text (lowercased, whitespace-collapsed). Used for deduplication. Must be exactly 64 hexadecimal characters. |

---

## 3. Column Order

The columns must appear in exactly this order in every canonical file:

    message_id, subject, body, sender, label, stratum, source,
    original_file, body_length, body_sha256

A file with columns in any other order fails schema validation.

---

## 4. Null and Empty String Policy

This is a strict two-tier policy:

**Tier 1 — Columns that must NEVER be null AND must contain a value:**
message_id, body, label, stratum, source, original_file,
body_length, body_sha256

If any of these columns contain a null value in any row, the row is
a parser bug. It must be fixed before the file is committed.

**Tier 2 — Columns that must NEVER be null but MAY be empty string:**
subject, sender

When a subject or sender is absent from the original email, the parser
writes an empty string "" — not NaN, not None, not the string "null",
not a space. Exactly two double-quote characters with nothing between them.

There are NO nulls of any kind in any canonical file. The only
flexibility is that subject and sender may be empty strings.

---

## 5. Body Cleaning Rules

These rules are applied by every parser in exactly this order:

1. Extract the MIME text/plain part. If no text/plain part exists,
   fall back to text/html and strip all HTML tags using BeautifulSoup
   with the lxml parser.
2. Decode the resulting bytes to UTF-8. If the declared charset fails,
   try chardet detection. If that also fails, decode with
   errors="replace" to substitute undecodable bytes with the Unicode
   replacement character U+FFFD.
3. Strip all HTML tags from the decoded string (even if the source was
   text/plain — some emails embed HTML inside plain-text parts).
4. Collapse all consecutive whitespace characters (spaces, tabs,
   newlines, carriage returns) to a single space.
5. Strip leading and trailing whitespace from the result.
6. If the result is an empty string after these steps, discard the
   email entirely — do not write a row with an empty body.

The value stored in the body column is the result of steps 1–5.
The value stored in body_sha256 is the SHA-256 of the body text
after additionally lowercasing it (for case-insensitive deduplication).

---

## 6. message_id Construction

The message_id is constructed deterministically from the content
so that re-running a parser on the same input always produces the
same message_id. It is NOT derived from the original email's
Message-ID header (which is often absent, malformed, or duplicated).

Format: S{stratum}_{source_abbrev}_{body_sha256[:16]}

Source abbreviations:
  enron        → enron
  nazario      → naz
  phishing_pot → pp
  csdmc2010    → csdmc
  phishfuzzer phishing → pf_ph
  phishfuzzer valid    → pf_vl

Examples:
  SI_enron_a3f1b2c4d5e6f7a8   (Stratum I, Enron, legitimate)
  SI_naz_b4c5d6e7f8a9b0c1     (Stratum I, Nazario, phishing)
  SII_pp_c5d6e7f8a9b0c1d2     (Stratum II, phishing_pot, phishing)
  SII_csdmc_d6e7f8a9b0c1d2e3  (Stratum II, CSDMC2010, legitimate)
  SIII_pf_ph_e7f8a9b0c1d2e3f4 (Stratum III, PhishFuzzer, phishing)
  SIII_pf_vl_f8a9b0c1d2e3f4a5 (Stratum III, PhishFuzzer, legitimate)

---

## 7. File Naming Convention

| Stratum | Canonical file path |
|---------|---------------------|
| I       | data/processed/stratum_i/stratum_i_combined.csv |
| II      | data/processed/stratum_ii/stratum_ii_combined.csv |
| III     | data/processed/stratum_iii/stratum_iii_combined.csv |

Intermediate per-corpus files (produced before combining):
  data/processed/stratum_i/enron_ham.csv
  data/processed/stratum_i/nazario_phishing.csv
  data/processed/stratum_ii/csdmc2010_ham.csv
  data/processed/stratum_ii/phishing_pot_phishing.csv
  data/processed/stratum_iii/stratum_iii_combined.csv  (produced directly — no split)

Metadata file (Stratum III only):
  data/processed/stratum_iii/stratum_iii_metadata.csv

---

## 8. Mandatory CSV Read Convention

All canonical CSV files in this project MUST be read with pandas using:

    pd.read_csv(path, keep_default_na=False, na_values=[])

Rationale: pandas converts empty strings to NaN by default. Subject and
sender fields are legitimately empty for many emails. Without this
parameter, a schema-compliant file with empty strings will appear to
contain nulls when read, causing false test failures and incorrect null
counts. Every script, test, and notebook in this project that reads a
canonical CSV must use these two parameters.

## 9. Encoding and Format

- File encoding: UTF-8 (no BOM)
- Line endings: LF (Unix-style). Windows CRLF line endings are not used.
- CSV quoting: standard CSV quoting as produced by pandas to_csv with
  default parameters (quotes fields containing commas, quotes, or newlines)
- Header row: always present, exactly matching the column names in
  Section 2 with no leading or trailing spaces
- Index column: pandas index is NOT written (index=False in to_csv)

---

## 10. Validation

The file tests/test_schema.py enforces this schema automatically.
Run it after every parser execution:

    pytest tests/test_schema.py -v

A test failure means the canonical output is not valid and must not
be committed.

---

## 11. What Happens When Tests Fail

If pytest reports a failure:

FAIL test_no_nulls_in_required_columns[stratum_ii_combined.csv]
  → Find the parser that produced this file.
  → Add a fillna("") call for subject and sender.
  → Add an assertion before writing that checks null counts.
  → Re-run the parser and re-run pytest.
  → Commit only when the test passes.

FAIL test_columns[stratum_iii_combined.csv]
  → The column names or their order do not match Section 2.
  → Find the line in the parser that writes the CSV.
  → Enforce column order: df = df[CANONICAL_COLS] before to_csv.
  → Re-run and re-test.

FAIL test_label_domain[stratum_i_combined.csv]
  → A label value other than 0 or 1 exists in the file.
  → Check the parser's label assignment logic.
  → Fix and re-test.