# SAMPLE_SIZE_RATIONALE.md
## Confirmed Sample Sizes — Derived from Parser Outputs
**Version:** 1.0\
**Date:** 2026-04-26\
**Status:** LOCKED for Strata II\
**Authority:** All figures in this document are derived from actual CSV
               row counts produced by the parser scripts. No figure is
               an estimate or a pre-parser assumption.

---

## Guiding Principle

Sample size figures are recorded AFTER parsers run, not before.
The parser output is the ground truth. This document describes those
outputs and explains the decisions that produced them.

---

## Stratum II — CONFIRMED AND LOCKED

Stratum II processing completed: 2026-04-25\
Source CSVs:
  - data/processed/stratum_ii/csdmc2010_ham.csv
  - data/processed/stratum_ii/phishing_pot_phishing.csv
  - data/processed/stratum_ii/stratum_ii_combined.csv

### Legitimate Class — CSDMC2010

| Stage | Count | Notes |
|-------|-------|-------|
| Raw files discovered | 2,949 | .eml.txt files, __MACOSX excluded |
| Parse errors | 0 | All files successfully parsed |
| Dropped — empty body | 0 | All files had extractable content |
| Dropped — exact duplicate (body_sha256) | 186 | Same body text, different files |
| **Retained** | **2,763** | **Authoritative figure** |

### Phishing Class — phishing_pot

| Stage | Count | Notes |
|-------|-------|-------|
| Raw files discovered | 7,911 | .eml files, recursive discovery |
| Parse errors | 0 | All files successfully parsed |
| Dropped — empty body | 225 | Image-only or empty HTML emails |
| Dropped — exact duplicate (body_sha256) | 2,601 | Template phishing emails sent to multiple targets |
| **Retained** | **5,085** | **Authoritative figure** |

### Stratum II Combined

| Metric | Value |
|--------|-------|
| Ham (label=0) | 2,763 |
| Phishing (label=1) | 5,085 |
| **Total** | **7,848** |
| Phishing:Ham ratio | 1.84:1 |
| Imbalance handling | Natural imbalance preserved in combined file. Handled at modelling time via stratified sampling and class-weight adjustment. |

---

## Stratum III — CONFIRMED AND LOCKED

Stratum III processing completed: 2026-04-26
Source files:
  data/raw/stratum_iii/PhishFuzzer/PhishFuzzer_emails_entity_rephrased_v1.json
  data/processed/stratum_iii/stratum_iii_combined.csv
  data/processed/stratum_iii/stratum_iii_metadata.csv

### Inclusion / Exclusion

| Type | Count | Decision | Reason |
|------|-------|----------|--------|
| Phishing | 6,756 | INCLUDED | label=1 |
| Valid | 6,600 | INCLUDED | label=0 |
| Spam | 6,444 | EXCLUDED | Spam is not phishing and not legitimate enterprise mail |
| **Total in file** | **19,800** | | |

### Retained Counts

| Stage | Count | Notes |
|-------|-------|-------|
| Records after type filter | 13,356 | Phishing + Valid only |
| Dropped — empty body | 0 | All LLM-generated records have non-empty body |
| Dropped — exact duplicate | 0 | All bodies are unique |
| **Retained** | **13,356** | **Authoritative figure** |

### Class Distribution

| Class | Count |
|-------|-------|
| Phishing (label=1) | 6,756 |
| Valid (label=0) | 6,600 |
| **Total** | **13,356** |
| Phishing:Valid ratio | 1.02:1 (near-balanced) |

### Metadata Distribution (stratum_iii_metadata.csv)

| Field | Values | Distribution |
|-------|--------|-------------|
| entity_type | well_known, fabricated | 6,678 each (exactly balanced) |
| length_type | short, medium, long | 4,452 each (exactly balanced) |

Note: The perfect balance across entity_type and length_type reflects
PhishFuzzer's designed factorial structure. This is documented as a
characteristic of the synthetic corpus in the Methods chapter.

---

## Stratum I — PENDING (Phase 4)

Stratum I figures will be recorded here after Phase 4 completes.
Stratum I requires:
  1. Enron tarball extraction
  2. Empirical folder audit (enron_folder_audit.md)
  3. Enron ham parser
  4. Nazario phishing parser

---

## Grand Total — PENDING COMPLETION OF ALL STRATA

| Stratum | Legitimate (0) | Phishing (1) | Total | Status |
|---------|----------------|--------------|-------|--------|
| I | TBD | TBD | TBD | Pending Phase 4 |
| II | 2,763 | 5,085 | 7,848 | CONFIRMED |
| III | 6,600 | 6,756 | 13,356 | CONFIRMED |
| **Grand Total** | **TBD** | **TBD** | **TBD** | Pending Phase 4 (Stratum I) |

This table will be completed and the document status changed to
FULLY LOCKED upon completion of Phases 3 and 4.