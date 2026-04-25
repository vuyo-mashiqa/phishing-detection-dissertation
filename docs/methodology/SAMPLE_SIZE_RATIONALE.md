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

## Stratum III — PENDING (Phase 3)

Stratum III figures will be recorded here after Phase 3 completes.
Expected phishing records: 6,756 (from DATA_CARD filter logic)
Expected legitimate records: 6,600 (from DATA_CARD filter logic)
These are pre-parser estimates based on PhishFuzzer JSON schema
inspection. The authoritative figures will replace these upon
completion of the Stratum III parser.

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

| Stratum | Ham | Phishing | Total | Status |
|---------|-----|----------|-------|--------|
| I | TBD | TBD | TBD | Pending Phase 4 |
| II | 2,763 | 5,085 | 7,848 | CONFIRMED |
| III | TBD | TBD | TBD | Pending Phase 3 |
| **Grand Total** | **TBD** | **TBD** | **TBD** | |

This table will be completed and the document status changed to
FULLY LOCKED upon completion of Phases 3 and 4.