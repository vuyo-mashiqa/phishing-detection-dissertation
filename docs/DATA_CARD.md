# DATA_CARD.md
## Dataset Governance Document
**Project:** Defending Against AI-Generated Phishing Emails: Evaluating
Large-Language Model-Based Detection Under Realistic Organisational Constraints\
**Author:** Vuyo Mashiqa\
**Supervisor:** Dr Linda Yang\
**Institution:** University of Portsmouth, Faculty of Technology\
**Programme:** MSc Cybersecurity and Digital Forensics\
**Version:** 1.0\
**Date:** 2026-04-23\

---

## 1. Purpose of This Document

This document is the single authoritative governance record for all datasets
used in this dissertation. It records, for each corpus: source provenance,
licence terms, temporal coverage, PII risk assessment, contamination status,
role boundaries within the three-stratum design, inclusion and exclusion
criteria, and deduplication policy.

No dataset decision exists outside this file.

---

## 2. Three-Stratum Design Summary

| Stratum | Role | Phishing Source | Legitimate Source |
|---------|------|-----------------|-------------------|
| I — Backward Comparability | Anchor to legacy benchmarks | Nazario (5 mbox files) | Enron ham (received-only, audited folders) |
| II — Contemporary General | Modern pre-LLM threat surface | phishing_pot (2022–2023) | CSDMC2010 ham |
| III — LLM-Generated | LLM-era threat surface | PhishFuzzer Type=Phishing LLM variants | PhishFuzzer Type=Valid LLM variants |

Performance on Stratum I establishes methodological comparability with prior
work. It is NOT a validity claim about current-threat detection capability.
The primary dissertation findings rest on Strata II and III.

---

## 3. Stratum I — Backward Comparability Corpus

### 3.1 Enron Email Corpus (Legitimate Class)

| Field | Value |
|-------|-------|
| Role | Legitimate (ham) email — Stratum I ONLY |
| Source URL | https://www.cs.cmu.edu/~enron/enron_mail_20150507.tar.gz |
| Acquisition date | 2026-04-12 |
| File path | data/raw/stratum_i/enron_mail_20150507.tar.gz |
| Size | 443,254,787 bytes (422.7 MB) |
| SHA-256 | b3da1b3fe0369ec3140bb4fbce94702c33b7da810ec15d718b3fadf5cd748ca7 |
| Licence | Public domain — released under US Federal court order during the FERC investigation of Enron Corporation (2003). No copyright restriction on research use. |
| Temporal coverage | October 1999 – June 2002 (bulk correspondence: 2000–2001) |
| PII risk | HIGH. Noever (2020) documents over 50,000 PII instances in the corpus including names, addresses, phone numbers, financial figures, and personal communications. Raw corpus is stored locally only and is excluded from version control via .gitignore. Processed CSV outputs used for training contain cleaned text but inherently carry residual PII. No raw email text, sender addresses, or identifiable content will appear in any published artefact, dissertation appendix, or supplementary material. |
| Contamination status | CLEAN. Verified structurally independent of Strata II and III. Note: certain composite public corpora including TREC 2007 and several Kaggle spam datasets embed Enron messages in their ham classes. None of those composite corpora are used in this study. |
| Role boundary | Stratum I ONLY. Results on this stratum establish whether the models implemented here reproduce the performance figures reported in the prior literature (Salloum et al., 2022; Meléndez et al., 2024), confirming experimental fidelity. Performance on Stratum I is explicitly NOT interpreted as evidence of current-threat detection capability. The corpus predates SPF, DKIM, and DMARC deployment (Koide et al., 2024) and the phishing landscape it pairs with has been superseded. |
| Inclusion rule | Received-mail folders only. See docs/methodology/enron_folder_audit.md for the complete empirical inclusion/exclusion audit. |
| Exclusion categories | Sent mail, sent items, calendar, notes, tasks, drafts, deleted items, discussion threads, all-documents catch-all archives — all outgoing, administrative, and non-received-mail content excluded. Rationale: a deployed phishing detector operates on received mail only; training on outgoing mail introduces distribution shift. |

### 3.2 Nazario Phishing Corpus (Phishing Class)

| Field | Value |
|-------|-------|
| Role | Phishing email — Stratum I ONLY |
| Source URL | https://monkey.org/~jose/phishing/ |
| Acquisition date | 2026-04-12 |
| Files used | phishing0.mbox, phishing1.mbox, phishing2.mbox, phishing3.mbox, 20051114.mbox |
| SHA-256 hashes | phishing0: 6184b0a34ed8cbbb252676e21c1eab9eac5fa4be89c61e30cc9954ebbb25d848 |
| | phishing1: 99f02474a11086408a5ce6d11ef38823f003c6611a4312c31fbee96773eeba28 |
| | phishing2: 5113277984eae759a5ff958ccf408b542441ce033a49b923ab3b0f324075e6bd |
| | phishing3: b29336d2e31c2dff19639415e98ed2aced5b4d63675ea5e29bea1a7d4e452841 |
| | 20051114: 928cbf1c394d01d0b398ded8dc7cbd94dd4ce9cb0d4efdecd12696338c247926 |
| Combined size | 41,870,807 bytes across 5 files |
| Licence | Public domain / academic research use (curator: Jose Nazario) |
| Temporal coverage | Approximately 2003–2007, consistent with Enron correspondence era |
| PII risk | LOW. Phishing bait emails; no private personal correspondence. |
| Contamination status | CLEAN. Structurally independent of Strata II and III. |
| Role boundary | Stratum I ONLY. Temporal alignment with Enron (2000–2002 ham, ~2003–2007 phishing) is the governing design constraint for this stratum. |

#### Nazario Inclusion/Exclusion Rule — Definitive and Final

**Files INCLUDED (5 total):**
phishing0.mbox, phishing1.mbox, phishing2.mbox, phishing3.mbox, 20051114.mbox

**Files EXCLUDED:**
phishing-2015.mbox through phishing-2025.mbox (yearly collection files)

**Exclusion rationale:**
The yearly Nazario files (phishing-2015 onward) represent a materially
different threat era — post-ChatGPT precursor, post-DMARC adoption,
post-mobile-first phishing evolution. Including them in Stratum I would
corrupt the backward-comparability purpose of the stratum by mixing
two incompatible threat distributions. The five included files cover
approximately 2003–2007, which is temporally consistent with the Enron
legitimate mail corpus (2000–2002) and directly mirrors the benchmark
configuration used in Salloum et al. (2022), the authoritative survey
of classical ML phishing detection.

---

## 4. Stratum II — Contemporary General Phishing Corpus

### 4.1 phishing_pot (Phishing Class)

| Field | Value |
|-------|-------|
| Role | Phishing email — Stratum II ONLY |
| Source URL | https://github.com/rf-peixoto/phishing_pot |
| Repository archived | 2025-01-08 |
| Acquisition date | 2026-04-12 |
| File path | data/raw/stratum_ii/phishing_pot/ |
| EML file count | 7,911 |
| Total size | 313,754,988 bytes (299.2 MB) as counted by corpus_sha256_manifest.json |
| Licence | Public repository — academic and research use |
| Temporal coverage | August 2022 – October 2023 |
| Languages | 19 |
| Brands impersonated | 193 |
| PII risk | LOW. Phishing bait emails collected by a honeypot; no private personal correspondence. |
| Contamination status | CLEAN. Structurally and temporally independent of Enron and Nazario (different source, different era, different collection mechanism). |
| Literature justification | Koide et al. (2024) constructed the ChatSpamDetector evaluation corpus using phishing_pot on the explicit grounds that legacy benchmarks predate the deployment of SPF, DKIM, and DMARC authentication standards and therefore cannot characterise the difficulty of detecting contemporary attacks. This study adopts the same corpus for the same reason. Afane et al. (2024) demonstrate that post-authentication-era email constitutes a materially harder detection surface than Enron-vintage legitimate mail, further validating the choice. |
| Role boundary | Stratum II ONLY. Contemporary phishing class. |

### 4.2 CSDMC2010 (Legitimate Class)

| Field | Value |
|-------|-------|
| Role | Legitimate (ham) email — Stratum II ONLY |
| Source URL | https://github.com/zrz1996/Spam-Email-Classifier-DataSet |
| Acquisition date | 2026-04-12 |
| File path | data/raw/stratum_ii/csdmc2010/ |
| Raw file count | 2,955 files (excluding __MACOSX artefact directory) |
| Total size | 10,526,242 bytes (10.0 MB) excluding __MACOSX |
| Licence | Academic / research use (Shams & Mercer, 2014) |
| Temporal coverage | Approximately 2010 |
| PII risk | LOW. Randomly sampled inbox messages; no litigation-derived or named-individual PII at scale. Shams and Mercer (2014) document collection from random inboxes with no single-organisation concentration. |
| Contamination status | CLEAN. Shams and Mercer (2014) document collection from random inboxes, structurally independent of Enron. No Enron content present. |
| Role boundary | Stratum II ONLY (legitimate class). |

#### CSDMC2010 Temporal Limitation — Explicit Statement

CSDMC2010 dates to approximately 2010. This is a known and acknowledged
limitation of the Stratum II legitimate class.

The ham side of Stratum II is chosen for two reasons that do not depend
on temporal recency: (1) structural independence from Enron, which is
essential to the cross-stratum validity of the design (Noever, 2020;
Shams & Mercer, 2014), and (2) documented random-inbox collection
methodology, which avoids the single-organisation concentration bias
of Enron.

The phishing side of Stratum II (phishing_pot, August 2022 – October 2023)
carries the primary temporal contemporaneity claim for this stratum.

---

## 5. Stratum III — LLM-Generated Corpus

### 5.1 PhishFuzzer Dataset — LLM Variants Only (Both Classes)

| Field | Value |
|-------|-------|
| Role | BOTH phishing and legitimate email — Stratum III ONLY |
| Source URL | https://github.com/DataPhish/PhishFuzzer |
| Acquisition date | 2026-04-12 |
| Licence | Creative Commons Attribution 4.0 International (CC BY 4.0) |
| File USED | PhishFuzzer_emails_entity_rephrased_v1.json (variants) |
| File EXCLUDED | PhishFuzzer_emails_original_seed_v1.json (seeds — never opened or read in any script) |
| Variants file size | 48,092,464 bytes (45.9 MB) |
| Variants SHA-256 | 2e2a4da749b38de5869d3e1901222a0cf000c18a910d13faf4f1b0054d70b38d |
| Seeds SHA-256 | afc88e4dbc000d9b45a9f3d65a82064b956ab677d1c2874526b0ca179e003765 |

#### Filter Logic — Definitive

| Filter | Value | Records |
|--------|-------|---------|
| Type = "Phishing" AND Created by = "LLM" | Phishing class (label=1) | 6,756 |
| Type = "Valid" AND Created by = "LLM" | Legitimate class (label=0) | 6,600 |
| Type = "Spam" | EXCLUDED — not used | 6,444 |
| Total in variants file | | 19,800 |

Spam records are excluded because spam is a categorically different
phenomenon from phishing — it is unsolicited commercial mail rather
than deceptive credential-harvesting. Including spam in either class
would corrupt the binary phishing/legitimate classification task.

#### Seed File Exclusion Rationale

PhishFuzzer_emails_original_seed_v1.json contains the real-world emails
used as seeds for LLM generation. These seeds overlap with real-world
corpora and may share content with emails present in Strata I and II.
Using them in Stratum III would violate the mutual exclusivity requirement
between strata and introduce cross-stratum contamination. The seed file
SHA-256 hash is recorded in the integrity manifest solely to document
what was excluded and why. The file is never opened or read by any
script in this project.

#### Stylometric Confound Control

Both the phishing class and the legitimate class for Stratum III are
drawn from the same LLM generation pipeline (PhishFuzzer entity-rephrased
variants). This is a deliberate methodological decision addressing the
stylometric confound identified by Eze and Shamir (2024).

If phishing emails are LLM-generated but legitimate emails are
human-written, any classifier trained on this data may learn to separate
writing registers — AI-style text vs human-style text — rather than
learning to detect phishing intent. Because both classes originate from
the same generation process, register differences are controlled. Any
residual discriminative signal must reflect content intent, not
generation source. This makes Stratum III the cleanest test of whether
a model has genuinely learned to detect phishing semantics rather than
stylistic proxies.

---

## 6. Cross-Stratum Contamination Policy

### 6.1 Independence Requirements

Each stratum must be mutually exclusive from the others. An email that
appears in more than one stratum would allow a model trained on one
stratum to achieve artificially high performance on another by memorising
individual examples rather than learning generalisable patterns.

### 6.2 Contamination Checks Performed

| Check | Method | Status                                    |
|-------|--------|-------------------------------------------|
| Stratum I vs II source independence | Corpus provenance review | CLEAN                                     |
| Stratum I vs III source independence | Corpus provenance review | CLEAN                                     |
| Stratum II vs III source independence | Corpus provenance review | CLEAN                                     |
| Within-stratum exact duplicates | SHA-256 body hash | Applied at parse time                     |
| Cross-stratum near-duplicates | MinHash-LSH (128 perms, Jaccard ≥ 0.85) | Applied after all three strata are parsed |

The MinHash-LSH cross-stratum audit is conducted after all three strata
are parsed. Results are recorded in
outputs/validation/cross_stratum_leakage_report.json.

---

## 7. Deduplication Policy

Within each stratum, exact duplicate removal is applied during parsing:
- Normalise body text: lowercase, collapse all whitespace to single spaces,
  strip HTML tags
- Compute SHA-256 of normalised text
- If hash already seen within the corpus, discard the duplicate

Cross-stratum near-duplicate removal is applied after all strata are
parsed, using MinHash-LSH with 128 permutations and a Jaccard similarity
threshold of 0.85. Any email with a near-duplicate in another stratum
is removed from the lower-priority stratum (III > II > I, meaning if a
near-duplicate spans Strata II and III, the copy in Stratum III is removed).

---

## 8. Split Assignment Policy

Train/validation/test splits are generated exactly once using a fixed
random seed of 42. The split ratio is 70% train / 15% validation /
15% test, stratified on the label column to preserve class balance
within every split.

After generation, split file SHA-256 hashes are recorded in
outputs/manifests/splits_manifest.json. Splits are never regenerated
without a documented methodology change committed to this repository.

---

## 9. Integrity Manifest Reference

All SHA-256 hashes recorded in this document are sourced from:
outputs/manifests/corpus_sha256_manifest.json
Generated: 2026-04-23T11:22:09.278376+00:00

This manifest is the authoritative cryptographic integrity record
for this project.