# Phishing Detection Dissertation

**MSc Cybersecurity and Digital Forensics — University of Portsmouth**

**Title:** Defending Against AI-Generated Phishing Emails: Evaluating Large-Language Model-Based Detection Under Realistic Organisational Constraints

**Author:** Vuyo Mashiqa\
**Supervisor:** Dr Linda Yang

---

## Project Overview

This repository contains all code, processed data, documentation, and results for the above dissertation.
The research constructs a three-stratum evaluation framework to compare classical ML, transformer, and LLM-based phishing detection systems against legacy, contemporary, and LLM-generated email corpora under operational constraints (latency, cost, privacy, explainability).

## Repository Structure

| Folder | Contents |
|--------|----------|
| `data/processed/` | Canonical CSV outputs from each stratum parser |
| `data/splits/` | Frozen train/val/test splits |
| `docs/` | Schema definitions, data governance, dissertation chapter drafts |
| `outputs/` | Manifests, validation reports, audit logs |
| `results/` | Model metrics, figures, explanations, latency benchmarks |
| `src/` | All source code: parsers, feature engineering, models, evaluation |
| `tests/` | Automated test suite |
| `prompts/` | Versioned LLM prompt files |

## Reproduction Instructions

See `docs/methodology/REPRODUCTION.md` once complete.
