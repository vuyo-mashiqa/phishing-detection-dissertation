"""
train_qlora_llama_step2_prepare_datasets.py
Phase 10 -- QLoRA Llama-3-8B: Dataset Preparation

The split CSVs (data/processed/splits/) contain message_id + structural features
but NOT the raw text. Subject and body are joined from the canonical CSVs
(data/processed/stratum_{i,ii,iii}/stratum_{i,ii,iii}_combined.csv) on message_id.
This is the same source-of-truth used by train_transformer.py (load_canonical).

Writes Llama-3 instruction-format JSONL files to data/processed/qlora/:
  train_pooled.jsonl        (123,516 rows)
  val_pooled.jsonl          (26,468  rows)
  test_pooled.jsonl         (26,468  rows)
  test_stratum_i.jsonl      (23,288  rows)
  test_stratum_ii.jsonl     (1,177   rows)
  test_stratum_iii.jsonl    (2,004   rows)

Run AFTER Step 1 passes. Do NOT proceed to Step 3 until this script exits
with "ALL FILES WRITTEN. Safe to proceed to Step 3."
"""

import json
import hashlib
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT         = Path(__file__).resolve().parents[2]
SPLITS_DIR   = ROOT / "data" / "processed" / "splits"
CANON_DIR    = ROOT / "data" / "processed"
OUT_DIR      = ROOT / "data" / "processed" / "qlora"
MANIFEST_DIR = ROOT / "outputs" / "manifests"

OUT_DIR.mkdir(parents=True, exist_ok=True)
MANIFEST_DIR.mkdir(parents=True, exist_ok=True)

# ── Canonical CSV paths (mirrors load_canonical in train_transformer.py) ─────
CANONICAL_CSVS = {
    "stratum_i":   CANON_DIR / "stratum_i"   / "stratum_i_combined.csv",
    "stratum_ii":  CANON_DIR / "stratum_ii"  / "stratum_ii_combined.csv",
    "stratum_iii": CANON_DIR / "stratum_iii" / "stratum_iii_combined.csv",
}

# ── Split file map ────────────────────────────────────────────────────────────
SPLIT_MAP = {
    "train_pooled":     "train_pooled.csv",
    "val_pooled":       "val_pooled.csv",
    "test_pooled":      "test_pooled.csv",
    "test_stratum_i":   "test_stratum_i.csv",
    "test_stratum_ii":  "test_stratum_ii.csv",
    "test_stratum_iii": "test_stratum_iii.csv",
}

EXPECTED_ROWS = {
    "train_pooled":     123_516,
    "val_pooled":        26_468,
    "test_pooled":       26_468,
    "test_stratum_i":    23_288,
    "test_stratum_ii":    1_177,
    "test_stratum_iii":   2_004,
}

# ── Prompt construction ───────────────────────────────────────────────────────
# Truncation at 1,800 chars targets the same 512-token max sequence length
# used by train_transformer.py (Methods §1.10.1). At ~3.5 chars/token,
# 1,800 chars ≈ 514 tokens, leaving headroom for special tokens.
# Right-truncation preserves subject and opening body content where
# phishing signals are concentrated (Methods §1.10.1).
MAX_EMAIL_CHARS = 1_800

SYSTEM_PROMPT = (
    "You are an expert email security analyst. "
    "Your task is to classify the following email as either legitimate or phishing. "
    "Respond with exactly one word: legitimate or phishing."
)

def build_prompt(subject: str, body: str) -> str:
    subject = str(subject).strip() if pd.notna(subject) and subject else ""
    body    = str(body).strip()    if pd.notna(body)    and body    else ""

    email_text = f"Subject: {subject}\n\n{body}" if subject else body

    # Right-truncate -- matches transformer right-truncation (Methods §1.10.1)
    if len(email_text) > MAX_EMAIL_CHARS:
        email_text = email_text[:MAX_EMAIL_CHARS]

    return (
        f"<|begin_of_text|>"
        f"<|start_header_id|>system<|end_header_id|>\n\n"
        f"{SYSTEM_PROMPT}"
        f"<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n"
        f"{email_text}"
        f"<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n"
    )

def label_to_text(label: int) -> str:
    return "phishing" if int(label) == 1 else "legitimate"

# ── Helpers ───────────────────────────────────────────────────────────────────
def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"

# ── Build global message_id → (subject, body) lookup from canonical CSVs ─────
def build_text_lookup() -> dict:
    """
    Loads subject and body from all three canonical CSVs into a single dict
    keyed on message_id. Exactly mirrors how train_transformer.py calls
    load_canonical() to resolve text for each split row.
    """
    print("Building text lookup from canonical CSVs...")
    lookup = {}
    for strat, csv_path in CANONICAL_CSVS.items():
        if not csv_path.exists():
            print(f"  [XX] Canonical CSV NOT FOUND: {csv_path}")
            sys.exit(1)
        df = pd.read_csv(
            csv_path,
            usecols=["message_id", "subject", "body"],
            dtype={"message_id": str, "subject": str, "body": str},
        )
        before = len(lookup)
        for _, row in df.iterrows():
            lookup[row["message_id"]] = (row["subject"], row["body"])
        print(f"  [OK] {strat}: {len(df):,} rows loaded "
              f"(lookup size now {len(lookup):,})")
    print(f"  Total lookup entries: {len(lookup):,}\n")
    return lookup

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("Phase 10 -- QLoRA Llama-3-8B: Dataset Preparation")
    print("=" * 60)
    print()

    all_ok   = True
    manifest = {
        "step":            "step2_prepare_datasets",
        "git_sha":         git_sha(),
        "timestamp":       time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "max_email_chars": MAX_EMAIL_CHARS,
        "files":           {},
    }

    # Build lookup once -- shared across all splits
    text_lookup = build_text_lookup()

    for split_name, csv_name in SPLIT_MAP.items():
        csv_path   = SPLITS_DIR / csv_name
        jsonl_path = OUT_DIR / f"{split_name}.jsonl"

        print(f"[{split_name}]")

        if not csv_path.exists():
            print(f"  [XX] Split CSV NOT FOUND: {csv_path}")
            all_ok = False
            continue

        split_df = pd.read_csv(
            csv_path,
            usecols=["message_id", "label"],
            dtype={"message_id": str, "label": int},
        )
        n_rows = len(split_df)
        print(f"  Loaded {n_rows:,} rows from {csv_name}")

        expected = EXPECTED_ROWS[split_name]
        if n_rows != expected:
            print(f"  [XX] Row count mismatch: got {n_rows:,}, expected {expected:,}")
            all_ok = False

        # Resolve text via lookup
        missing = 0
        n_phish = 0
        n_ham   = 0

        with open(jsonl_path, "w", encoding="utf-8") as f:
            for _, row in split_df.iterrows():
                mid   = str(row["message_id"])
                label = int(row["label"])

                if mid in text_lookup:
                    subject, body = text_lookup[mid]
                else:
                    # Should never happen if data pipeline is intact
                    subject, body = "", ""
                    missing += 1

                record = {
                    "message_id": mid,
                    "prompt":     build_prompt(subject, body),
                    "completion": label_to_text(label),
                    "label":      label,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

                if label == 1:
                    n_phish += 1
                else:
                    n_ham += 1

        if missing > 0:
            print(f"  [XX] {missing:,} message_ids had no text in lookup "
                  f"(canonical CSV mismatch -- check data pipeline)")
            all_ok = False

        sha = file_sha256(jsonl_path)
        manifest["files"][split_name] = {
            "source_csv": csv_name,
            "jsonl_file": jsonl_path.name,
            "n_rows":     n_rows,
            "n_phishing": n_phish,
            "n_ham":      n_ham,
            "n_missing":  missing,
            "sha256":     sha,
        }

        print(f"  [OK] Written {n_rows:,} rows  "
              f"(phishing={n_phish:,}  ham={n_ham:,}  missing={missing})")
        print(f"       {jsonl_path.name}  sha256={sha[:16]}...")
        print()

    # ── Spot-check: first and last record of train_pooled.jsonl ──────────────
    spot_path = OUT_DIR / "train_pooled.jsonl"
    if spot_path.exists():
        print("[Spot-check] First and last record of train_pooled.jsonl:")
        with open(spot_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        for idx in [0, -1]:
            rec = json.loads(lines[idx])
            print(f"  message_id : {rec['message_id']}")
            print(f"  label      : {rec['label']}  ({rec['completion']})")
            print(f"  prompt_len : {len(rec['prompt'])} chars")
            snippet = rec["prompt"].replace("\n", " ")[:140]
            print(f"  prompt_head: {snippet!r}")
            print()

    # ── Manifest ──────────────────────────────────────────────────────────────
    ts = time.strftime("%Y%m%d%H%M%S", time.gmtime())
    manifest_path = MANIFEST_DIR / f"qlora_step2_{ts}.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[Manifest] Written to {manifest_path.name}")

    # ── Verdict ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    if all_ok:
        print("ALL FILES WRITTEN. Safe to proceed to Step 3.")
    else:
        print("FAILED. Fix the [XX] issues above before proceeding.")
    print("=" * 60)
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
