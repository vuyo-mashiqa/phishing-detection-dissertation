"""
train_qlora_llama_step2_prepare_datasets.py
QLoRA Llama-3-8B: Dataset Preparation

Reads the canonical split CSVs
and writes Llama-3 instruction-format JSONL files:

  data/processed/qlora/
    train_pooled.jsonl        (123,516 rows -- training)
    val_pooled.jsonl          (26,468 rows  -- validation / threshold calibration)
    test_pooled.jsonl         (26,468 rows  -- pooled test)
    test_stratum_i.jsonl      (23,288 rows  -- cross-stratum eval)
    test_stratum_ii.jsonl     (1,177 rows   -- cross-stratum eval)
    test_stratum_iii.jsonl    (2,004 rows   -- cross-stratum eval)

"""

import json
import hashlib
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

# ── Paths (mirrors train_transformer.py exactly) ────────────────────────────
ROOT       = Path(__file__).resolve().parents[2]
SPLITS_DIR = ROOT / "data" / "processed" / "splits"
OUT_DIR    = ROOT / "data" / "processed" / "qlora"
MANIFEST_DIR = ROOT / "outputs" / "manifests"

OUT_DIR.mkdir(parents=True, exist_ok=True)
MANIFEST_DIR.mkdir(parents=True, exist_ok=True)

# ── Split file map: output name → source CSV in SPLITS_DIR ──────────────────
SPLIT_MAP = {
    "train_pooled":    "train_pooled.csv",
    "val_pooled":      "val_pooled.csv",
    "test_pooled":     "test_pooled.csv",
    "test_stratum_i":  "test_stratum_i.csv",
    "test_stratum_ii": "test_stratum_ii.csv",
    "test_stratum_iii":"test_stratum_iii.csv",
}

# Expected row counts -- same values confirmed by Step 1 and test_splits.py
EXPECTED_ROWS = {
    "train_pooled":     123_516,
    "val_pooled":        26_468,
    "test_pooled":       26_468,
    "test_stratum_i":    23_288,
    "test_stratum_ii":    1_177,
    "test_stratum_iii":   2_004,
}

# ── Prompt construction ──────────────────────────────────────────────────────
# Max characters of email text passed to the model.
# Llama-3-8B context = 8,192 tokens; we use max_seq_len=512 (matching
# transformers for comparability -- Methods §1.10.1).
# At ~3.5 chars/token, 512 tokens ≈ 1,792 chars.
# We use 1,800 chars as the truncation limit for the combined subject+body
# so the prompt + special tokens comfortably fits in 512 tokens.
MAX_EMAIL_CHARS = 1_800

SYSTEM_PROMPT = (
    "You are an expert email security analyst. "
    "Your task is to classify the following email as either legitimate or phishing. "
    "Respond with exactly one word: legitimate or phishing."
)

def build_prompt(subject: str, body: str) -> str:
    """
    Llama-3 chat-template format.
    Combines subject and body, truncates to MAX_EMAIL_CHARS.
    Format mirrors the input representation used by train_transformer.py:
    subject text is prepended to body text with a separator, matching the
    tokenisation input '[SEP]'.join([subject, body]) used in the transformer
    EmailDataset.
    """
    subject = str(subject).strip() if subject else ""
    body    = str(body).strip()    if body    else ""

    if subject:
        email_text = f"Subject: {subject}\n\n{body}"
    else:
        email_text = body

    # Truncate from the right -- preserves subject and opening body content
    # where phishing signals are concentrated (Methods §1.10.1).
    if len(email_text) > MAX_EMAIL_CHARS:
        email_text = email_text[:MAX_EMAIL_CHARS]

    prompt = (
        f"<|begin_of_text|>"
        f"<|start_header_id|>system<|end_header_id|>\n\n"
        f"{SYSTEM_PROMPT}"
        f"<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n"
        f"{email_text}"
        f"<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    return prompt

def label_to_text(label: int) -> str:
    """Maps integer label to the completion token the model must predict."""
    return "phishing" if int(label) == 1 else "legitimate"

# ── SHA-256 helpers ──────────────────────────────────────────────────────────
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

# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("Phase 10 -- QLoRA Llama-3-8B: Dataset Preparation")
    print("=" * 60)

    all_ok   = True
    manifest = {
        "step":        "step2_prepare_datasets",
        "git_sha":     git_sha(),
        "timestamp":   time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "max_email_chars": MAX_EMAIL_CHARS,
        "files":       {},
    }

    for split_name, csv_name in SPLIT_MAP.items():
        csv_path  = SPLITS_DIR / csv_name
        jsonl_path = OUT_DIR / f"{split_name}.jsonl"

        print(f"\n[{split_name}]")

        # ── Load CSV ────────────────────────────────────────────────────────
        if not csv_path.exists():
            print(f"  [XX] CSV NOT FOUND: {csv_path}")
            all_ok = False
            continue

        df = pd.read_csv(csv_path, dtype={"label": int, "subject": str, "body": str})
        n_rows = len(df)
        print(f"  Loaded {n_rows:,} rows from {csv_name}")

        # Row count check
        expected = EXPECTED_ROWS[split_name]
        if n_rows != expected:
            print(f"  [XX] Row count mismatch: got {n_rows:,}, expected {expected:,}")
            all_ok = False

        # Required columns
        for col in ["message_id", "subject", "body", "label"]:
            if col not in df.columns:
                print(f"  [XX] Missing column: {col}")
                all_ok = False

        if not all_ok:
            continue

        # ── Fill NaN ────────────────────────────────────────────────────────
        df["subject"] = df["subject"].fillna("").astype(str)
        df["body"]    = df["body"].fillna("").astype(str)

        # ── Write JSONL ─────────────────────────────────────────────────────
        n_phish = int((df["label"] == 1).sum())
        n_ham   = int((df["label"] == 0).sum())

        with open(jsonl_path, "w", encoding="utf-8") as f:
            for _, row in df.iterrows():
                record = {
                    "message_id": str(row["message_id"]),
                    "prompt":     build_prompt(row["subject"], row["body"]),
                    "completion": label_to_text(row["label"]),
                    "label":      int(row["label"]),
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        sha = file_sha256(jsonl_path)
        manifest["files"][split_name] = {
            "source_csv":  csv_name,
            "jsonl_file":  jsonl_path.name,
            "n_rows":      n_rows,
            "n_phishing":  n_phish,
            "n_ham":       n_ham,
            "sha256":      sha,
        }

        print(f"  [OK] Written {n_rows:,} rows  "
              f"(phishing={n_phish:,}  ham={n_ham:,})")
        print(f"       {jsonl_path.name}  sha256={sha[:16]}...")

    # ── Spot-check: read back first and last record of train_pooled ─────────
    print("\n[Spot-check] First and last record of train_pooled.jsonl:")
    spot_path = OUT_DIR / "train_pooled.jsonl"
    if spot_path.exists():
        with open(spot_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        for idx in [0, -1]:
            rec = json.loads(lines[idx])
            print(f"  message_id : {rec['message_id']}")
            print(f"  label      : {rec['label']}  ({rec['completion']})")
            print(f"  prompt_len : {len(rec['prompt'])} chars")
            print(f"  prompt_head: {rec['prompt'][:120].replace(chr(10), ' ')!r}")
            print()

    # ── Write manifest ──────────────────────────────────────────────────────
    ts = time.strftime("%Y%m%d%H%M%S", time.gmtime())
    manifest_path = MANIFEST_DIR / f"qlora_step2_{ts}.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[Manifest] Written to {manifest_path.name}")

    # ── Final verdict ───────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    if all_ok:
        print("ALL FILES WRITTEN. Safe to proceed to Step 3.")
    else:
        print("FAILED. Fix the [XX] issues above before proceeding.")
    print("=" * 60)
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
