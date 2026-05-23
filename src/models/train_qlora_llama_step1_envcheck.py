"""
train_qlora_llama_step1_envcheck.py
"""

import importlib
import os
import sys
from pathlib import Path

REQUIRED = [
    ("torch",        "torch"),
    ("transformers", "transformers"),
    ("peft",         "peft"),
    ("bitsandbytes", "bitsandbytes"),
    ("accelerate",   "accelerate"),
    ("pandas",       "pandas"),
    ("numpy",        "numpy"),
    ("scikit-learn", "sklearn"),
    ("scipy",        "scipy"),
    ("tqdm",         "tqdm"),
]

OPTIONAL = [
    ("captum", "captum"),
]

MIN_VRAM_GB = 20.0

ROOT = Path(__file__).resolve().parents[2]
DATA_PROCESSED = ROOT / "data" / "processed"
SPLITS_DIR     = DATA_PROCESSED / "splits"

SPLIT_FILES = [
    ("train_pooled.csv",           123_516),
    ("val_pooled.csv",              26_468),
    ("test_pooled.csv",             26_468),
    ("test_stratum_i.csv",          23_288),
    ("test_stratum_ii.csv",          1_177),
    ("test_stratum_iii.csv",         2_004),
]

CANONICAL_CSVS = [
    ("stratum_i/stratum_i_combined.csv",    155_250),
    ("stratum_ii/stratum_ii_combined.csv",    7_846),
    ("stratum_iii/stratum_iii_combined.csv", 13_356),
]


def _row_count(path: Path) -> int:
    with open(path, "rb") as f:
        return sum(1 for _ in f) - 1  # subtract header


def main():
    print("=" * 60)
    print("Phase 10 -- QLoRA Llama-3-8B: Environment Check")
    print("=" * 60)

    all_ok = True

    # ── Required packages ────────────────────────────────────────
    print("\nRequired packages:")
    for display, import_name in REQUIRED:
        try:
            mod = importlib.import_module(import_name)
            ver = getattr(mod, "__version__", "unknown")
            print(f"  [OK]  {display:<22s} {ver}")
        except ImportError:
            print(f"  [XX]  {display:<22s} NOT INSTALLED  <-- fix this")
            all_ok = False

    # ── Optional packages ────────────────────────────────────────
    print("\nOptional packages (needed for explainability in Step 5):")
    for display, import_name in OPTIONAL:
        try:
            mod = importlib.import_module(import_name)
            ver = getattr(mod, "__version__", "unknown")
            print(f"  [OK]  {display:<22s} {ver}")
        except ImportError:
            print(f"  [--]  {display:<22s} NOT INSTALLED  (install before Step 5)")

    # ── CUDA / GPU ───────────────────────────────────────────────
    print("\nCUDA / GPU:")
    try:
        import torch
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                vram_gb = props.total_memory / (1024 ** 3)
                print(f"  GPU {i}: {props.name}  ({vram_gb:.1f} GB VRAM)")
            print(f"  CUDA version:  {torch.version.cuda}")
            print(f"  PyTorch build: {torch.__version__}")
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            if vram_gb >= MIN_VRAM_GB:
                print(f"\n  [OK] {vram_gb:.1f} GB VRAM -- sufficient for training.")
            else:
                print(f"\n  [XX] {vram_gb:.1f} GB VRAM -- insufficient (need >= {MIN_VRAM_GB} GB).")
                all_ok = False
        else:
            print("  [XX] No CUDA GPU detected.")
            all_ok = False
    except ImportError:
        print("  [XX] torch not installed -- cannot check GPU.")
        all_ok = False

    # ── HuggingFace token ────────────────────────────────────────
    print("\nHuggingFace access (Llama-3 is a gated model):")
    hf_token = os.environ.get("HF_TOKEN", "")
    if hf_token:
        print(f"  [OK] HF_TOKEN found in environment (starts with: {hf_token[:10]}...)")
    else:
        print("  [XX] HF_TOKEN not set -- export HF_TOKEN=hf_...")
        all_ok = False

    # ── Project paths ────────────────────────────────────────────
    print("\nProject paths:")
    print(f"  Project root:  {ROOT}")

    if DATA_PROCESSED.exists():
        print(f"  [OK] data/processed: {DATA_PROCESSED}")
    else:
        print(f"  [XX] data/processed missing: {DATA_PROCESSED}")
        all_ok = False

    if SPLITS_DIR.exists():
        print(f"  [OK] splits dir: {SPLITS_DIR}")
    else:
        print(f"  [XX] splits dir missing: {SPLITS_DIR}")
        all_ok = False

    print("\n  Checking split files:")
    for fname, expected_rows in SPLIT_FILES:
        fpath = SPLITS_DIR / fname
        if fpath.exists():
            n = _row_count(fpath)
            print(f"    [OK] {fname:<35s} ({n:,} rows)")
        else:
            print(f"    [XX] {fname:<35s} MISSING")
            all_ok = False

    print("\n  Checking canonical CSVs:")
    for rel_path, expected_rows in CANONICAL_CSVS:
        fpath = DATA_PROCESSED / rel_path
        if fpath.exists():
            n = _row_count(fpath)
            print(f"    [OK] {rel_path:<45s} ({n:,} rows)")
        else:
            print(f"    [XX] {rel_path:<45s} MISSING")
            all_ok = False

    # ── Result ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    if all_ok:
        print("ALL CHECKS PASSED. Proceed to Step 2.")
    else:
        print("FAILED. Fix the [XX] issues above before proceeding.")
    print("=" * 60)
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
