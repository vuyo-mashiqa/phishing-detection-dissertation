"""
train_qlora_llama_step1_envcheck.py

"""

import sys
import importlib
import os

# Force UTF-8 output on Windows so Unicode chars don't crash cp1252
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

# ── Required packages ─────────────────────────────────────────────────────────
REQUIRED = {
    "torch":         "2.2.0",
    "transformers":  "4.40.0",
    "peft":          "0.10.0",
    "bitsandbytes":  "0.43.0",
    "accelerate":    "0.29.0",
    "pandas":        "2.2.0",
    "numpy":         "1.26.0",
    "scikit-learn":  "1.4.0",
    "scipy":         "1.13.0",
    "tqdm":          "4.66.0",
}

OPTIONAL = {
    "captum": "0.7.0",  # needed for Step 5 explainability only
}

print("=" * 60)
print("Phase 10 -- QLoRA Llama-3-8B: Environment Check")
print("=" * 60)

all_ok = True

print("\nRequired packages:")
for pkg, min_ver in REQUIRED.items():
    try:
        mod = importlib.import_module(pkg.replace("-", "_"))
        ver = getattr(mod, "__version__", "unknown")
        print(f"  [OK]  {pkg:<22s} {ver}")
    except ImportError:
        print(f"  [XX]  {pkg:<22s} NOT INSTALLED  <-- fix this")
        all_ok = False

print("\nOptional packages (needed for explainability in Step 5):")
for pkg, min_ver in OPTIONAL.items():
    try:
        mod = importlib.import_module(pkg)
        ver = getattr(mod, "__version__", "unknown")
        print(f"  [OK]  {pkg:<22s} {ver}")
    except ImportError:
        print(f"  [--]  {pkg:<22s} NOT INSTALLED  (install before Step 5)")

# ── CUDA / GPU check ──────────────────────────────────────────────────────────
print("\nCUDA / GPU:")
try:
    import torch
    if torch.cuda.is_available():
        n = torch.cuda.device_count()
        min_vram = float("inf")
        for i in range(n):
            name = torch.cuda.get_device_name(i)
            vram = torch.cuda.get_device_properties(i).total_memory / 1e9
            min_vram = min(min_vram, vram)
            print(f"  GPU {i}: {name}  ({vram:.1f} GB VRAM)")
        print(f"  CUDA version:  {torch.version.cuda}")
        print(f"  PyTorch build: {torch.__version__}")

        if min_vram < 10:
            print("\n  [XX] ERROR: Less than 10 GB VRAM detected.")
            print("       Llama-3-8B NF4 requires ~10-12 GB minimum.")
            all_ok = False
        elif min_vram < 16:
            print(f"\n  [!!] WARNING: {min_vram:.1f} GB VRAM -- training is tight.")
            print("       Use batch_size=1 and grad_accum=32 in Step 3.")
            print("       Effective batch size will still be 32.")
        else:
            print(f"\n  [OK] {min_vram:.1f} GB VRAM -- sufficient for training.")
    else:
        print("  [XX] CUDA not available. GPU is required for QLoRA training.")
        all_ok = False

except Exception as e:
    print(f"  [XX] ERROR during CUDA check: {e}")
    all_ok = False

# ── HuggingFace token check ───────────────────────────────────────────────────
print("\nHuggingFace access (Llama-3 is a gated model):")
hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
if hf_token:
    print(f"  [OK] HF_TOKEN found in environment (starts with: {hf_token[:8]}...)")
else:
    print("  [!!] HF_TOKEN not set in environment.")
    print("       Accept the Llama-3 licence at:")
    print("       https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct")
    print("       Then run:  set HF_TOKEN=hf_xxxxxxxxxxxxxxxx  (Windows)")
    print("       Or pass:   --hf_token hf_xxxxxxxxxxxxxxxx   (Step 3 argument)")

# ── Project path sanity check ─────────────────────────────────────────────────
print("\nProject paths:")
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
print(f"  Project root:  {ROOT}")

CANON_DIR  = ROOT / "data" / "processed"
SPLITS_DIR = ROOT / "data" / "processed" / "splits"

for label, path in [("data/processed", CANON_DIR), ("splits dir", SPLITS_DIR)]:
    if path.exists():
        print(f"  [OK] {label}: {path}")
    else:
        print(f"  [XX] {label} NOT FOUND: {path}")
        print("       Make sure you are running from inside the project repo.")
        all_ok = False

# ── Split files check (same files used by train_transformer.py) ───────────────
# QLoRA trains on pooled only, but evaluates on all three per-stratum test sets
expected_splits = [
    "train_pooled.csv",
    "val_pooled.csv",
    "test_pooled.csv",
    "test_stratum_i.csv",
    "test_stratum_ii.csv",
    "test_stratum_iii.csv",
]
print("\n  Checking split files:")
if SPLITS_DIR.exists():
    import pandas as pd
    for fname in expected_splits:
        p = SPLITS_DIR / fname
        if p.exists():
            n = len(pd.read_csv(p, usecols=["message_id"]))
            print(f"    [OK] {fname:<28s} ({n:,} rows)")
        else:
            print(f"    [XX] {fname:<28s} NOT FOUND")
            all_ok = False

# ── Canonical CSV check (same as train_transformer.py _load_canonical) ────────
print("\n  Checking canonical CSVs:")
STRATA_DIRS = {
    "stratum_i":   "stratum_i_combined.csv",
    "stratum_ii":  "stratum_ii_combined.csv",
    "stratum_iii": "stratum_iii_combined.csv",
}
if CANON_DIR.exists():
    import pandas as pd
    for stratum, fname in STRATA_DIRS.items():
        p = CANON_DIR / stratum / fname
        if p.exists():
            n = len(pd.read_csv(p, usecols=["message_id"]))
            print(f"    [OK] {stratum}/{fname:<32s} ({n:,} rows)")
        else:
            print(f"    [XX] {stratum}/{fname}  NOT FOUND")
            all_ok = False

# ── Final verdict ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
if all_ok:
    print("ALL CHECKS PASSED. Safe to proceed to Step 2.")
else:
    print("FAILED. Fix the [XX] issues above before proceeding.")
    sys.exit(1)
    