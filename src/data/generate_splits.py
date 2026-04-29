"""
generate_splits.py
==================
Generates deterministic stratified 70/15/15 train/val/test splits
for each stratum independently, plus a pooled split across all three.

Design decisions (documented for Methods chapter):
    - Stratified on label to preserve class balance in every split.
    - Fixed random_state=42 for full reproducibility.
    - Splits are generated per-stratum AND pooled (all three strata combined).
    - No row appears in more than one split within a stratum.
    - SHA-256 of every output file is recorded in splits_manifest.json.
    - The split files are gitignored (too large); the manifest is committed.

Split ratios:
    Train: 70%   Validation: 15%   Test: 15%

Outputs:
    data/splits/stratum_i/   train.csv  val.csv  test.csv
    data/splits/stratum_ii/  train.csv  val.csv  test.csv
    data/splits/stratum_iii/ train.csv  val.csv  test.csv
    data/splits/pooled/      train.csv  val.csv  test.csv
    outputs/manifests/splits_manifest.json

Run:
    python src/data/generate_splits.py
Then validate:
    pytest tests/test_splits.py -v
"""

import csv
import hashlib
import json
import sys
from pathlib import Path

csv.field_size_limit(10_000_000)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.manifest_utils import ManifestWriter

try:
    import pandas as pd
    from sklearn.model_selection import train_test_split
except ImportError as e:
    print(f"ERROR: {e}. Run: pip install pandas scikit-learn")
    sys.exit(1)

# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------
RANDOM_STATE = 42
VAL_RATIO    = 0.15
TEST_RATIO   = 0.15
# train = 1 - VAL_RATIO - TEST_RATIO = 0.70

STRATA = {
    "stratum_i":   Path("data/processed/stratum_i/stratum_i_combined.csv"),
    "stratum_ii":  Path("data/processed/stratum_ii/stratum_ii_combined.csv"),
    "stratum_iii": Path("data/processed/stratum_iii/stratum_iii_combined.csv"),
}

SPLITS_DIR    = Path("data/splits")
MANIFEST_PATH = Path("outputs/manifests/splits_manifest.json")
MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)

CANONICAL_COLS = [
    "message_id", "subject", "body", "sender", "label",
    "stratum", "source", "original_file", "body_length", "body_sha256",
]


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def split_stratum(df: pd.DataFrame, name: str) -> dict:
    """Split one stratum's DataFrame into train/val/test. Returns split dict."""
    # First split: train vs (val+test)
    val_test_size = VAL_RATIO + TEST_RATIO
    train_df, val_test_df = train_test_split(
        df, test_size=val_test_size,
        stratify=df["label"], random_state=RANDOM_STATE
    )
    # Second split: val vs test (equal halves of val_test)
    relative_test = TEST_RATIO / val_test_size
    val_df, test_df = train_test_split(
        val_test_df, test_size=relative_test,
        stratify=val_test_df["label"], random_state=RANDOM_STATE
    )
    return {"train": train_df, "val": val_df, "test": test_df}


def write_splits(splits: dict, out_dir: Path, name: str) -> dict:
    """Write split CSVs, return manifest records."""
    out_dir.mkdir(parents=True, exist_ok=True)
    records = {}
    for split_name, df in splits.items():
        path = out_dir / f"{split_name}.csv"
        df[CANONICAL_COLS].to_csv(path, index=False, encoding="utf-8",
                                   lineterminator="\n")
        sha = file_sha256(path)
        n_ham   = int((df["label"] == 0).sum())
        n_phish = int((df["label"] == 1).sum())
        records[split_name] = {
            "path":       str(path),
            "rows":       len(df),
            "ham":        n_ham,
            "phishing":   n_phish,
            "sha256":     sha,
        }
        print(f"    {split_name:6s}: {len(df):>7,} rows  "
              f"(ham={n_ham:,}  phish={n_phish:,})  sha256={sha[:16]}...")
    return records


def main():
    print("=" * 65)
    print("GENERATE DETERMINISTIC STRATIFIED SPLITS")
    print(f"  random_state = {RANDOM_STATE}")
    print(f"  train/val/test = 70% / 15% / 15%")
    print("=" * 65)

    mw = ManifestWriter(
        script_name="generate_splits",
        random_seed=RANDOM_STATE,
        parameters={
            "random_state": RANDOM_STATE,
            "val_ratio":    VAL_RATIO,
            "test_ratio":   TEST_RATIO,
            "train_ratio":  round(1 - VAL_RATIO - TEST_RATIO, 2),
            "stratify_on":  "label",
        },
        notes=(
            "Deterministic stratified 70/15/15 splits for each stratum "
            "and pooled. Fixed random_state=42. Stratified on label. "
            "SHA-256 of every output file committed to splits_manifest.json."
        ),
    )

    all_dfs = {}
    manifest = {
        "random_state": RANDOM_STATE,
        "split_ratios": {"train": 0.70, "val": 0.15, "test": 0.15},
        "strata": {},
    }

    # Per-stratum splits
    for name, path in STRATA.items():
        print(f"\n{name.upper()} <- {path.name}")
        mw.add_input(str(path))
        df = pd.read_csv(path, encoding="utf-8", encoding_errors="replace",
                         keep_default_na=False)
        print(f"  Loaded: {len(df):,} rows  "
              f"(ham={int((df['label']==0).sum()):,}  "
              f"phish={int((df['label']==1).sum()):,})")
        all_dfs[name] = df
        splits = split_stratum(df, name)
        out_dir = SPLITS_DIR / name
        records = write_splits(splits, out_dir, name)
        manifest["strata"][name] = records

    # Pooled splits
    print("\nPOOLED (all three strata combined)")
    pooled_df = pd.concat(list(all_dfs.values()), ignore_index=True)
    print(f"  Loaded: {len(pooled_df):,} rows  "
          f"(ham={int((pooled_df['label']==0).sum()):,}  "
          f"phish={int((pooled_df['label']==1).sum()):,})")
    pooled_splits = split_stratum(pooled_df, "pooled")
    pooled_records = write_splits(pooled_splits, SPLITS_DIR / "pooled", "pooled")
    manifest["strata"]["pooled"] = pooled_records

    # Write manifest
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest written -> {MANIFEST_PATH}")

    # Summary
    total_train = sum(
        manifest["strata"][s]["train"]["rows"]
        for s in ["stratum_i", "stratum_ii", "stratum_iii"]
    )
    total_val  = sum(
        manifest["strata"][s]["val"]["rows"]
        for s in ["stratum_i", "stratum_ii", "stratum_iii"]
    )
    total_test = sum(
        manifest["strata"][s]["test"]["rows"]
        for s in ["stratum_i", "stratum_ii", "stratum_iii"]
    )

    print("\n" + "=" * 65)
    print("SPLIT SUMMARY (per-stratum totals, excl. pooled)")
    print(f"  Train: {total_train:,}")
    print(f"  Val:   {total_val:,}")
    print(f"  Test:  {total_test:,}")
    print(f"  Total: {total_train + total_val + total_test:,}")
    print("=" * 65)
    print("DONE. Run: pytest tests/test_splits.py -v")

    mw.add_output(str(MANIFEST_PATH))
    mw.set_counts({
        "total_rows_split": total_train + total_val + total_test,
        "train": total_train,
        "val":   total_val,
        "test":  total_test,
    })
    mw.write()


if __name__ == "__main__":
    main()
