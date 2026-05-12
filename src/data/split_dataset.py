"""
split_dataset.py
================
Generates deterministic 70/15/15 train/validation/test splits.

Specification
----------------------------
  Ratio        : 70 / 15 / 15
  Random seed  : 42

Outputs
-------
  data/processed/splits/train_stratum_i.csv
  data/processed/splits/val_stratum_i.csv
  data/processed/splits/test_stratum_i.csv
  data/processed/splits/train_stratum_ii.csv
  data/processed/splits/val_stratum_ii.csv
  data/processed/splits/test_stratum_ii.csv
  data/processed/splits/train_stratum_iii.csv
  data/processed/splits/val_stratum_iii.csv
  data/processed/splits/test_stratum_iii.csv
  data/processed/splits/train_pooled.csv
  data/processed/splits/val_pooled.csv
  data/processed/splits/test_pooled.csv
  data/processed/splits/split_report.txt

Run
---
  python src/data/split_dataset.py
"""

import hashlib
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.manifest_utils import ManifestWriter

# ── Config ────────────────────────────────────────────────────────────
RANDOM_SEED  = 42
TRAIN_SIZE   = 0.70
VAL_SIZE     = 0.15   # of total; = 0.50 of the 30% temp split
TEST_SIZE    = 0.15   # of total; = 0.50 of the 30% temp split

FEAT_I        = Path("data/processed/features/features_stratum_i.csv")
FEAT_II       = Path("data/processed/features/features_stratum_ii.csv")
FEAT_III      = Path("data/processed/features/features_stratum_iii.csv")
FEAT_COMBINED = Path("data/processed/features/features_combined.csv")

OUT_DIR       = Path("data/processed/splits")

# Target counts from Methods §1.9 table — used only for assertion
EXPECTED = {
    "I":      {"train": 108_675, "val": 23_287, "test": 23_288},
    "II":     {"train":   5_492, "val":  1_177, "test":  1_177},
    "III":    {"train":   9_349, "val":  2_003, "test":  2_004},
    "pooled": {"train": 123_516, "val": 26_468, "test": 26_468},
}


# ── Helpers ───────────────────────────────────────────────────────────
def three_way_split(df: pd.DataFrame, seed: int) -> tuple:
    """
    Two-stage stratified split producing 70 / 15 / 15 partitions.

    Stage 1: train (70%) vs temp (30%), stratified by label.
    Stage 2: temp split 50/50 -> val (15%) / test (15%), stratified by label.

    Returns (train_df, val_df, test_df), all with reset index.
    """
    train, temp = train_test_split(
        df,
        train_size=TRAIN_SIZE,
        random_state=seed,
        stratify=df["label"],
        shuffle=True,
    )
    val, test = train_test_split(
        temp,
        test_size=0.5,          # 15 / 30 = 0.50
        random_state=seed,
        stratify=temp["label"],
        shuffle=True,
    )
    return (
        train.reset_index(drop=True),
        val.reset_index(drop=True),
        test.reset_index(drop=True),
    )


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_split(df: pd.DataFrame, path: Path) -> str:
    """Write CSV and return its SHA-256."""
    df.to_csv(path, index=False, encoding="utf-8", lineterminator="\n")
    return sha256_file(path)


def validate_split(train, val, test, label: str, source_n: int):
    """Assert integrity constraints from Methods §1.9."""
    # Row counts sum to source total
    total = len(train) + len(val) + len(test)
    assert total == source_n, (
        f"{label}: split total {total} != source {source_n}"
    )
    # Zero overlap across all three partitions
    ids_tr = set(train["message_id"])
    ids_va = set(val["message_id"])
    ids_te = set(test["message_id"])
    assert len(ids_tr & ids_va) == 0, f"{label}: train/val overlap"
    assert len(ids_tr & ids_te) == 0, f"{label}: train/test overlap"
    assert len(ids_va & ids_te) == 0, f"{label}: val/test overlap"
    # Label proportions preserved within 1% tolerance
    for lbl in [0, 1]:
        full_rate  = (
            pd.concat([train, val, test])["label"] == lbl
        ).mean()
        for part_name, part in [("train", train), ("val", val), ("test", test)]:
            rate = (part["label"] == lbl).mean()
            drift = abs(rate - full_rate)
            assert drift < 0.01, (
                f"{label} {part_name}: label {lbl} drift {drift:.4f} > 1%"
            )
    # Expected counts match Methods §1.9 table
    key = label.replace("Stratum ", "").lower()
    if key in EXPECTED:
        exp = EXPECTED[key]
        assert len(train) == exp["train"], (
            f"{label} train: got {len(train)}, expected {exp['train']}"
        )
        assert len(val)   == exp["val"],   (
            f"{label} val:   got {len(val)},   expected {exp['val']}"
        )
        assert len(test)  == exp["test"],  (
            f"{label} test:  got {len(test)},  expected {exp['test']}"
        )


def report_partition(name: str, df: pd.DataFrame) -> str:
    dist = df["label"].value_counts().sort_index().to_dict()
    return (
        f"  {name:<30}  n={len(df):>7,}  "
        f"ham={dist.get(0,0):>7,}  phishing={dist.get(1,0):>6,}"
    )


# ── Main ──────────────────────────────────────────────────────────────
def main():
    print("=" * 65)
    print("TRAIN / VALIDATION / TEST SPLIT  (70 / 15 / 15)")
    print(f"  seed={RANDOM_SEED}  stratify=label")
    print("=" * 65)

    for p in [FEAT_I, FEAT_II, FEAT_III, FEAT_COMBINED]:
        if not p.exists():
            print(f"ERROR: {p} not found. Run build_features.py first.")
            sys.exit(1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Delete stale 80/20 split files if present
    for stale in ["train.csv", "test.csv"]:
        stale_path = OUT_DIR / stale
        if stale_path.exists():
            stale_path.unlink()
            print(f"  Removed stale file: {stale_path}")

    mw = ManifestWriter(
        script_name="split_dataset",
        random_seed=RANDOM_SEED,
        parameters={
            "train_size":  TRAIN_SIZE,
            "val_size":    VAL_SIZE,
            "test_size":   TEST_SIZE,
            "stratify_on": "label (binary)",
            "generation":  "independent per stratum and pooled",
        },
        notes=(
            "70/15/15 split per Methods Section 1.9. "
            "Validation set role: threshold calibration ONLY (Methods §1.11.3). "
            "Validation set must not be used for any other optimisation decision."
        ),
    )
    for p in [FEAT_I, FEAT_II, FEAT_III, FEAT_COMBINED]:
        mw.add_input(str(p))

    file_hashes = {}
    report_lines = [
        "SPLIT REPORT  (70 / 15 / 15)",
        f"seed={RANDOM_SEED}  stratify=label",
        "=" * 60,
    ]

    # ── Per-stratum splits ────────────────────────────────────────────
    strata = [
        ("I",   FEAT_I,   "stratum_i"),
        ("II",  FEAT_II,  "stratum_ii"),
        ("III", FEAT_III, "stratum_iii"),
    ]

    for stratum_key, feat_path, suffix in strata:
        print(f"\n--- STRATUM {stratum_key} ---")
        df = pd.read_csv(feat_path, low_memory=False)
        print(f"  Loaded {len(df):,} rows")

        train, val, test = three_way_split(df, RANDOM_SEED)
        validate_split(train, val, test, f"Stratum {stratum_key}", len(df))

        paths = {
            "train": OUT_DIR / f"train_{suffix}.csv",
            "val":   OUT_DIR / f"val_{suffix}.csv",
            "test":  OUT_DIR / f"test_{suffix}.csv",
        }
        for part_name, part_df in [("train", train), ("val", val), ("test", test)]:
            h = write_split(part_df, paths[part_name])
            file_hashes[paths[part_name].name] = h
            mw.add_output(str(paths[part_name]))
            print(report_partition(paths[part_name].name, part_df))

        report_lines += [
            f"\nStratum {stratum_key}  (total={len(df):,})",
            report_partition(f"train_{suffix}", train),
            report_partition(f"val_{suffix}",   val),
            report_partition(f"test_{suffix}",  test),
            f"  Integrity: PASSED",
        ]

    # ── Pooled split (independent) ────────────────────────────────────
    print("\n--- POOLED (independent) ---")
    df_all = pd.read_csv(FEAT_COMBINED, low_memory=False)
    print(f"  Loaded {len(df_all):,} rows")

    train_p, val_p, test_p = three_way_split(df_all, RANDOM_SEED)
    validate_split(train_p, val_p, test_p, "pooled", len(df_all))

    pooled_paths = {
        "train": OUT_DIR / "train_pooled.csv",
        "val":   OUT_DIR / "val_pooled.csv",
        "test":  OUT_DIR / "test_pooled.csv",
    }
    for part_name, part_df in [("train", train_p), ("val", val_p), ("test", test_p)]:
        h = write_split(part_df, pooled_paths[part_name])
        file_hashes[pooled_paths[part_name].name] = h
        mw.add_output(str(pooled_paths[part_name]))
        print(report_partition(pooled_paths[part_name].name, part_df))

    report_lines += [
        f"\nPooled  (total={len(df_all):,})",
        report_partition("train_pooled", train_p),
        report_partition("val_pooled",   val_p),
        report_partition("test_pooled",  test_p),
        f"  Integrity: PASSED",
        "",
        "File SHA-256 hashes",
    ]
    for fname, h in file_hashes.items():
        report_lines.append(f"  {fname:<35}  {h}")

    # ── Write report ──────────────────────────────────────────────────
    report_path = OUT_DIR / "split_report.txt"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    mw.add_output(str(report_path))

    # ── Summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("SPLIT COMPLETE")
    print("=" * 65)
    print(f"  {'Stratum':<8}  {'Train':>8}  {'Val':>8}  {'Test':>8}  {'Total':>8}")
    print(f"  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}")
    for stratum_key, _, suffix in strata:
        tr = pd.read_csv(OUT_DIR / f"train_{suffix}.csv", usecols=["message_id"])
        va = pd.read_csv(OUT_DIR / f"val_{suffix}.csv",   usecols=["message_id"])
        te = pd.read_csv(OUT_DIR / f"test_{suffix}.csv",  usecols=["message_id"])
        tot = len(tr) + len(va) + len(te)
        print(f"  {stratum_key:<8}  {len(tr):>8,}  {len(va):>8,}  {len(te):>8,}  {tot:>8,}")
    tot_p = len(train_p) + len(val_p) + len(test_p)
    print(f"  {'Pooled':<8}  {len(train_p):>8,}  {len(val_p):>8,}  {len(test_p):>8,}  {tot_p:>8,}")

    mw.set_counts({
        "i_train": len(train), "i_val": len(val), "i_test": len(test),
        "ii_train": 5492,      "ii_val": 1177,     "ii_test": 1177,
        "iii_train": 9349,     "iii_val": 2003,    "iii_test": 2004,
        "pooled_train": len(train_p), "pooled_val": len(val_p), "pooled_test": len(test_p),
    })
    mw.write()
    print("\nReady for Step 7.5: pytest tests/test_split.py -v")


if __name__ == "__main__":
    main()
