"""
src/data/combine_stratum_ii.py

Combines CSDMC2010 ham and phishing_pot phishing into the single
canonical Stratum II combined file.

Inputs:
  data/processed/stratum_ii/csdmc2010_ham.csv
  data/processed/stratum_ii/phishing_pot_phishing.csv

Output:
  data/processed/stratum_ii/stratum_ii_combined.csv

Design decision:
  Natural class imbalance is preserved (2,763 ham : 5,085 phishing).
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.manifest_utils import ManifestWriter

# ------------------------------------------------------------------ #
#  PATHS                                                              #
# ------------------------------------------------------------------ #
HAM_PATH      = Path("data/processed/stratum_ii/csdmc2010_ham.csv")
PHISH_PATH    = Path("data/processed/stratum_ii/phishing_pot_phishing.csv")
OUTPUT_PATH   = Path("data/processed/stratum_ii/stratum_ii_combined.csv")

# ------------------------------------------------------------------ #
#  CANONICAL SCHEMA COLUMNS                                           #
# ------------------------------------------------------------------ #
CANONICAL_COLS = [
    "message_id", "subject", "body", "sender",
    "label", "stratum", "source", "original_file",
    "body_length", "body_sha256",
]

READ_KWARGS = dict(keep_default_na=False, na_values=[],
                   dtype={"label": int, "body_length": int},
                   encoding="utf-8", encoding_errors="replace")


def main():
    print("=" * 65)
    print("  STRATUM II COMBINE SCRIPT")
    print("=" * 65)

    mw = ManifestWriter(
        script_name="combine_stratum_ii",
        random_seed=None,
        parameters={
            "ham_source":   str(HAM_PATH),
            "phish_source": str(PHISH_PATH),
            "imbalance_policy": "natural -- handled at modelling time",
        },
        notes=(
            "Combines CSDMC2010 ham (label=0) and phishing_pot phishing "
            "(label=1) into stratum_ii_combined.csv. Natural class imbalance "
            "preserved. No resampling applied at this stage."
        )
    )

    # ---- Load both component files ------------------------------------
    print(f"  Loading {HAM_PATH.name}...")
    ham = pd.read_csv(HAM_PATH, **READ_KWARGS)
    print(f"    Shape: {ham.shape} | Labels: {ham['label'].value_counts().to_dict()}")

    print(f"  Loading {PHISH_PATH.name}...")
    phish = pd.read_csv(PHISH_PATH, **READ_KWARGS)
    print(f"    Shape: {phish.shape} | Labels: {phish['label'].value_counts().to_dict()}")

    mw.add_input(str(HAM_PATH))
    mw.add_input(str(PHISH_PATH))

    # ---- Concatenate --------------------------------------------------
    df = pd.concat([ham, phish], ignore_index=True)
    print(f"\n  Combined shape (pre-validation): {df.shape}")

    # ---- Enforce canonical column order -------------------------------
    df = df[CANONICAL_COLS]

    # ---- Null policy enforcement --------------------------------------
    df["subject"] = df["subject"].where(df["subject"].notna(), other="")
    df["sender"]  = df["sender"].where(df["sender"].notna(), other="")

    # ---- Drop any phantom rows ----------------------------------------
    df = df[df["message_id"].notna() & (df["message_id"] != "")]
    df = df[df["body"].notna() & (df["body"].str.strip() != "")]

    # ---- Dtype enforcement --------------------------------------------
    df["label"]       = df["label"].astype(int)
    df["body_length"] = df["body_length"].astype(int)

    # ---- Cross-corpus duplicate check ---------------------------------
    # If the same body_sha256 appears in both ham and phishing (extremely
    # unlikely but must be checked), it indicates contamination.
    cross_dupes = df[df.duplicated(subset=["body_sha256"], keep=False)]
    if len(cross_dupes) > 0:
        print(f"  WARNING: {len(cross_dupes)} rows share body_sha256 across corpora.")
        print(f"  These will be removed -- likely identical boilerplate.")
        df = df.drop_duplicates(subset=["body_sha256"], keep="first")

    # ---- Final null assertion -----------------------------------------
    null_counts = df.isnull().sum()
    if null_counts.any():
        raise ValueError(
            f"Null values in combined file:\n{null_counts[null_counts > 0]}"
        )

    # ---- Write --------------------------------------------------------
    # Recompute body_length and body_sha256 from the final body text.
    # The UTF-8 errors=replace sanitisation may have altered body content,
    # making stored body_length values stale. Recompute both derived fields
    # from the actual final body string to guarantee consistency.
    import hashlib
    df["body_length"] = df["body"].str.len().astype(int)
    df["body_sha256"] = df["body"].apply(
        lambda x: hashlib.sha256(x.lower().encode("utf-8", errors="replace")).hexdigest()
    )

    # Write with errors=replace to guarantee clean UTF-8 output.
    with open(OUTPUT_PATH, "w", encoding="utf-8", errors="replace", newline="\n") as fh:
        df.to_csv(fh, index=False, lineterminator="\n")

    n_ham   = (df["label"] == 0).sum()
    n_phish = (df["label"] == 1).sum()
    ratio   = n_phish / n_ham if n_ham > 0 else float("inf")

    print(f"\n  Output written -> {OUTPUT_PATH}")
    print(f"  Total rows:    {len(df):,}")
    print(f"  Ham   (0):     {n_ham:,}")
    print(f"  Phish (1):     {n_phish:,}")
    print(f"  Phish:Ham ratio: {ratio:.2f}:1")

    mw.add_output(str(OUTPUT_PATH))
    mw.set_counts({
        "ham_rows":         int(n_ham),
        "phishing_rows":    int(n_phish),
        "total_rows":       len(df),
        "phish_ham_ratio":  round(ratio, 3),
        "cross_dupes_removed": len(cross_dupes) // 2 if len(cross_dupes) > 0 else 0,
    })
    mw.write()

    print("=" * 65)
    print("  DONE. Run: pytest tests/test_schema.py -v")
    print("=" * 65)


if __name__ == "__main__":
    main()
    