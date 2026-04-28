"""
check_cross_stratum_leakage.py
==============================
Runs a two-pass cross-stratum leakage audit across all three canonical CSVs.

Pass 1 — Exact match:
    Any two rows sharing the same body_sha256 across different strata are
    exact duplicates. These are unambiguous leakage candidates.

Pass 2 — Near-duplicate (MinHash-LSH):
    MinHash with 128 permutations, Jaccard threshold 0.85 on normalised
    (lowercased, whitespace-collapsed) body text. Catches paraphrased or
    lightly reformatted copies of the same email.

Output:
    outputs/manifests/cross_stratum_leakage_report.json

Interpretation:
    - Zero exact matches + zero near-duplicates = CLEAN. Data freeze approved.
    - Any matches = review the offending pairs before proceeding.

Parameters (per project design document):
    Permutations : 128
    Jaccard threshold : 0.85
    Shingle size : 5 characters

Run:
    python src/data/check_cross_stratum_leakage.py
"""

import csv
import hashlib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.manifest_utils import ManifestWriter

csv.field_size_limit(10_000_000)

try:
    from datasketch import MinHash, MinHashLSH
except ImportError:
    print("ERROR: datasketch not installed. Run: pip install datasketch")
    sys.exit(1)

# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------
STRATA = {
    "I":   Path("data/processed/stratum_i/stratum_i_combined.csv"),
    "II":  Path("data/processed/stratum_ii/stratum_ii_combined.csv"),
    "III": Path("data/processed/stratum_iii/stratum_iii_combined.csv"),
}
NUM_PERM       = 128
JACCARD_THRESH = 0.85
SHINGLE_SIZE   = 5
OUTPUT_PATH    = Path("outputs/manifests/cross_stratum_leakage_report.json")
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)


def normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().strip())


def shingle(text: str, k: int = SHINGLE_SIZE) -> set:
    return {text[i:i+k] for i in range(max(1, len(text) - k + 1))}


def make_minhash(text: str) -> MinHash:
    m = MinHash(num_perm=NUM_PERM)
    for s in shingle(normalise(text)):
        m.update(s.encode("utf-8"))
    return m


def load_stratum(path: Path, label: str) -> list[dict]:
    print(f"  Loading Stratum {label} from {path.name}...")
    rows = []
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            rows.append({
                "stratum":     label,
                "message_id":  row["message_id"],
                "body_sha256": row["body_sha256"],
                "body":        row["body"],
                "label":       row["label"],
                "source":      row["source"],
            })
    print(f"    Rows: {len(rows):,}")
    return rows


def main():
    print("=" * 65)
    print("CROSS-STRATUM LEAKAGE AUDIT")
    print(f"  MinHash permutations : {NUM_PERM}")
    print(f"  Jaccard threshold    : {JACCARD_THRESH}")
    print(f"  Shingle size         : {SHINGLE_SIZE} chars")
    print("=" * 65)

    mw = ManifestWriter(
        script_name="check_cross_stratum_leakage",
        random_seed=None,
        parameters={
            "num_perm": NUM_PERM,
            "jaccard_threshold": JACCARD_THRESH,
            "shingle_size": SHINGLE_SIZE,
        },
        notes=(
            "Cross-stratum leakage audit. Pass 1: exact body_sha256 matches. "
            "Pass 2: MinHash-LSH near-duplicates (Jaccard >= 0.85). "
            "Zero matches required before data freeze."
        ),
    )

    for path in STRATA.values():
        mw.add_input(str(path))

    # ---- Load all three strata ----------------------------------
    print("\nLoading strata...")
    all_rows = {}
    for label, path in STRATA.items():
        all_rows[label] = load_stratum(path, label)

    # ================================================================
    # PASS 1 — Exact duplicate check (body_sha256)
    # ================================================================
    print("\nPass 1 — Exact duplicate check (body_sha256)...")
    hash_index: dict[str, list[str]] = {}  # sha256 -> list of strata
    for label, rows in all_rows.items():
        for row in rows:
            h = row["body_sha256"]
            hash_index.setdefault(h, []).append(label)

    exact_pairs = []
    for sha, strata_list in hash_index.items():
        unique_strata = list(set(strata_list))
        if len(unique_strata) > 1:
            exact_pairs.append({
                "body_sha256": sha,
                "found_in_strata": unique_strata,
                "occurrence_count": len(strata_list),
            })

    print(f"  Exact cross-stratum duplicates found: {len(exact_pairs)}")

    # ================================================================
    # PASS 2 — Near-duplicate check (MinHash-LSH)
    # ================================================================
    print("\nPass 2 — Near-duplicate check (MinHash-LSH)...")
    print("  Building MinHash index (this will take several minutes)...")

    lsh = MinHashLSH(threshold=JACCARD_THRESH, num_perm=NUM_PERM)
    minhashes: dict[str, tuple[str, MinHash]] = {}  # key -> (stratum, minhash)

    total = sum(len(r) for r in all_rows.values())
    processed = 0

    for label, rows in all_rows.items():
        for row in rows:
            key = f"{label}::{row['message_id']}"
            mh = make_minhash(row["body"])
            minhashes[key] = (label, mh)
            try:
                lsh.insert(key, mh)
            except ValueError:
                pass  # duplicate key — already inserted
            processed += 1
            if processed % 10000 == 0:
                pct = processed / total * 100
                print(f"    Indexed {processed:,}/{total:,} ({pct:.1f}%)", end="\r")

    print(f"    Indexed {total:,}/{total:,} (100.0%)          ")
    print("  Querying for near-duplicates across strata...")

    near_dup_pairs = []
    seen_pairs: set[frozenset] = set()

    for key, (stratum_a, mh_a) in minhashes.items():
        results = lsh.query(mh_a)
        for other_key in results:
            if other_key == key:
                continue
            stratum_b = minhashes[other_key][0]
            if stratum_a == stratum_b:
                continue  # same stratum — not cross-stratum leakage
            pair_id = frozenset([key, other_key])
            if pair_id in seen_pairs:
                continue
            seen_pairs.add(pair_id)
            near_dup_pairs.append({
                "key_a": key,
                "key_b": other_key,
                "stratum_a": stratum_a,
                "stratum_b": stratum_b,
            })

    print(f"  Near-duplicate cross-stratum pairs found: {len(near_dup_pairs)}")

    # ================================================================
    # RESULT
    # ================================================================
    total_issues = len(exact_pairs) + len(near_dup_pairs)
    verdict = "CLEAN — data freeze approved" if total_issues == 0 \
              else f"ISSUES FOUND — {total_issues} leakage candidates require review"

    print("\n" + "=" * 65)
    print(f"VERDICT: {verdict}")
    print("=" * 65)

    report = {
        "verdict": verdict,
        "parameters": {
            "num_perm": NUM_PERM,
            "jaccard_threshold": JACCARD_THRESH,
            "shingle_size": SHINGLE_SIZE,
        },
        "stratum_sizes": {k: len(v) for k, v in all_rows.items()},
        "pass_1_exact_duplicates": {
            "count": len(exact_pairs),
            "pairs": exact_pairs[:50],  # cap at 50 for readability
        },
        "pass_2_near_duplicates": {
            "count": len(near_dup_pairs),
            "pairs": near_dup_pairs[:50],
        },
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Report written -> {OUTPUT_PATH}")

    mw.add_output(str(OUTPUT_PATH))
    mw.set_counts({
        "total_rows_audited":        total,
        "exact_cross_stratum_dupes": len(exact_pairs),
        "near_dup_cross_stratum":    len(near_dup_pairs),
        "verdict":                   verdict,
    })
    mw.write()

    if total_issues > 0:
        print("\nACTION REQUIRED: Review the pairs in the report before proceeding.")
        sys.exit(1)
    else:
        print("\nAll clear. Proceed to Step 5.3 (PII scrubbing).")


if __name__ == "__main__":
    main()
    