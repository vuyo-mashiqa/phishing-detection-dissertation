"""
build_representation.py

Builds the hybrid TF-IDF + structural feature representation for all 4
dataset configurations (stratum_i, stratum_ii, stratum_iii, pooled).

Vectorisers are fit ONLY on the training split; val/test are transform-only.
All matrices saved as .npz; fitted vectorisers saved as pipeline.joblib.
"""

import hashlib
import json
import re
import subprocess
import time
from pathlib import Path
from urllib.parse import urlparse

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
SPLITS_DIR  = ROOT / "data" / "processed" / "splits"
CANON_DIR   = ROOT / "data" / "processed"
OUTPUT_DIR  = ROOT / "outputs" / "features"

CONFIGS    = ["stratum_i", "stratum_ii", "stratum_iii", "pooled"]
PARTITIONS = ["train", "val", "test"]

# Strata that compose each config
CONFIG_STRATA = {
    "stratum_i":   ["stratum_i"],
    "stratum_ii":  ["stratum_ii"],
    "stratum_iii": ["stratum_iii"],
    "pooled":      ["stratum_i", "stratum_ii", "stratum_iii"],
}

# TF-IDF hyperparameters (Methods §1.10.1)
TFIDF_WORD_PARAMS = dict(
    analyzer="word",
    ngram_range=(1, 2),
    sublinear_tf=True,
    min_df=2,
    max_features=100_000,
    dtype=np.float32,
)
TFIDF_CHAR_PARAMS = dict(
    analyzer="char_wb",
    ngram_range=(3, 5),
    sublinear_tf=True,
    min_df=3,
    max_features=50_000,
    dtype=np.float32,
)

# Suspicious TLDs (Methods §1.10.1 — operationally relevant per Saka et al. 2024)
SUSPICIOUS_TLDS = {
    ".tk", ".ml", ".ga", ".cf", ".gq", ".xyz", ".top", ".click",
    ".download", ".loan", ".win", ".racing", ".online", ".stream",
    ".gdn", ".bid", ".faith", ".date", ".review", ".trade",
}

# ── Structural feature extraction ─────────────────────────────────────────────

def _extract_urls(text: str) -> list:
    """Extract all URLs from body text."""
    return re.findall(r'https?://[^\s<>"\']+', text)


def _url_count(text: str) -> int:
    return len(_extract_urls(text))


def _suspicious_tld(text: str) -> int:
    """1 if any URL contains a suspicious TLD."""
    for url in _extract_urls(text):
        try:
            hostname = (urlparse(url).hostname or "").lower()
        except ValueError:
            continue
        for tld in SUSPICIOUS_TLDS:
            if hostname.endswith(tld):
                return 1
    return 0


def _ip_in_url(text: str) -> int:
    """1 if any URL uses a bare IP address instead of a hostname."""
    ip_pattern = re.compile(r'https?://(\d{1,3}\.){3}\d{1,3}[/:]?')
    return 1 if ip_pattern.search(text) else 0


def _subdomain_depth(text: str) -> int:
    """Max number of subdomains (dots in hostname minus 1) across all URLs."""
    max_depth = 0
    for url in _extract_urls(text):
        try:
            hostname = urlparse(url).hostname or ""
        except ValueError:
            continue
        parts = hostname.split(".")
        depth = max(0, len(parts) - 2)
        if depth > max_depth:
            max_depth = depth
    return max_depth


def _html_char_ratio(body: str) -> float:
    """Fraction of body characters that are HTML tag characters."""
    if not body:
        return 0.0
    tag_chars = sum(len(m.group()) for m in re.finditer(r'<[^>]+>', body))
    return tag_chars / len(body)


def _sender_mismatch(sender: str, body: str) -> int:
    """
    1 if the display-name portion of the sender contains a domain that
    differs from the actual sending domain in the address.
    e.g. 'PayPal Support <attacker@evil.com>' → mismatch=1
    """
    if not sender:
        return 0
    # Extract actual sending domain from angle-bracket address
    addr_match = re.search(r'<([^>]+)>', sender)
    if addr_match:
        addr = addr_match.group(1)
        display = sender[:sender.index('<')].strip().strip('"\'')
    else:
        addr = sender.strip()
        display = ""
    addr_domain_m = re.search(r'@([\w.\-]+)', addr)
    addr_domain = addr_domain_m.group(1).lower() if addr_domain_m else ""
    # Look for a domain-like pattern in the display name
    display_domain_m = re.search(r'([\w\-]+\.(?:com|org|net|io|gov|edu|co))', display, re.I)
    if display_domain_m:
        display_domain = display_domain_m.group(1).lower()
        if addr_domain and display_domain not in addr_domain and addr_domain not in display_domain:
            return 1
    return 0


def _reply_to_mismatch(sender: str, body: str) -> int:
    """
    1 if a Reply-To pattern appears in the body that has a different domain
    than the sender address — a strong phishing signal.
    """
    if not sender:
        return 0
    addr_match = re.search(r'@([\w.\-]+)', sender)
    sender_domain = addr_match.group(1).lower() if addr_match else ""
    reply_to_match = re.search(r'reply[- ]to[:\s]+[^\s<]*@([\w.\-]+)', body, re.I)
    if reply_to_match:
        reply_domain = reply_to_match.group(1).lower()
        if sender_domain and reply_domain != sender_domain:
            return 1
    return 0


def extract_structural_features(df: pd.DataFrame) -> np.ndarray:
    """
    Compute all 7 Methods-prescribed structural features for each row.
    Returns a float32 array of shape (n, 7).
    Column order matches STRUCTURAL_COL_NAMES below.
    """
    bodies  = df["body"].fillna("").tolist()
    senders = df["sender"].fillna("").tolist()

    n = len(df)
    out = np.zeros((n, 7), dtype=np.float32)

    for i, (body, sender) in enumerate(zip(bodies, senders)):
        out[i, 0] = _url_count(body)
        out[i, 1] = _suspicious_tld(body)
        out[i, 2] = _ip_in_url(body)
        out[i, 3] = _subdomain_depth(body)
        out[i, 4] = _html_char_ratio(body)
        out[i, 5] = _sender_mismatch(sender, body)
        out[i, 6] = _reply_to_mismatch(sender, body)

        if (i + 1) % 10_000 == 0:
            print(f"      Structural features: {i+1:,}/{n:,}")

    return out


STRUCTURAL_COL_NAMES = [
    "url_count",
    "suspicious_tld",
    "ip_in_url",
    "subdomain_depth",
    "html_char_ratio",
    "sender_mismatch",
    "reply_to_mismatch",
]

# ── Data loading ──────────────────────────────────────────────────────────────

def _load_canonical_text(config: str) -> pd.DataFrame:
    """
    Load message_id, subject, body, sender from canonical combined CSVs.
    For pooled, concatenate all three strata.
    """
    strata = CONFIG_STRATA[config]
    parts = []
    for stratum in strata:
        path = CANON_DIR / stratum / f"{stratum}_combined.csv"
        if not path.exists():
            raise FileNotFoundError(f"Canonical CSV not found: {path}")
        df = pd.read_csv(
            path,
            usecols=["message_id", "subject", "body", "sender", "label"],
            dtype={"label": int},
        )
        parts.append(df)
    return pd.concat(parts, ignore_index=True)


def _load_split_ids(config: str, partition: str) -> pd.Series:
    """Load the message_id list for a given config/partition from split files."""
    path = SPLITS_DIR / f"{partition}_{config}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Split file not found: {path}")
    df = pd.read_csv(path, usecols=["message_id"])
    return df["message_id"]


def _get_partition_df(canonical: pd.DataFrame, split_ids: pd.Series) -> pd.DataFrame:
    """
    Filter canonical dataframe to rows matching split_ids,
    preserving the split's original order.
    """
    id_set = set(split_ids)
    df = canonical[canonical["message_id"].isin(id_set)].copy()
    # Reorder to match split order exactly
    order = {mid: i for i, mid in enumerate(split_ids)}
    df["_order"] = df["message_id"].map(order)
    df = df.sort_values("_order").drop(columns=["_order"]).reset_index(drop=True)
    return df


# ── Helpers ───────────────────────────────────────────────────────────────────

def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ── Main build ────────────────────────────────────────────────────────────────

def build_config(config: str) -> None:
    print(f"\n{'='*60}")
    print(f"  Building representation: {config.upper()}")
    print(f"{'='*60}")

    out_dir = OUTPUT_DIR / config
    out_dir.mkdir(parents=True, exist_ok=True)

    # Skip if already complete
    if (out_dir / "manifest.json").exists() and \
       (out_dir / "train.npz").exists() and \
       (out_dir / "val.npz").exists() and \
       (out_dir / "test.npz").exists():
        print(f"  [SKIP] {config} already complete - loading from cache")
        return

    # Load full canonical text for this config
    print("  Loading canonical text...")
    canonical = _load_canonical_text(config)
    print(f"    Canonical rows: {len(canonical):,}")

    # Build per-partition dataframes
    print("  Aligning splits to canonical text...")
    partition_dfs = {}
    for partition in PARTITIONS:
        split_ids = _load_split_ids(config, partition)
        df = _get_partition_df(canonical, split_ids)
        assert len(df) == len(split_ids), (
            f"{config}/{partition}: aligned {len(df)} rows but split has {len(split_ids)} ids"
        )
        partition_dfs[partition] = df
        print(f"    {partition}: {len(df):,} rows")

    train_df = partition_dfs["train"]
    val_df   = partition_dfs["val"]
    test_df  = partition_dfs["test"]

    # Build text input: subject + " " + body (Methods §1.10.1)
    def make_text(df):
        return (df["subject"].fillna("") + " " + df["body"].fillna("")).str.strip()

    # Extract structural features (all 7 Methods §1.10.1 features)
    print("  Extracting structural features (train)...")
    t0 = time.time()
    struct_train = extract_structural_features(train_df)
    print(f"    Done ({time.time()-t0:.1f}s)")

    print("  Extracting structural features (val)...")
    t0 = time.time()
    struct_val = extract_structural_features(val_df)
    print(f"    Done ({time.time()-t0:.1f}s)")

    print("  Extracting structural features (test)...")
    t0 = time.time()
    struct_test = extract_structural_features(test_df)
    print(f"    Done ({time.time()-t0:.1f}s)")

    # Fit TF-IDF vectorisers on TRAINING SET ONLY
    print("  Fitting TF-IDF word vectoriser on training set...")
    t0 = time.time()
    vec_word = TfidfVectorizer(**TFIDF_WORD_PARAMS)
    X_train_word = vec_word.fit_transform(make_text(train_df))
    print(f"    Word TF-IDF: {X_train_word.shape[1]:,} features  ({time.time()-t0:.1f}s)")

    print("  Fitting TF-IDF char vectoriser on training set...")
    t0 = time.time()
    vec_char = TfidfVectorizer(**TFIDF_CHAR_PARAMS)
    X_train_char = vec_char.fit_transform(make_text(train_df))
    print(f"    Char TF-IDF: {X_train_char.shape[1]:,} features  ({time.time()-t0:.1f}s)")

    # Transform val and test (NO re-fitting)
    print("  Transforming val and test...")
    X_val_word  = vec_word.transform(make_text(val_df))
    X_val_char  = vec_char.transform(make_text(val_df))
    X_test_word = vec_word.transform(make_text(test_df))
    X_test_char = vec_char.transform(make_text(test_df))

    # Concatenate: [word_tfidf | char_tfidf | structural]
    print("  Concatenating components...")
    struct_train_sp = sp.csr_matrix(struct_train)
    struct_val_sp   = sp.csr_matrix(struct_val)
    struct_test_sp  = sp.csr_matrix(struct_test)

    X_train = sp.hstack([X_train_word, X_train_char, struct_train_sp], format="csr")
    X_val   = sp.hstack([X_val_word,   X_val_char,   struct_val_sp],   format="csr")
    X_test  = sp.hstack([X_test_word,  X_test_char,  struct_test_sp],  format="csr")

    n_feat = X_train.shape[1]
    print(f"    Final dim: {n_feat:,}  "
          f"(word={X_train_word.shape[1]:,} + "
          f"char={X_train_char.shape[1]:,} + structural=7)")

    # Extract labels
    y_train = train_df["label"].values.astype(np.int32)
    y_val   = val_df["label"].values.astype(np.int32)
    y_test  = test_df["label"].values.astype(np.int32)

    # Save matrices and labels
    print("  Saving matrices...")
    sp.save_npz(out_dir / "train.npz", X_train)
    sp.save_npz(out_dir / "val.npz",   X_val)
    sp.save_npz(out_dir / "test.npz",  X_test)
    np.save(out_dir / "y_train.npy", y_train)
    np.save(out_dir / "y_val.npy",   y_val)
    np.save(out_dir / "y_test.npy",  y_test)

    # Save message_id order for downstream alignment checks
    train_df[["message_id"]].to_csv(out_dir / "train_ids.csv", index=False)
    val_df[["message_id"]].to_csv(out_dir / "val_ids.csv",     index=False)
    test_df[["message_id"]].to_csv(out_dir / "test_ids.csv",   index=False)

    # Save fitted vectorisers
    print("  Saving fitted vectorisers...")
    joblib.dump({"vec_word": vec_word, "vec_char": vec_char},
                out_dir / "pipeline.joblib")

    # Write manifest
    manifest = {
        "config": config,
        "git_sha": _git_sha(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_train": int(len(train_df)),
        "n_val":   int(len(val_df)),
        "n_test":  int(len(test_df)),
        "n_features":            int(n_feat),
        "n_word_features":       int(X_train_word.shape[1]),
        "n_char_features":       int(X_train_char.shape[1]),
        "n_structural_features": 7,
        "structural_col_names":  STRUCTURAL_COL_NAMES,
        "tfidf_word_params":     {k: str(v) for k, v in TFIDF_WORD_PARAMS.items()},
        "tfidf_char_params":     {k: str(v) for k, v in TFIDF_CHAR_PARAMS.items()},
        "train_sha256": _file_sha256(out_dir / "train.npz"),
        "val_sha256":   _file_sha256(out_dir / "val.npz"),
        "test_sha256":  _file_sha256(out_dir / "test.npz"),
    }
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"  [OK] {config} complete - dim={n_feat:,}")


def main():
    print("\nBUILD HYBRID REPRESENTATION")
    print("Methods §1.10.1: word TF-IDF + char TF-IDF + 7 structural features")
    print(f"Configs: {CONFIGS}\n")

    t_total = time.time()
    for config in CONFIGS:
        build_config(config)

    print(f"\n{'='*60}")
    print(f"  ALL CONFIGS COMPLETE  ({time.time()-t_total:.0f}s total)")
    print(f"{'='*60}")
    print("\nOutputs written to: outputs/features/")
    print("Next step: pytest tests/test_representation.py -v")


if __name__ == "__main__":
    main()
