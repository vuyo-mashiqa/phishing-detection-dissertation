"""
build_features.py
=================
Builds the feature matrix for classical ML experiments.

Feature groups:
  1. TF-IDF body (word 1-2gram + char 3-5gram, concatenated)
  2. TF-IDF subject (word 1-2gram)
  3. URL features  (5 scalar features from body text)
  4. Sender features (2 scalar features)

All groups are combined via ColumnTransformer into a single sparse matrix.

Usage — fit and transform (training):
    from src.features.build_features import FeaturePipeline
    fp = FeaturePipeline()
    X_train = fp.fit_transform(df_train)
    fp.save("outputs/features/stratum_i_pipeline.joblib")

Usage — transform only (val/test/cross-stratum):
    fp = FeaturePipeline.load("outputs/features/stratum_i_pipeline.joblib")
    X_test = fp.transform(df_test)
"""

import re
import sys
from pathlib import Path

import joblib
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler


# ------------------------------------------------------------------
# URL FEATURES
# ------------------------------------------------------------------
SUSPICIOUS_TLDS = {
    ".xyz", ".top", ".click", ".loan", ".work", ".date", ".win",
    ".bid", ".trade", ".stream", ".download", ".accountant",
}

URL_PATTERN = re.compile(
    r"https?://[^\s<>\"']+|www\.[^\s<>\"']+", re.IGNORECASE
)


def extract_url_features(body: str) -> list[float]:
    """Return 5 URL-based scalar features from body text."""
    urls = URL_PATTERN.findall(body)
    n = len(urls)
    if n == 0:
        return [0.0, 0.0, 0.0, 0.0, 0.0]
    n_suspicious_tld = sum(
        1 for u in urls
        if any(u.lower().endswith(t) or f"{t}/" in u.lower() for t in SUSPICIOUS_TLDS)
    )
    n_ip_url = sum(1 for u in urls if re.search(r"https?://\d{1,3}(\.\d{1,3}){3}", u))
    n_at_symbol = sum(1 for u in urls if "@" in u)
    max_len = max(len(u) for u in urls)
    return [
        float(n),                           # total URL count
        float(n_suspicious_tld) / n,        # fraction with suspicious TLD
        float(n_ip_url),                    # IP-literal URL count
        float(n_at_symbol),                 # @ in URL (credential phishing signal)
        float(min(max_len, 500)),           # max URL length (capped)
    ]


# ------------------------------------------------------------------
# SENDER FEATURES
# ------------------------------------------------------------------
def extract_sender_features(sender: str, subject: str) -> list[float]:
    """Return 2 sender/header scalar features."""
    sender = sender or ""
    subject = subject or ""
    # Mismatch proxy: display name contains a different domain than address
    domain_in_name = bool(re.search(r"@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}", sender.split("<")[0]))
    # Urgency markers in subject
    urgency_terms = ["urgent", "action required", "verify", "suspended",
                     "unusual activity", "security alert", "immediately",
                     "limited time", "click here", "confirm your"]
    urgency = any(t in subject.lower() for t in urgency_terms)
    return [float(domain_in_name), float(urgency)]


# ------------------------------------------------------------------
# FEATURE PIPELINE
# ------------------------------------------------------------------
class FeaturePipeline:
    """
    Fits and transforms a canonical email DataFrame into a sparse feature matrix.

    Feature dimensions:
      - TF-IDF body word:   50,000 features
      - TF-IDF body char:   30,000 features
      - TF-IDF subject:     10,000 features
      - URL scalars:             5 features
      - Sender scalars:          2 features
    """

    BODY_WORD_MAX   = 50000
    BODY_CHAR_MAX   = 30000
    SUBJECT_MAX     = 10000

    def __init__(self):
        self.body_word_tfidf = TfidfVectorizer(
            analyzer="word", ngram_range=(1, 2),
            max_features=self.BODY_WORD_MAX,
            sublinear_tf=True, strip_accents="unicode",
            min_df=2, max_df=0.95,
        )
        self.body_char_tfidf = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 5),
            max_features=self.BODY_CHAR_MAX,
            sublinear_tf=True, strip_accents="unicode",
            min_df=2, max_df=0.95,
        )
        self.subject_tfidf = TfidfVectorizer(
            analyzer="word", ngram_range=(1, 2),
            max_features=self.SUBJECT_MAX,
            sublinear_tf=True, strip_accents="unicode",
            min_df=2,
        )
        self.scalar_scaler = MaxAbsScaler()
        self._fitted = False

    def _extract_scalars(self, df) -> np.ndarray:
        url_feats     = [extract_url_features(b) for b in df["body"].fillna("")]
        sender_feats  = [
            extract_sender_features(s, subj)
            for s, subj in zip(df["sender"].fillna(""), df["subject"].fillna(""))
        ]
        return np.array([u + sv for u, sv in zip(url_feats, sender_feats)],
                        dtype=np.float32)

    def fit_transform(self, df) -> sp.csr_matrix:
        bodies   = df["body"].fillna("").tolist()
        subjects = df["subject"].fillna("").tolist()
        scalars  = self._extract_scalars(df)

        X_bw = self.body_word_tfidf.fit_transform(bodies)
        X_bc = self.body_char_tfidf.fit_transform(bodies)
        X_s  = self.subject_tfidf.fit_transform(subjects)
        X_sc = sp.csr_matrix(self.scalar_scaler.fit_transform(scalars))

        self._fitted = True
        mat = sp.hstack([X_bw, X_bc, X_s, X_sc], format="csr")
        print(f"  Feature matrix: {mat.shape[0]:,} rows x {mat.shape[1]:,} cols")
        return mat

    def transform(self, df) -> sp.csr_matrix:
        if not self._fitted:
            raise RuntimeError("Pipeline not fitted. Call fit_transform first.")
        bodies   = df["body"].fillna("").tolist()
        subjects = df["subject"].fillna("").tolist()
        scalars  = self._extract_scalars(df)

        X_bw = self.body_word_tfidf.transform(bodies)
        X_bc = self.body_char_tfidf.transform(bodies)
        X_s  = self.subject_tfidf.transform(subjects)
        X_sc = sp.csr_matrix(self.scalar_scaler.transform(scalars))

        return sp.hstack([X_bw, X_bc, X_s, X_sc], format="csr")

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        print(f"  Pipeline saved -> {path}")

    @staticmethod
    def load(path: str) -> "FeaturePipeline":
        fp = joblib.load(path)
        print(f"  Pipeline loaded <- {path}")
        return fp

    @property
    def n_features(self) -> int:
        return (self.BODY_WORD_MAX + self.BODY_CHAR_MAX +
                self.SUBJECT_MAX + 7)  # 7 = 5 URL + 2 sender scalars


if __name__ == "__main__":
    """Quick smoke test — reads Stratum I train split."""
    import csv
    import pandas as pd

    csv.field_size_limit(10000000)
    TRAIN = Path("data/splits/stratum_i/train.csv")
    if not TRAIN.exists():
        print("ERROR: Run generate_splits.py first.")
        sys.exit(1)

    print("Loading Stratum I train split...")
    df = pd.read_csv(TRAIN, encoding="utf-8", encoding_errors="replace",
                     keep_default_na=False)
    print(f"  Rows: {len(df):,}  "
          f"Ham: {int((df['label']==0).sum()):,}  "
          f"Phish: {int((df['label']==1).sum()):,}")

    print("Fitting feature pipeline...")
    fp = FeaturePipeline()
    X = fp.fit_transform(df)

    print(f"\nSmoke test PASSED")
    print(f"  Matrix shape: {X.shape}")
    print(f"  Matrix dtype: {X.dtype}")
    print(f"  NNZ:          {X.nnz:,}")
    fp.save("outputs/features/smoke_test_pipeline.joblib")
    print("Done.")
