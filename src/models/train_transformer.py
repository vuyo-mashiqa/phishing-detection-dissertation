"""
train_transformer.py
"""

import copy
import csv
import hashlib
import json
import random
import subprocess
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    roc_auc_score, average_precision_score,
    balanced_accuracy_score, matthews_corrcoef,
    roc_curve,
)
import pandas as pd

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parents[2]
CANON_DIR   = ROOT / "data" / "processed"
SPLITS_DIR  = ROOT / "data" / "processed" / "splits"
MODELS_DIR  = ROOT / "models" / "transformers"
RESULTS_DIR = ROOT / "results" / "transformers"

CONFIGS = ["stratum_i", "stratum_ii", "stratum_iii", "pooled"]
STRATA  = ["stratum_i", "stratum_ii", "stratum_iii"]

CONFIG_STRATA = {
    "stratum_i":   ["stratum_i"],
    "stratum_ii":  ["stratum_ii"],
    "stratum_iii": ["stratum_iii"],
    "pooled":      ["stratum_i", "stratum_ii", "stratum_iii"],
}

MODELS = {
    "distilbert": "distilbert-base-uncased",
    "roberta":    "roberta-base",
    "deberta":    "microsoft/deberta-v3-base",
}

# ── Hyperparameters — UNIFORM ACROSS ALL THREE TRANSFORMER MODELS ─────────────
MAX_LENGTH   = 512
BATCH_SIZE   = 16
GRAD_ACCUM   = 2           # effective batch = 32
EPOCHS       = 5
PATIENCE     = 2           # early stopping on val macro-F1
LR           = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.10
FOCAL_ALPHA  = 0.25        # Methods §1.11.2
FOCAL_GAMMA  = 2.0         # Methods §1.11.2
N_THRESH     = 100         # threshold search points (Methods §1.11.3)
LATENCY_N    = 200         # emails sampled for latency benchmark
SEEDS        = [42, 123, 456]  # Methods §1.11.6

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Reproducibility ───────────────────────────────────────────────────────────
def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── Focal Loss ─────────────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    """
    Focal loss via BCEWithLogitsLoss.
    alpha=0.25, gamma=2.0. IDENTICAL for all three transformer models.
    Numerically stable: no explicit sigmoid before this call.
    """
    def __init__(self, alpha: float = FOCAL_ALPHA, gamma: float = FOCAL_GAMMA):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        logits  = logits.view(-1)
        labels  = labels.float().view(-1)
        bce     = nn.functional.binary_cross_entropy_with_logits(
            logits, labels, reduction="none"
        )
        probs   = torch.sigmoid(logits)
        p_t     = probs * labels + (1.0 - probs) * (1.0 - labels)
        alpha_t = self.alpha * labels + (1.0 - self.alpha) * (1.0 - labels)
        return (alpha_t * (1.0 - p_t) ** self.gamma * bce).mean()


# ── Dataset ────────────────────────────────────────────────────────────────────
class EmailDataset(Dataset):
    """Tokenised subject+body pairs with integer labels. Right-truncation."""
    def __init__(self, texts: list, labels: list, tokenizer,
                 max_length: int = MAX_LENGTH):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


# ── Data loading ───────────────────────────────────────────────────────────────
def _load_canonical(config: str) -> pd.DataFrame:
    parts = []
    for stratum in CONFIG_STRATA[config]:
        path = CANON_DIR / stratum / f"{stratum}_combined.csv"
        if not path.exists():
            raise FileNotFoundError(f"Canonical CSV not found: {path}")
        parts.append(pd.read_csv(
            path, usecols=["message_id", "subject", "body", "label"]
        ))
    return pd.concat(parts, ignore_index=True)


def _align_partition(canonical: pd.DataFrame, config: str, partition: str):
    """Return (texts, labels) aligned to frozen split file order."""
    split_path = SPLITS_DIR / f"{partition}_{config}.csv"
    if not split_path.exists():
        raise FileNotFoundError(f"Split file not found: {split_path}")
    split_ids = pd.read_csv(split_path, usecols=["message_id"])["message_id"]
    df = canonical[canonical["message_id"].isin(set(split_ids))].copy()
    order = {mid: i for i, mid in enumerate(split_ids)}
    df["_order"] = df["message_id"].map(order)
    df = df.sort_values("_order").drop(columns=["_order"]).reset_index(drop=True)
    assert len(df) == len(split_ids), (
        f"{config}/{partition}: aligned {len(df)}, split has {len(split_ids)}"
    )
    texts  = (df["subject"].fillna("") + " " + df["body"].fillna("")).str.strip().tolist()
    labels = df["label"].astype(int).tolist()
    return texts, labels


def _load_test_stratum(stratum: str):
    return _align_partition(_load_canonical(stratum), stratum, "test")


# ── Helpers ────────────────────────────────────────────────────────────────────
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


# ── Evaluation helpers ─────────────────────────────────────────────────────────
def _find_f1_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> tuple:
    """t* = argmax F1(t) over N_THRESH points on validation set."""
    best_t, best_f1 = 0.5, 0.0
    for t in np.linspace(0.0, 1.0, N_THRESH):
        f1 = f1_score(y_true, (y_prob >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t, best_f1


def _fpr_at_95tpr(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return float(fpr[min(np.searchsorted(tpr, 0.95), len(fpr) - 1)])


def _youden_j(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return float(np.max(tpr - fpr))


def _compute_metrics(y_true: np.ndarray, y_prob: np.ndarray,
                     threshold: float) -> dict:
    """Full 8-metric suite (Methods §1.11.1) at given threshold."""
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "threshold":         float(threshold),
        "f1_macro":          float(f1_score(y_true, y_pred, average="macro",   zero_division=0)),
        "precision":         float(precision_score(y_true, y_pred,              zero_division=0)),
        "recall":            float(recall_score(y_true, y_pred,                 zero_division=0)),
        "roc_auc":           float(roc_auc_score(y_true, y_prob)),
        "pr_auc":            float(average_precision_score(y_true, y_prob)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "mcc":               float(matthews_corrcoef(y_true, y_pred)),
        "fpr_at_95tpr":      _fpr_at_95tpr(y_true, y_prob),
        "youden_j":          _youden_j(y_true, y_prob),
    }


# ── Inference ──────────────────────────────────────────────────────────────────
@torch.no_grad()
def _predict_proba(model: nn.Module, dataloader: DataLoader) -> np.ndarray:
    """Sigmoid probabilities for positive class. Consistent across all 3 models."""
    model.eval()
    probs = []
    for batch in dataloader:
        input_ids      = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if "token_type_ids" in batch:
            kwargs["token_type_ids"] = batch["token_type_ids"].to(DEVICE)
        outputs = model(**kwargs)
        # AutoModelForSequenceClassification always returns (B, 2) for num_labels=2
        logits  = outputs.logits[:, 1] if outputs.logits.shape[1] == 2 \
                  else outputs.logits.view(-1)
        probs.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(probs)


def _latency_benchmark(model: nn.Module, tokenizer,
                       texts: list, n: int = LATENCY_N) -> dict:
    """Per-email latency (ms) at p50/p95/p99."""
    idx = np.random.default_rng(42).choice(
        len(texts), size=min(n, len(texts)), replace=False
    )
    model.eval()
    times = []
    with torch.no_grad():
        for i in idx:
            enc = tokenizer(
                texts[i], truncation=True, padding="max_length",
                max_length=MAX_LENGTH, return_tensors="pt",
            )
            enc = {k: v.to(DEVICE) for k, v in enc.items()}
            t0  = time.perf_counter()
            model(**enc)
            times.append((time.perf_counter() - t0) * 1000.0)
    times = np.array(times)
    return {
        "latency_p50_ms": float(np.percentile(times, 50)),
        "latency_p95_ms": float(np.percentile(times, 95)),
        "latency_p99_ms": float(np.percentile(times, 99)),
    }


# ── Single-seed training run ───────────────────────────────────────────────────
def _train_one_seed(config: str, model_key: str, seed: int,
                    train_texts, train_labels,
                    val_texts,   val_labels,
                    test_texts,  test_labels,
                    tokenizer) -> dict:
    """
    Complete training + evaluation for one seed. Returns result dict.
    Weights of best epoch held in memory; not written to disk here.
    """
    _set_seed(seed)
    if model_key == "deberta":
        torch.set_default_dtype(torch.float32)
        if DEVICE.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = False
    model_name = MODELS[model_key]

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2
    ).to(DEVICE)

    train_ds     = EmailDataset(train_texts, train_labels, tokenizer)
    val_ds       = EmailDataset(val_texts,   val_labels,   tokenizer)
    test_ds      = EmailDataset(test_texts,  test_labels,  tokenizer)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=(DEVICE.type == "cuda"))
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE * 2, shuffle=False,
                              num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE * 2, shuffle=False,
                              num_workers=0)

    criterion = FocalLoss(alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )
    total_steps  = (len(train_loader) // GRAD_ACCUM) * EPOCHS
    warmup_steps = int(WARMUP_RATIO * total_steps)
    scheduler    = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    best_val_f1    = 0.0
    best_epoch     = 0
    best_t_star    = 0.5
    best_state     = None
    patience_count = 0
    train_log      = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader, 1):
            input_ids      = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels         = batch["labels"].to(DEVICE)
            kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
            if "token_type_ids" in batch:
                kwargs["token_type_ids"] = batch["token_type_ids"].to(DEVICE)

            outputs = model(**kwargs)
            logits  = outputs.logits[:, 1] if outputs.logits.shape[1] == 2 \
                      else outputs.logits.view(-1)
            loss = criterion(logits, labels) / GRAD_ACCUM
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"    [seed={seed}] WARNING: nan/inf loss at step {step}, skipping batch")
                optimizer.zero_grad()
                continue
            loss.backward()
            epoch_loss += loss.item() * GRAD_ACCUM

            if step % GRAD_ACCUM == 0 or step == len(train_loader):
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        val_probs = _predict_proba(model, val_loader)
        t_star_ep, val_f1 = _find_f1_threshold(np.array(val_labels), val_probs)
        avg_loss = epoch_loss / len(train_loader)

        print(f"    [seed={seed}] Epoch {epoch}/{EPOCHS}  "
              f"loss={avg_loss:.4f}  val_F1={val_f1:.4f}  t*={t_star_ep:.4f}")

        train_log.append({
            "epoch": epoch, "train_loss": avg_loss,
            "val_f1": val_f1, "t_star": t_star_ep,
        })

        if val_f1 > best_val_f1:
            best_val_f1    = val_f1
            best_epoch     = epoch
            best_t_star    = t_star_ep
            patience_count = 0
            best_state     = copy.deepcopy(model.state_dict())
        else:
            patience_count += 1
            if patience_count >= PATIENCE:
                print(f"    [seed={seed}] Early stopping at epoch {epoch} "
                      f"(best val_F1={best_val_f1:.4f} @ epoch {best_epoch})")
                break

    # Load best weights for evaluation
    if best_state is None:
        raise RuntimeError(
            f"No valid epoch completed for seed={seed} config={config} model={model_key}. "
            f"All epochs produced nan loss. Check learning rate and input data."
        )
    model.load_state_dict(best_state)

    # Test evaluation
    test_probs  = _predict_proba(model, test_loader)
    test_metrics = _compute_metrics(np.array(test_labels), test_probs, best_t_star)

    # Cross-stratum evaluation
    cross_stratum = {}
    for eval_stratum in STRATA:
        cs_texts, cs_labels = _load_test_stratum(eval_stratum)
        cs_ds     = EmailDataset(cs_texts, cs_labels, tokenizer)
        cs_loader = DataLoader(cs_ds, batch_size=BATCH_SIZE * 2, shuffle=False,
                               num_workers=0)
        cs_probs  = _predict_proba(model, cs_loader)
        cross_stratum[eval_stratum] = _compute_metrics(
            np.array(cs_labels), cs_probs, best_t_star
        )

    # Latency benchmark (Methods §1.11.4)
    latency = _latency_benchmark(model, tokenizer, test_texts)

    return {
        "seed":          seed,
        "best_epoch":    best_epoch,
        "best_val_f1":   best_val_f1,
        "best_t_star":   best_t_star,
        "test_metrics":  test_metrics,
        "cross_stratum": cross_stratum,
        "latency":       latency,
        "train_log":     train_log,
        "model_state":   best_state,   # returned for saving best-seed weights
    }


# ── Multi-seed orchestrator ────────────────────────────────────────────────────
def train_and_evaluate(config: str, model_key: str):
    out_dir = MODELS_DIR / config / model_key
    out_dir.mkdir(parents=True, exist_ok=True)

    if (out_dir / "manifest.json").exists():
        print(f"  [SKIP] {config}/{model_key} already complete")
        return

    model_name = MODELS[model_key]
    print(f"\n  Training: {config.upper()} / {model_key}  ({model_name})")
    print(f"  Device: {DEVICE}  Seeds: {SEEDS}")

    # Load data once — shared across all seeds
    print("  Loading data...")
    canonical = _load_canonical(config)
    train_texts, train_labels = _align_partition(canonical, config, "train")
    val_texts,   val_labels   = _align_partition(canonical, config, "val")
    test_texts,  test_labels  = _align_partition(canonical, config, "test")

    n_phish = sum(train_labels)
    n_ham   = len(train_labels) - n_phish
    print(f"  train={len(train_labels):,}  phish={n_phish:,}  ham={n_ham:,}")

    # Tokeniser loaded once — identical for all seeds (Methods §1.10.2)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    seed_results     = []
    best_val_f1_all  = -1.0
    best_seed_res    = None

    for seed in SEEDS:
        print(f"\n  --- Seed {seed} ---")
        t0  = time.time()
        res = _train_one_seed(
            config, model_key, seed,
            train_texts, train_labels,
            val_texts,   val_labels,
            test_texts,  test_labels,
            tokenizer,
        )
        elapsed = time.time() - t0
        print(f"  Seed {seed} complete ({elapsed:.0f}s) | "
              f"test F1={res['test_metrics']['f1_macro']:.4f}  "
              f"ROC-AUC={res['test_metrics']['roc_auc']:.4f}  "
              f"PR-AUC={res['test_metrics']['pr_auc']:.4f}  "
              f"FPR@95TPR={res['test_metrics']['fpr_at_95tpr']:.4f}")
        for es in STRATA:
            print(f"  cross[{es}]: "
                  f"F1={res['cross_stratum'][es]['f1_macro']:.4f}  "
                  f"ROC-AUC={res['cross_stratum'][es]['roc_auc']:.4f}")
        print(f"  Latency p50={res['latency']['latency_p50_ms']:.1f}ms  "
              f"p95={res['latency']['latency_p95_ms']:.1f}ms  "
              f"p99={res['latency']['latency_p99_ms']:.1f}ms")

        # Store without model weights (don't serialise state_dict to JSON)
        seed_results.append({k: v for k, v in res.items()
                              if k != "model_state"})

        if res["best_val_f1"] > best_val_f1_all:
            best_val_f1_all = res["best_val_f1"]
            best_seed_res   = res

    # ── Persist best-seed model weights ──────────────────────────────────────
    # Temporarily reload model to save via save_pretrained
    best_model_dir = out_dir / "best_model"
    best_model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2
    )
    best_model.load_state_dict(best_seed_res["model_state"])
    best_model.save_pretrained(str(best_model_dir))
    tokenizer.save_pretrained(str(best_model_dir))
    del best_model   # free memory
    print(f"\n  Best seed={best_seed_res['seed']} "
          f"(val_F1={best_val_f1_all:.4f}) saved to {best_model_dir}")

    # ── Seed aggregate (mean±std) ───────────────────────────
    metric_keys = [
        "f1_macro", "precision", "recall", "roc_auc", "pr_auc",
        "balanced_accuracy", "mcc", "fpr_at_95tpr", "youden_j",
    ]
    seed_aggregate = {
        m: {
            "mean": float(np.mean([r["test_metrics"][m] for r in seed_results])),
            "std":  float(np.std( [r["test_metrics"][m] for r in seed_results])),
        }
        for m in metric_keys
    }

    # ── Write artefacts ───────────────────────────────────────────────────────
    with open(out_dir / "train_log.json", "w") as f:
        json.dump(
            [entry for r in seed_results for entry in r["train_log"]],
            f, indent=2
        )

    with open(out_dir / "seed_results.json", "w") as f:
        json.dump(seed_results, f, indent=2)

    with open(out_dir / "threshold.json", "w") as f:
        json.dump({
            "threshold":   best_seed_res["best_t_star"],
            "config":      config,
            "model":       model_key,
            "best_seed":   best_seed_res["seed"],
            "best_val_f1": best_val_f1_all,
        }, f, indent=2)

    # results.json — canonical result = best seed's test metrics
    results = {
        "config":         config,
        "model":          model_key,
        "best_seed":      best_seed_res["seed"],
        "best_epoch":     best_seed_res["best_epoch"],
        "best_val_f1":    best_val_f1_all,
        "test_metrics":   best_seed_res["test_metrics"],
        "cross_stratum":  best_seed_res["cross_stratum"],
        "latency":        best_seed_res["latency"],
        "seed_aggregate": seed_aggregate,
    }
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    # manifest.json
    manifest = {
        "config":        config,
        "model_key":     model_key,
        "model_name":    model_name,
        "git_sha":       _git_sha(),
        "timestamp":     time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "device":        str(DEVICE),
        "seeds":         SEEDS,
        "best_seed":     best_seed_res["seed"],
        "best_epoch":    best_seed_res["best_epoch"],
        "best_val_f1":   best_val_f1_all,
        "t_star":        best_seed_res["best_t_star"],
        "n_train":       len(train_labels),
        "n_val":         len(val_labels),
        "n_test":        len(test_labels),
        "hyperparameters": {
            "max_length":       MAX_LENGTH,
            "batch_size":       BATCH_SIZE,
            "grad_accum":       GRAD_ACCUM,
            "effective_batch":  BATCH_SIZE * GRAD_ACCUM,
            "epochs":           EPOCHS,
            "patience":         PATIENCE,
            "lr":               LR,
            "weight_decay":     WEIGHT_DECAY,
            "warmup_ratio":     WARMUP_RATIO,
            "focal_alpha":      FOCAL_ALPHA,
            "focal_gamma":      FOCAL_GAMMA,
            "loss":             "FocalLoss(BCEWithLogitsLoss-based)",
        },
        "split_sha256": {
            p: _file_sha256(SPLITS_DIR / f"{p}_{config}.csv")
            for p in ["train", "val", "test"]
        },
    }
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"  [OK] {config}/{model_key} complete  "
          f"(best seed={best_seed_res['seed']}, "
          f"epoch={best_seed_res['best_epoch']}, "
          f"val_F1={best_val_f1_all:.4f})")


# ── Collect all results → CSV ──────────────────────────────────────────────────
def collect_results():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for config in CONFIGS:
        for model_key in MODELS:
            rpath = MODELS_DIR / config / model_key / "results.json"
            if not rpath.exists():
                print(f"  [WARN] Missing: {config}/{model_key}")
                continue
            with open(rpath) as f:
                r = json.load(f)
            base = {"train_config": config, "model": model_key}
            row  = {**base, "eval_stratum": config, "eval_type": "matched"}
            row.update(r["test_metrics"])
            row.update(r["latency"])
            rows.append(row)
            for eval_stratum, metrics in r["cross_stratum"].items():
                eval_type = "matched" if eval_stratum in config else "cross"
                row = {**base, "eval_stratum": eval_stratum,
                       "eval_type": eval_type}
                row.update(metrics)
                row.update(r["latency"])
                rows.append(row)
    if not rows:
        print("  [WARN] No results found")
        return
    out_path = RESULTS_DIR / "all_results.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Results: {out_path}  ({len(rows)} rows)")


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print("\n" + "=" * 60)
    print("  TRANSFORMER FINE-TUNING")
    print("  Models: DistilBERT, RoBERTa, DeBERTa-v3-base")
    print(f"  Loss: FocalLoss(BCEWithLogitsLoss) alpha={FOCAL_ALPHA} "
          f"gamma={FOCAL_GAMMA} -- UNIFORM ACROSS ALL 3")
    print(f"  Seeds: {SEEDS}  (Variance estimation)")
    print(f"  Device: {DEVICE}")
    print("  4 configs x 3 models x 3 seeds = 36 training runs")
    print("=" * 60)

    t_total = time.time()
    for config in CONFIGS:
        print(f"\n[CONFIG: {config.upper()}]")
        for model_key in MODELS:
            train_and_evaluate(config, model_key)

    elapsed = time.time() - t_total
    print(f"\n{'='*60}")
    print(f"  ALL COMPLETE  ({elapsed/3600:.2f}h total)")
    print(f"{'='*60}")
    print("\nCollecting results...")
    collect_results()
    print("\nNext step: pytest tests/test_transformer.py -v")


if __name__ == "__main__":
    main()
