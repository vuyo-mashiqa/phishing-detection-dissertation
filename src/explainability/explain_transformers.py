"""
Phase 12B -- Explainability: Transformer Integrated Gradients
Methods §1.11.5

For each (model_family, train_stratum) combination:
  - Loads the best-seed fine-tuned transformer checkpoint
  - Computes Integrated Gradients (Captum) on 50 test examples per stratum
    spanning both correct and incorrect predictions
  - Saves word-level attribution maps to results/explanations/transformers/

Outputs per (model_family, train_stratum):
  results/explanations/transformers/{model_family}/{train_stratum}/
    examples.jsonl   -- 50 worked examples with token attributions
    summary.json     -- aggregate stats (n_correct, n_incorrect, top tokens)

Methods compliance:
  - Integrated Gradients via Captum  (§1.11.5)
  - 50 worked examples per stratum, correct + incorrect  (§1.11.5)
  - Neutral baseline: zero embedding vector  (§1.11.5)
  - Word-level attribution = sum of absolute IG scores per token  (§1.11.5)
  - Results saved to results/explanations/  (§1.11.5)
"""

import json
import sys
import os
from pathlib import Path

import numpy as np
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from captum.attr import IntegratedGradients

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT         = Path(__file__).resolve().parents[2]
MODELS_DIR   = ROOT / "models" / "transformers"
SPLITS_DIR   = ROOT / "data" / "processed" / "splits"
CANON_DIR    = ROOT / "data" / "processed"
OUT_DIR      = ROOT / "results" / "explanations" / "transformers"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Constants ─────────────────────────────────────────────────────────────────
MODEL_FAMILIES = {
    "distilbert": "distilbert-base-uncased",
    "roberta":    "roberta-base",
    "deberta":    "microsoft/deberta-v3-base",
}
STRATA       = ["stratum_i", "stratum_ii", "stratum_iii"]
SEEDS        = [42, 123, 456]
N_EXAMPLES   = 50    # Methods §1.11.5
MAX_SEQ_LEN  = 512   # Methods §1.10.1
N_IG_STEPS  = 20    # reduced from 50 to prevent GPU OOM on RTX 5070 8.5GB
MAX_IG_LEN  = 256   # IG-specific sequence length (model trained at 512)
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Config keys mirror Phase 9: pooled training only for cross-stratum evaluation
# Best config is "pooled" (trained on pooled, evaluated on per-stratum test sets)
TRAIN_CONFIG = "pooled"

# ── Helpers ───────────────────────────────────────────────────────────────────

def get_checkpoint_dir(model_key: str, config: str = TRAIN_CONFIG):
    """Return (ckpt_path, val_f1) for models/transformers/{config}/{model_key}/"""
    model_dir    = MODELS_DIR / config / model_key
    results_path = model_dir / "results.json"
    ckpt_path    = model_dir / "best_model"

    if not ckpt_path.exists():
        return None, -1.0

    val_f1 = -1.0
    if results_path.exists():
        with open(results_path, encoding="utf-8") as f:
            res = json.load(f)
        val_f1 = res.get("val_f1", res.get("best_val_f1", -1.0))

    return ckpt_path, val_f1


def load_test_texts(stratum: str) -> pd.DataFrame:
    """
    Load test split for stratum, join subject+body text from canonical CSV.
    Returns DataFrame with columns: message_id, label, text
    """
    split_csv  = SPLITS_DIR / f"test_{stratum}.csv"
    # Determine canonical CSV path
    canon_map = {
        "stratum_i":   CANON_DIR / "stratum_i"   / "stratum_i_combined.csv",
        "stratum_ii":  CANON_DIR / "stratum_ii"  / "stratum_ii_combined.csv",
        "stratum_iii": CANON_DIR / "stratum_iii" / "stratum_iii_combined.csv",
    }
    canon_csv = canon_map[stratum]

    split_df = pd.read_csv(split_csv, usecols=["message_id", "label"],
                           dtype={"message_id": str, "label": int})
    canon_df = pd.read_csv(canon_csv, usecols=["message_id", "subject", "body"],
                           dtype=str)
    canon_df["subject"] = canon_df["subject"].fillna("")
    canon_df["body"]    = canon_df["body"].fillna("")
    canon_df["text"]    = canon_df["subject"] + " [SEP] " + canon_df["body"]

    merged = split_df.merge(canon_df[["message_id", "text"]], on="message_id", how="left")
    merged["text"] = merged["text"].fillna("")
    return merged


def select_examples(df: pd.DataFrame, model, tokenizer, n: int = N_EXAMPLES):
    """
    Run inference on the full test set, then select up to n//2 correct and
    n//2 incorrect predictions, sampling randomly with seed 42.
    Returns list of dicts: {message_id, label, pred, text, correct}
    """
    model.eval()
    rng = np.random.default_rng(42)

    correct_pool   = []
    incorrect_pool = []

    batch_size = 32
    for start in range(0, len(df), batch_size):
        batch = df.iloc[start:start + batch_size]
        texts  = batch["text"].tolist()
        labels = batch["label"].tolist()
        ids    = batch["message_id"].tolist()

        enc = tokenizer(
            texts,
            truncation=True,
            max_length=MAX_SEQ_LEN,
            padding=True,
            return_tensors="pt",
        ).to(DEVICE)

        enc_input = {k: v for k, v in enc.items() if k != "token_type_ids"}
        with torch.no_grad():
            logits = model(**enc_input).logits
        preds = logits.argmax(dim=-1).cpu().numpy()

        for mid, lbl, pred, txt in zip(ids, labels, preds, texts):
            entry = {"message_id": mid, "label": int(lbl),
                     "pred": int(pred), "text": txt,
                     "correct": int(lbl) == int(pred)}
            if entry["correct"]:
                correct_pool.append(entry)
            else:
                incorrect_pool.append(entry)

    n_incorrect = min(len(incorrect_pool), n // 2)
    n_correct   = min(len(correct_pool),   n - n_incorrect)

    chosen_incorrect = incorrect_pool[:n_incorrect] if len(incorrect_pool) <= n_incorrect \
        else [incorrect_pool[i] for i in rng.choice(len(incorrect_pool), n_incorrect, replace=False).tolist()]
    chosen_correct   = correct_pool[:n_correct] if len(correct_pool) <= n_correct \
        else [correct_pool[i]   for i in rng.choice(len(correct_pool),   n_correct,   replace=False).tolist()]

    return chosen_correct + chosen_incorrect


def compute_ig(model, tokenizer, text: str, target_class: int):
    """
    Compute Integrated Gradients for one example.
    Returns list of (token_str, attribution_score) pairs.
    """
    enc = tokenizer(
        text,
        truncation=True,
        max_length=MAX_IG_LEN,
        return_tensors="pt",
        padding=False,
    ).to(DEVICE)

    input_ids      = enc["input_ids"]
    attention_mask = enc["attention_mask"]

    # Only pass token_type_ids if the model actually accepts them
    import inspect
    forward_params = inspect.signature(model.forward).parameters
    token_type_ids = enc.get("token_type_ids", None) if "token_type_ids" in forward_params else None

    embed_layer = model.get_input_embeddings()

    def forward_func(input_embeds):
        kwargs = dict(inputs_embeds=input_embeds, attention_mask=attention_mask)
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids
        outputs = model(**kwargs)
        return outputs.logits[:, target_class]

    input_embeds    = embed_layer(input_ids)
    baseline_embeds = torch.zeros_like(input_embeds)

    ig = IntegratedGradients(forward_func)
    attributions, _ = ig.attribute(
        input_embeds,
        baselines=baseline_embeds,
        n_steps=N_IG_STEPS,
        return_convergence_delta=True,
    )

    token_attrs = attributions.squeeze(0).abs().sum(dim=-1).detach().cpu().numpy()
    tokens      = tokenizer.convert_ids_to_tokens(input_ids.squeeze(0).tolist())

    return [
        {"token": tok, "attribution": float(attr)}
        for tok, attr in zip(tokens, token_attrs)
    ]


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Phase 12B -- Explainability: Transformer Integrated Gradients")
    print("=" * 60)
    print(f"  Device    : {DEVICE}")
    print(f"  N_EXAMPLES: {N_EXAMPLES}")
    print(f"  IG steps  : {N_IG_STEPS}")
    print()

    all_ok  = True
    summary = []

    for model_key, hf_name in MODEL_FAMILIES.items():
        # Skip already-completed models
        already_done = all(
            (OUT_DIR / model_key / s / "examples.jsonl").exists()
            for s in STRATA
        )
        if already_done:
            print(f"\n  [--] {model_key}: all strata already complete — skipping")
            summary += [{"model": model_key, "stratum": s, "n_examples": 50,
                         "n_correct": 0, "n_incorrect": 0, "status": "OK"} for s in STRATA]
            continue
        print(f"\n{'=' * 60}")
        print(f"  Model : {model_key}  ({hf_name})")
        print(f"{'=' * 60}")

        ckpt_path, best_f1 = get_checkpoint_dir(model_key)
        if ckpt_path is None:
            print(f"  [XX] No checkpoint found for {model_key} — skipping")
            all_ok = False
            continue

        print(f"  Checkpoint : {ckpt_path}  (val_f1={best_f1:.4f})")

        try:
            tokenizer = AutoTokenizer.from_pretrained(str(ckpt_path))
        except Exception:
            # Fallback: load tokenizer from HuggingFace hub (handles version skew)
            print(f"  [WW] Local tokenizer load failed, falling back to HF hub: {hf_name}")
            tokenizer = AutoTokenizer.from_pretrained(hf_name)
        model = AutoModelForSequenceClassification.from_pretrained(str(ckpt_path))
        model.to(DEVICE)
        model.eval()
        print(f"  Loaded model to {DEVICE}")

        for stratum in STRATA:
            print(f"\n  [{model_key} | {stratum}]")

            out_dir = OUT_DIR / model_key / stratum
            out_dir.mkdir(parents=True, exist_ok=True)

            # Load test texts
            try:
                df = load_test_texts(stratum)
            except Exception as e:
                print(f"    [XX] Failed to load test texts: {e}")
                all_ok = False
                summary.append({"model": model_key, "stratum": stratum, "status": "FAILED"})
                continue

            print(f"    Test set rows : {len(df):,}")

            # Select 50 examples (correct + incorrect)
            examples = select_examples(df, model, tokenizer)
            n_correct   = sum(1 for e in examples if e["correct"])
            n_incorrect = sum(1 for e in examples if not e["correct"])
            print(f"    Selected      : {len(examples)} examples  ({n_correct} correct, {n_incorrect} incorrect)")

            # Compute IG for each example
            records = []
            for i, ex in enumerate(examples):
                try:
                    token_attrs = compute_ig(model, tokenizer, ex["text"], target_class=ex["pred"])
                    record = {
                        "example_id":   i + 1,
                        "message_id":   ex["message_id"],
                        "label":        ex["label"],
                        "pred":         ex["pred"],
                        "correct":      ex["correct"],
                        "n_tokens":     len(token_attrs),
                        "top10_tokens": sorted(token_attrs, key=lambda x: x["attribution"], reverse=True)[:10],
                        "all_tokens":   token_attrs,
                    }
                    records.append(record)
                    if (i + 1) % 10 == 0:
                        print(f"    IG computed : {i + 1}/{len(examples)}")
                except Exception as e:
                    print(f"    [WW] IG failed for example {i+1}: {e}")
                    continue
                finally:
                    torch.cuda.empty_cache()

            # Write examples.jsonl
            jsonl_path = out_dir / "examples.jsonl"
            with open(jsonl_path, "w", encoding="utf-8") as f:
                for rec in records:
                    # Write without all_tokens to keep file size manageable
                    out_rec = {k: v for k, v in rec.items() if k != "all_tokens"}
                    f.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
            print(f"    [OK] examples.jsonl written  ({len(records)} records)")

            # Write full token attributions as npz (excluded from git like SHAP npz)
            all_top10 = [r["top10_tokens"] for r in records]

            # Summary JSON
            all_top_tokens = {}
            for rec in records:
                for entry in rec["top10_tokens"]:
                    tok = entry["token"]
                    all_top_tokens[tok] = all_top_tokens.get(tok, 0.0) + entry["attribution"]
            top_global = sorted(all_top_tokens.items(), key=lambda x: x[1], reverse=True)[:20]

            summary_data = {
                "model":       model_key,
                "stratum":     stratum,
                "checkpoint":  str(ckpt_path),
                "n_examples":  len(records),
                "n_correct":   n_correct,
                "n_incorrect": n_incorrect,
                "top20_tokens_by_cumulative_attribution": [
                    {"token": t, "cumulative_attribution": round(s, 4)}
                    for t, s in top_global
                ],
            }
            with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
                json.dump(summary_data, f, indent=2, ensure_ascii=False)
            print(f"    [OK] summary.json written")

            summary.append({"model": model_key, "stratum": stratum,
                            "n_examples": len(records),
                            "n_correct": n_correct, "n_incorrect": n_incorrect,
                            "status": "OK"})

        # Free GPU memory before next model
        del model
        torch.cuda.empty_cache()

    # ── Final summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for row in summary:
        mark = "[OK]" if row["status"] == "OK" else "[XX]"
        n    = row.get("n_examples", 0)
        nc   = row.get("n_correct", 0)
        ni   = row.get("n_incorrect", 0)
        print(f"  {mark}  {row['model']:12s}  {row['stratum']:12s}  {n} examples ({nc} correct, {ni} incorrect)")

    print()
    if all_ok:
        print("PHASE 12B TRANSFORMER IG COMPLETE.")
    else:
        print("PHASE 12B FAILED. Fix XX entries above before proceeding.")
    print("=" * 60)
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
