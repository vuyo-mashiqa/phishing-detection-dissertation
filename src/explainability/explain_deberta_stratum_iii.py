import json, sys, inspect
from pathlib import Path
import numpy as np
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from captum.attr import IntegratedGradients

ROOT       = Path(__file__).resolve().parents[2]
CKPT_PATH  = ROOT / "models" / "transformers" / "pooled" / "deberta" / "best_model"
SPLITS_DIR = ROOT / "data" / "processed" / "splits"
CANON_DIR  = ROOT / "data" / "processed"
OUT_DIR    = ROOT / "results" / "explanations" / "transformers" / "deberta" / "stratum_iii"
OUT_DIR.mkdir(parents=True, exist_ok=True)

STRATUM    = "stratum_iii"
N_EXAMPLES = 50
MAX_IG_LEN = 128    # shorter still — DeBERTa is large
N_IG_STEPS = 10     # fewer steps — sufficient for token ranking
DEVICE     = torch.device("cpu")   # CPU only — avoids GPU OOM

def load_test_texts():
    split_csv = SPLITS_DIR / f"test_{STRATUM}.csv"
    canon_csv = CANON_DIR / "stratum_iii" / "stratum_iii_combined.csv"
    split_df  = pd.read_csv(split_csv, usecols=["message_id", "label"],
                            dtype={"message_id": str, "label": int})
    canon_df  = pd.read_csv(canon_csv, usecols=["message_id", "subject", "body"],
                            dtype=str)
    canon_df["subject"] = canon_df["subject"].fillna("")
    canon_df["body"]    = canon_df["body"].fillna("")
    canon_df["text"]    = canon_df["subject"] + " [SEP] " + canon_df["body"]
    return split_df.merge(canon_df[["message_id", "text"]], on="message_id", how="left")

def select_examples(df, model, tokenizer):
    model.eval()
    rng = np.random.default_rng(42)
    correct_pool, incorrect_pool = [], []
    batch_size = 16
    for start in range(0, len(df), batch_size):
        batch  = df.iloc[start:start + batch_size]
        enc    = tokenizer(batch["text"].tolist(), truncation=True,
                           max_length=MAX_IG_LEN, padding=True,
                           return_tensors="pt")
        enc    = {k: v for k, v in enc.items() if k != "token_type_ids"}
        with torch.no_grad():
            preds = model(**enc).logits.argmax(dim=-1).numpy()
        for mid, lbl, pred, txt in zip(batch["message_id"], batch["label"], preds, batch["text"]):
            e = {"message_id": mid, "label": int(lbl), "pred": int(pred),
                 "text": txt, "correct": int(lbl) == int(pred)}
            (correct_pool if e["correct"] else incorrect_pool).append(e)
    n_inc = min(len(incorrect_pool), N_EXAMPLES // 2)
    n_cor = min(len(correct_pool),   N_EXAMPLES - n_inc)
    chosen_inc = [incorrect_pool[i] for i in rng.choice(len(incorrect_pool), n_inc, replace=False).tolist()] \
                 if len(incorrect_pool) > n_inc else incorrect_pool[:n_inc]
    chosen_cor = [correct_pool[i]   for i in rng.choice(len(correct_pool),   n_cor, replace=False).tolist()] \
                 if len(correct_pool) > n_cor else correct_pool[:n_cor]
    return chosen_cor + chosen_inc

def compute_ig(model, tokenizer, text, target_class):
    enc = tokenizer(text, truncation=True, max_length=MAX_IG_LEN,
                    return_tensors="pt", padding=False)
    input_ids      = enc["input_ids"]
    attention_mask = enc["attention_mask"]
    fwd_params     = inspect.signature(model.forward).parameters
    token_type_ids = enc.get("token_type_ids", None) if "token_type_ids" in fwd_params else None

    embed_layer = model.get_input_embeddings()

    def forward_func(input_embeds):
        kwargs = dict(inputs_embeds=input_embeds, attention_mask=attention_mask)
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids
        return model(**kwargs).logits[:, target_class]

    input_embeds    = embed_layer(input_ids)
    baseline_embeds = torch.zeros_like(input_embeds)
    ig = IntegratedGradients(forward_func)
    attributions, _ = ig.attribute(input_embeds, baselines=baseline_embeds,
                                   n_steps=N_IG_STEPS, return_convergence_delta=True)
    token_attrs = attributions.squeeze(0).abs().sum(dim=-1).detach().numpy()
    tokens      = tokenizer.convert_ids_to_tokens(input_ids.squeeze(0).tolist())
    return [{"token": t, "attribution": float(a)} for t, a in zip(tokens, token_attrs)]

def main():
    print("=" * 60)
    print("Phase 12B -- DeBERTa stratum_iii IG (CPU mode)")
    print("=" * 60)
    print(f"  Device     : {DEVICE}")
    print(f"  N_EXAMPLES : {N_EXAMPLES}")
    print(f"  IG steps   : {N_IG_STEPS}")
    print(f"  MAX_IG_LEN : {MAX_IG_LEN}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(str(CKPT_PATH))
    except Exception:
        print("  [WW] Local tokenizer failed, falling back to HF hub")
        tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")

    model = AutoModelForSequenceClassification.from_pretrained(str(CKPT_PATH))
    model.to(DEVICE)
    model.eval()
    print("  Model loaded to CPU")

    df       = load_test_texts()
    examples = select_examples(df, model, tokenizer)
    n_cor    = sum(1 for e in examples if e["correct"])
    n_inc    = sum(1 for e in examples if not e["correct"])
    print(f"  Selected : {len(examples)} examples ({n_cor} correct, {n_inc} incorrect)")

    records = []
    for i, ex in enumerate(examples):
        try:
            token_attrs = compute_ig(model, tokenizer, ex["text"], ex["pred"])
            records.append({
                "example_id":   i + 1,
                "message_id":   ex["message_id"],
                "label":        ex["label"],
                "pred":         ex["pred"],
                "correct":      ex["correct"],
                "n_tokens":     len(token_attrs),
                "top10_tokens": sorted(token_attrs, key=lambda x: x["attribution"], reverse=True)[:10],
            })
            if (i + 1) % 10 == 0:
                print(f"  IG computed : {i+1}/{len(examples)}")
        except Exception as e:
            print(f"  [WW] IG failed example {i+1}: {e}")

    jsonl_path = OUT_DIR / "examples.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"  [OK] examples.jsonl written ({len(records)} records)")

    all_top = {}
    for rec in records:
        for e in rec["top10_tokens"]:
            all_top[e["token"]] = all_top.get(e["token"], 0.0) + e["attribution"]
    top20 = sorted(all_top.items(), key=lambda x: x[1], reverse=True)[:20]

    with open(OUT_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump({
            "model": "deberta", "stratum": STRATUM,
            "n_examples": len(records), "n_correct": n_cor, "n_incorrect": n_inc,
            "top20_tokens_by_cumulative_attribution": [
                {"token": t, "cumulative_attribution": round(s, 4)} for t, s in top20
            ],
        }, f, indent=2, ensure_ascii=False)
    print("  [OK] summary.json written")

    if len(records) == N_EXAMPLES:
        print("\nDeBERTa stratum_iii COMPLETE.")
    else:
        print(f"\n[XX] Only {len(records)}/{N_EXAMPLES} records written.")
    sys.exit(0 if len(records) == N_EXAMPLES else 1)

if __name__ == "__main__":
    main()
