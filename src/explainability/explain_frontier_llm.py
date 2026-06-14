"""
Phase 12C -- Explainability: Frontier LLM Rationale Fidelity

"""

import json, os, sys, time, hashlib, re, inspect
from pathlib import Path
import numpy as np
from sklearn.metrics import cohen_kappa_score

ROOT         = Path(__file__).resolve().parents[2]
FRONTIER_DAT = ROOT / "data" / "processed" / "frontier"
OUT_DIR      = ROOT / "results" / "explanations" / "frontier"
LOG_DIR      = ROOT / "outputs" / "frontier_api_logs"
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ── Saka et al. (2024) Phishing Codebook ─────────────────────────────────────
CODEBOOK_CATEGORIES = [
    "urgency",
    "spoofed_sender",
    "call_to_action",
    "credential_harvesting",
    "authority_invocation",
]

MODELS   = ["gpt4o", "claude35"]
VARIANTS = ["cot", "feature_grounded"]
STRATA   = ["stratum_i", "stratum_ii", "stratum_iii"]
N_SAMPLE = 50
TEMPERATURE = 0

MODEL_API_NAMES = {
    "gpt4o":    "gpt-4o-mini",
    "claude35": "claude-sonnet-4-5-20250929",
}

# ── API clients ───────────────────────────────────────────────────────────────
_openai_client = _anthropic_client = None

def get_openai():
    global _openai_client
    if _openai_client is None:
        from openai import OpenAI
        _openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    return _openai_client

def get_anthropic():
    global _anthropic_client
    if _anthropic_client is None:
        import anthropic
        _anthropic_client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    return _anthropic_client

def log_api_call(template_id, body_sha, response_sha):
    with open(LOG_DIR / "phase12c_calls.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({
            "template_id": template_id,
            "redacted_body_sha": body_sha,
            "response_sha": response_sha,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }) + "\n")

# ── Prompt templates ──────────────────────────────────────────────────────────
SYSTEM_BASE = ("You are an expert email security analyst. Classify the following "
               "email as either legitimate or phishing.")

def build_cot_prompt(subject: str, body: str) -> str:
    email_text = f"Subject: {subject}\n\n{body}"[:1800]
    return (f"{email_text}\n\n"
            "First, reason step by step about the features of this email that "
            "indicate whether it is phishing or legitimate. Then on the final "
            "line respond with exactly one word: phishing or legitimate.")

def build_feature_grounded_prompt(subject: str, body: str) -> str:
    email_text = f"Subject: {subject}\n\n{body}"[:1800]
    return (f"{email_text}\n\n"
            "Analyse this email using the following phishing signal taxonomy "
            "(Saka et al., 2024 Phishing Codebook):\n"
            "1. Urgency: time pressure, deadlines, immediate action\n"
            "2. Spoofed sender: impersonation, fake domain, display-name mismatch\n"
            "3. Call to action: click link, download attachment\n"
            "4. Credential harvesting: login, verify account, enter password\n"
            "5. Authority invocation: bank, IT dept, CEO, government, brand\n\n"
            "For each category state whether it is present and why. "
            "Then on the final line respond with exactly one word: "
            "phishing or legitimate.")

def call_api(model_key: str, system: str, user: str) -> str:
    body_sha = hashlib.sha256(user.encode()).hexdigest()
    if model_key == "gpt4o":
        resp = get_openai().chat.completions.create(
            model=MODEL_API_NAMES[model_key], temperature=TEMPERATURE,
            messages=[{"role": "system", "content": system},
                      {"role": "user",   "content": user}],
            max_tokens=600,
        )
        text = resp.choices[0].message.content.strip()
    else:
        resp = get_anthropic().messages.create(
            model=MODEL_API_NAMES[model_key], temperature=TEMPERATURE,
            max_tokens=600, system=system,
            messages=[{"role": "user", "content": user}],
        )
        text = resp.content[0].text.strip()
    log_api_call(f"rationale_{model_key}", body_sha,
                 hashlib.sha256(text.encode()).hexdigest())
    return text

# ── Codebook coding ───────────────────────────────────────────────────────────
CODING_SYSTEM = """You are a research assistant performing content analysis of 
phishing detection RATIONALES (not emails directly).

Your task: read the rationale text below and identify which Saka et al. (2024) 
Phishing Codebook categories the rationale TEXT explicitly mentions as PRESENT.

Categories:
- urgency: rationale mentions time pressure, deadline, immediate action, expires
- spoofed_sender: rationale mentions impersonation, fake domain, display-name mismatch
- call_to_action: rationale mentions click link, download, open attachment, visit URL
- credential_harvesting: rationale mentions login, verify account, enter password, credentials
- authority_invocation: rationale mentions bank, IT dept, CEO, government, brand name

IMPORTANT: 
- Set a category to true ONLY if the rationale explicitly states that feature is PRESENT.
- If the rationale says a feature is "not present" or "absent", set it to false.
- Base your coding SOLELY on what the rationale states, not your own analysis.

Respond with a JSON object only, no other text, no markdown:
{"urgency": true/false, "spoofed_sender": true/false, "call_to_action": true/false, "credential_harvesting": true/false, "authority_invocation": true/false}"""

def code_rationale(rationale: str, coder: str = "gpt4o") -> dict:
    user = (f"CODE THIS RATIONALE TEXT:\n\n{rationale[:1000]}\n\n"
            f"Return JSON only with the five boolean fields.")
    body_sha = hashlib.sha256(user.encode()).hexdigest()
    if coder == "gpt4o":
        resp = get_openai().chat.completions.create(
            model="gpt-4o-mini", temperature=0,
            messages=[{"role": "system", "content": CODING_SYSTEM},
                      {"role": "user",   "content": user}],
            max_tokens=100,
        )
        raw = resp.choices[0].message.content.strip()
    else:
        resp = get_anthropic().messages.create(
            model=MODEL_API_NAMES["claude35"], temperature=0, max_tokens=100,
            system=CODING_SYSTEM,
            messages=[{"role": "user", "content": user}],
        )
        raw = resp.content[0].text.strip()
    log_api_call(f"coding_{coder}", body_sha,
                 hashlib.sha256(raw.encode()).hexdigest())
    # Strip markdown code fences if present
    raw_clean = re.sub(r"```[a-z]*\n?", "", raw).strip()
    try:
        codes = json.loads(raw_clean)
    except Exception:
        # Try extracting JSON object with regex
        m = re.search(r'\{[^}]+\}', raw_clean)
        codes = json.loads(m.group()) if m else {}
    return {c: bool(codes.get(c, False)) for c in CODEBOOK_CATEGORIES}

# ── Perturbation fidelity ─────────────────────────────────────────────────────
PERTURB_SYSTEM = ("You are an expert email security analyst. Classify the "
                  "following email as either legitimate or phishing. "
                  "Respond with exactly one word: legitimate or phishing.")

def extract_attributed_phrase(rationale: str) -> str:
    quoted = re.findall(r'"([^"]{3,60})"', rationale)
    if quoted:
        return quoted[0]
    urls = re.findall(r'https?://\S+', rationale)
    if urls:
        return urls[0]
    for trigger in ["because", "due to", "indicates", "contains", "includes"]:
        m = re.search(rf'{trigger}\s+([A-Za-z0-9@._\-]{{3,40}})', rationale, re.I)
        if m:
            return m.group(1)
    return ""

def perturb_check(model_key: str, body: str,
                  phrase: str, original_label: str) -> dict:
    if not phrase:
        return {"fidelity_checked": False, "reason": "no_attributed_phrase"}
    masked = body.replace(phrase, "[MASKED]")
    if masked == body:
        return {"fidelity_checked": False, "reason": "phrase_not_found_in_body"}
    try:
        new_label_raw = call_api(model_key, PERTURB_SYSTEM, masked[:1800])
        new_label = "phishing" if "phishing" in new_label_raw.lower() else "legitimate"
        flipped   = new_label != original_label
        return {
            "fidelity_checked":  True,
            "attributed_phrase": phrase,
            "original_label":    original_label,
            "new_label":         new_label,
            "label_flipped":     flipped,
            "alignment":         flipped,
        }
    except Exception as e:
        return {"fidelity_checked": False, "reason": str(e)}

# ── Load redacted test examples ───────────────────────────────────────────────
def load_sample(stratum: str, n: int = N_SAMPLE) -> list:
    path = FRONTIER_DAT / f"test_{stratum}_redacted.jsonl"
    if not path.exists():
        return []
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    rng = np.random.default_rng(42)
    if len(rows) > n:
        rows = [rows[i] for i in rng.choice(len(rows), n, replace=False).tolist()]
    return rows

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("Phase 12C -- Frontier LLM Rationale Fidelity")
    print("=" * 60)

    all_ok  = True
    summary = []

    for model_key in MODELS:
        for variant in VARIANTS:
            for stratum in STRATA:
                print(f"\n  [{model_key} | {variant} | {stratum}]")
                out_dir   = OUT_DIR / model_key / variant / stratum
                out_dir.mkdir(parents=True, exist_ok=True)
                done_path = out_dir / "coded_rationales.jsonl"

                if done_path.exists():
                    n = sum(1 for _ in open(done_path, encoding="utf-8"))
                    print(f"    Already complete ({n} records) -- skipping")
                    summary.append({"model": model_key, "variant": variant,
                                    "stratum": stratum, "status": "OK", "n": n})
                    continue

                examples = load_sample(stratum)
                if not examples:
                    print(f"    [XX] Redacted JSONL not found -- skipping")
                    all_ok = False
                    summary.append({"model": model_key, "variant": variant,
                                    "stratum": stratum, "status": "NO_DATA"})
                    continue

                print(f"    Loaded {len(examples)} examples")

                # ── Build prompts and call API ─────────────────────────────────
                coded = []
                for i, ex in enumerate(examples):
                    subj = ex.get("subject_redacted", "")
                    body = ex.get("body_redacted", "")
                    label_int = int(ex.get("label", 0))
                    orig_label = "phishing" if label_int == 1 else "legitimate"

                    try:
                        if variant == "cot":
                            user_prompt = build_cot_prompt(subj, body)
                        else:
                            user_prompt = build_feature_grounded_prompt(subj, body)

                        response  = call_api(model_key, SYSTEM_BASE, user_prompt)
                        rationale = response  # full response IS the rationale

                        # Extract predicted label from last word
                        last_word = response.strip().split()[-1].lower().rstrip(".,!?")
                        pred_label = "phishing" if "phishing" in last_word else "legitimate"
                        pred_int   = 1 if pred_label == "phishing" else 0

                    except Exception as e:
                        print(f"    [WW] API call failed example {i+1}: {e}")
                        time.sleep(5)
                        continue

                    # ── Code rationale (primary rater = GPT-4o-mini) ───────────
                    try:
                        primary_codes = code_rationale(rationale, coder="gpt4o")
                    except Exception as e:
                        print(f"    [WW] Coding failed example {i+1}: {e}")
                        primary_codes = {c: False for c in CODEBOOK_CATEGORIES}

                    # ── Perturbation fidelity check ────────────────────────────
                    phrase   = extract_attributed_phrase(rationale)
                    fidelity = perturb_check(model_key, body, phrase, orig_label)

                    coded.append({
                        "example_id":     i + 1,
                        "message_id":     ex.get("message_id", ""),
                        "label":          label_int,
                        "pred":           pred_int,
                        "correct":        label_int == pred_int,
                        "rationale":      rationale[:600],
                        "codebook_codes": primary_codes,
                        "fidelity":       fidelity,
                    })

                    if (i + 1) % 10 == 0:
                        print(f"    Progress : {i+1}/{len(examples)}")
                    time.sleep(0.5)  # rate limit buffer

                if not coded:
                    print(f"    [XX] No records coded")
                    all_ok = False
                    summary.append({"model": model_key, "variant": variant,
                                    "stratum": stratum, "status": "FAILED"})
                    continue

                # ── Cohen's kappa: 10% sample, second rater = Claude ──────────
                rng = np.random.default_rng(42)
                k_size = min(20, len(coded))
                # Stratified sample: proportional phishing/legitimate
                phish_idx = [i for i, r in enumerate(coded) if r["label"] == 1]
                legit_idx = [i for i, r in enumerate(coded) if r["label"] == 0]
                n_phish = max(1, round(k_size * len(phish_idx) / len(coded)))
                n_legit = k_size - n_phish
                sampled_phish = rng.choice(phish_idx, min(n_phish, len(phish_idx)),
                                           replace=False).tolist() if phish_idx else []
                sampled_legit = rng.choice(legit_idx, min(n_legit, len(legit_idx)),
                                           replace=False).tolist() if legit_idx else []
                k_idx = sampled_phish + sampled_legit
                k_size = len(k_idx)
                r1, r2 = [], []
                for idx in k_idx:
                    rec = coded[idx]
                    try:
                        r2_codes = code_rationale(rec["rationale"], coder="claude35")
                        rec["second_rater_codes"] = r2_codes
                        for c in CODEBOOK_CATEGORIES:
                            r1.append(int(rec["codebook_codes"][c]))
                            r2.append(int(r2_codes[c]))
                    except Exception as e:
                        print(f"    [WW] Second-rater coding failed: {e}")
                    time.sleep(0.5)

                try:
                    if r1 and len(set(r1 + r2)) > 1:
                        kappa = float(cohen_kappa_score(r1, r2))
                    elif r1:
                        # Both raters agree perfectly on all codes — report as 1.0
                        kappa = 1.0
                    else:
                        kappa = None
                except Exception:
                    kappa = None
                kappa_str = f"{kappa:.3f}" if kappa is not None else "N/A"
                print(f"    Cohen's kappa ({k_size} sample) : {kappa_str}")

                # ── Write outputs ──────────────────────────────────────────────
                with open(done_path, "w", encoding="utf-8") as f:
                    for rec in coded:
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                print(f"    [OK] coded_rationales.jsonl ({len(coded)} records)")

                cat_counts = {c: sum(1 for r in coded if r["codebook_codes"].get(c))
                              for c in CODEBOOK_CATEGORIES}
                n_checked  = sum(1 for r in coded if r["fidelity"].get("fidelity_checked"))
                n_aligned  = sum(1 for r in coded if r["fidelity"].get("alignment"))

                with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
                    json.dump({
                        "model": model_key, "variant": variant, "stratum": stratum,
                        "n_examples":           len(coded),
                        "codebook_frequencies": cat_counts,
                        "cohens_kappa":         round(kappa, 3) if kappa else None,
                        "kappa_sample_size":    k_size,
                        "fidelity": {
                            "n_checked":      n_checked,
                            "n_aligned":      n_aligned,
                            "alignment_rate": round(n_aligned / n_checked, 3)
                                              if n_checked > 0 else None,
                        },
                    }, f, indent=2, ensure_ascii=False)
                print(f"    [OK] summary.json")

                summary.append({"model": model_key, "variant": variant,
                                "stratum": stratum, "status": "OK",
                                "n": len(coded), "kappa": kappa})

    # ── Final summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for row in summary:
        mark  = "[OK]" if row["status"] == "OK" else "[XX]"
        kappa = f"  kappa={row['kappa']:.3f}" if row.get("kappa") is not None else ""
        print(f"  {mark}  {row['model']:10s}  {row['variant']:18s}  {row['stratum']}{kappa}")
    print()
    ok_count = sum(1 for r in summary if r["status"] == "OK")
    print(f"  {ok_count}/{len(summary)} configurations complete")
    if all_ok and ok_count == len(summary):
        print("\nPHASE 12C COMPLETE.")
    else:
        print("\nPHASE 12C PARTIAL. Check XX entries above.")
    print("=" * 60)
    sys.exit(0 if all_ok else 1)

if __name__ == "__main__":
    main()
