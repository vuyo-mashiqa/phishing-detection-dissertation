"""
src/utils/hash_corpora.py

Computes SHA-256 hashes and byte sizes for all raw corpus files.
Writes a permanent manifest to outputs/manifests/corpus_sha256_manifest.json.

Usage:
    python src/utils/hash_corpora.py
"""

import hashlib
import json
import os
from pathlib import Path
from datetime import datetime, timezone


# ------------------------------------------------------------------ #
#  CORPUS FILE REGISTRY                                                #
#  Each entry: (label, relative_path)                                  #
#  Label is used as the key in the output manifest.                    #
# ------------------------------------------------------------------ #
CORPUS_FILES = [
    (
        "stratum_i_enron_tarball",
        "data/raw/stratum_i/enron_mail_20150507.tar.gz"
    ),
    (
        "stratum_i_nazario_20051114",
        "data/raw/stratum_i/20051114.mbox"
    ),
    (
        "stratum_i_nazario_phishing0",
        "data/raw/stratum_i/phishing0.mbox"
    ),
    (
        "stratum_i_nazario_phishing1",
        "data/raw/stratum_i/phishing1.mbox"
    ),
    (
        "stratum_i_nazario_phishing2",
        "data/raw/stratum_i/phishing2.mbox"
    ),
    (
        "stratum_i_nazario_phishing3",
        "data/raw/stratum_i/phishing3.mbox"
    ),
    (
        "stratum_iii_phishfuzzer_variants_USED",
        "data/raw/stratum_iii/PhishFuzzer/PhishFuzzer_emails_entity_rephrased_v1.json"
    ),
    (
        "stratum_iii_phishfuzzer_seeds_EXCLUDED_NEVER_READ",
        "data/raw/stratum_iii/PhishFuzzer/PhishFuzzer_emails_original_seed_v1.json"
    ),
]


def sha256_file(path: str) -> str:
    """
    Compute SHA-256 hash of a file by reading it in 64 KB chunks.
    Reading in chunks avoids loading the entire file into memory,
    which matters for large files like the Enron tarball (443 MB).
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def count_directory(directory: str, extension: str) -> dict:
    """
    For directories containing many small files (e.g. phishing_pot EML files),
    count the number of files and compute the total byte size.
    Individual file hashing is impractical at 7,911 files.
    """
    p = Path(directory)
    if not p.exists():
        return {"status": "NOT_FOUND", "count": 0, "total_bytes": 0}
    files = list(p.rglob(f"*.{extension}"))
    total_bytes = sum(f.stat().st_size for f in files)
    return {
        "status": "OK",
        "count": len(files),
        "total_bytes": total_bytes,
        "note": (
            f"Individual {extension.upper()} files not hashed; "
            "aggregate count and size used for integrity verification."
        )
    }


def main():
    print("=" * 65)
    print("  SHA-256 CORPUS INTEGRITY MANIFEST GENERATOR")
    print("=" * 65)

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "purpose": (
            "SHA-256 integrity hashes for all raw corpus files. "
            "Supersedes MD5 acquisition log as authoritative provenance record. "
            "Any reviewer may re-download files and verify hashes to confirm "
            "byte-for-byte identity with the files used in this study."
        ),
        "hash_algorithm": "SHA-256",
        "files": {},
        "directories": {}
    }

    # ---- Hash individual files ----------------------------------------
    all_ok = True
    for label, filepath in CORPUS_FILES:
        p = Path(filepath)
        if not p.exists():
            print(f"  [MISSING]  {filepath}")
            manifest["files"][label] = {
                "path": filepath,
                "status": "NOT_FOUND"
            }
            all_ok = False
            continue

        size_bytes = p.stat().st_size
        size_mb = size_bytes / (1024 * 1024)
        print(f"  Hashing    {filepath}")
        print(f"             Size: {size_mb:.1f} MB — please wait...", end="", flush=True)

        digest = sha256_file(filepath)

        print(f"\r             SHA-256: {digest[:32]}...  ({size_mb:.1f} MB) [OK]")

        manifest["files"][label] = {
            "path": filepath,
            "sha256": digest,
            "size_bytes": size_bytes,
            "size_mb": round(size_mb, 2),
            "status": "OK"
        }

    # ---- Count phishing_pot directory ------------------------------------
    print(f"\n  Counting   data/raw/stratum_ii/phishing_pot/ (EML files)...")
    pp_info = count_directory("data/raw/stratum_ii/phishing_pot", "eml")
    manifest["directories"]["stratum_ii_phishing_pot"] = {
        "path": "data/raw/stratum_ii/phishing_pot",
        **pp_info
    }
    if pp_info["status"] == "OK":
        print(f"             {pp_info['count']:,} EML files  |  "
              f"{pp_info['total_bytes'] / (1024*1024):.1f} MB total  [OK]")
    else:
        print(f"             [MISSING] phishing_pot directory not found")
        all_ok = False

    # ---- Count CSDMC2010 ham directory -----------------------------------
    print(f"  Counting   data/raw/stratum_ii/csdmc2010/ (ham files)...")
    csdmc_path = "data/raw/stratum_ii/csdmc2010"
    csdmc_files = list(Path(csdmc_path).rglob("*")) if Path(csdmc_path).exists() else []
    csdmc_email_files = [
        f for f in csdmc_files
        if f.is_file()
        and "__MACOSX" not in str(f)
        and not f.name.startswith(".")
    ]
    csdmc_total_bytes = sum(f.stat().st_size for f in csdmc_email_files)
    manifest["directories"]["stratum_ii_csdmc2010"] = {
        "path": csdmc_path,
        "status": "OK" if csdmc_email_files else "NOT_FOUND",
        "file_count_excl_macosx": len(csdmc_email_files),
        "total_bytes_excl_macosx": csdmc_total_bytes,
        "note": "__MACOSX artefact directory excluded from count (macOS ZIP extraction artefact; contains no email data)"
    }
    print(f"             {len(csdmc_email_files):,} files (excl __MACOSX)  |  "
          f"{csdmc_total_bytes / (1024*1024):.1f} MB  [OK]")

    # ---- Write manifest --------------------------------------------------
    output_path = Path("outputs/manifests/corpus_sha256_manifest.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print()
    print("=" * 65)
    if all_ok:
        print("  RESULT: ALL FILES FOUND AND HASHED SUCCESSFULLY")
    else:
        print("  RESULT: ONE OR MORE FILES MISSING — review output above")
    print(f"  Manifest written to: {output_path}")
    print("=" * 65)


if __name__ == "__main__":
    main()
