"""
src/utils/manifest_utils.py

Reusable experiment manifest utility.
Import this module in every script that produces a processed or
results file to write a permanent, machine-readable audit trail.

Every manifest records:
  - run_id:          UUID unique to this execution
  - timestamp:       UTC ISO-8601 timestamp
  - script_name:     name of the calling script
  - git_commit_sha:  SHA of the current HEAD commit (links output to code)
  - git_is_clean:    whether the working tree was clean at run time
  - python_version:  Python version used
  - random_seed:     random seed used (pass None if not applicable)
  - parameters:      dict of any key parameters used in this run
  - inputs:          list of input files with path and SHA-256
  - outputs:         list of output files with path and SHA-256
  - counts:          dict of key record counts (e.g. total, phishing, legitimate)
  - notes:           free-text notes about this run

The manifest is saved to:
    outputs/manifests/{script_name}_{YYYYMMDD_HHMMSS}.json
"""

import hashlib
import json
import os
import platform
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path


def _sha256_file(path: str) -> str:
    """Compute SHA-256 of a file, reading in 64 KB chunks."""
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
    except (OSError, IOError):
        return "UNREADABLE"


def _sha256_directory(path: str, extension: str = None) -> dict:
    """
    For a directory, return file count and total bytes.
    Optionally filter by extension (e.g. 'eml').
    """
    p = Path(path)
    if not p.exists():
        return {"status": "NOT_FOUND", "count": 0, "total_bytes": 0}
    if extension:
        files = list(p.rglob(f"*.{extension}"))
    else:
        files = [f for f in p.rglob("*") if f.is_file()]
    total = sum(f.stat().st_size for f in files)
    return {"status": "OK", "count": len(files), "total_bytes": total}


def _git_commit_sha() -> str:
    """Return the current HEAD commit SHA, or 'UNKNOWN' if git is unavailable."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5
        )
        return result.stdout.strip() if result.returncode == 0 else "UNKNOWN"
    except Exception:
        return "UNKNOWN"


def _git_is_clean() -> bool:
    """Return True if the working tree has no uncommitted changes."""
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, timeout=5
        )
        return result.stdout.strip() == ""
    except Exception:
        return False


class ManifestWriter:
    """
    Builds and writes a JSON experiment manifest.

    Instantiate at the start of a script, call add_input() and
    add_output() as files are consumed and produced, call set_counts()
    with record statistics, then call write() at the end.
    """

    def __init__(
        self,
        script_name: str,
        random_seed=None,
        parameters: dict = None,
        notes: str = ""
    ):
        self._run_id      = str(uuid.uuid4())
        self._timestamp   = datetime.now(timezone.utc).isoformat()
        self._script_name = script_name
        self._random_seed = random_seed
        self._parameters  = parameters or {}
        self._notes       = notes
        self._inputs      = []
        self._outputs     = []
        self._counts      = {}

        # Capture environment at instantiation time
        self._git_sha   = _git_commit_sha()
        self._git_clean = _git_is_clean()
        self._python    = sys.version

        if not self._git_clean:
            print(
                f"WARNING: Working tree is not clean (uncommitted changes exist). "
                f"The manifest will record git_is_clean=False. "
                f"Commit your changes before running production scripts."
            )

    def add_input(self, path: str, label: str = None):
        """
        Register an input file or directory.
        For files: records SHA-256 and size.
        For directories: records file count and total size.
        """
        p = Path(path)
        entry = {"path": str(path), "label": label or str(path)}
        if p.is_file():
            entry["type"]       = "file"
            entry["sha256"]     = _sha256_file(path)
            entry["size_bytes"] = p.stat().st_size
        elif p.is_dir():
            entry["type"]  = "directory"
            info = _sha256_directory(path)
            entry.update(info)
        else:
            entry["type"]   = "not_found"
            entry["status"] = "NOT_FOUND"
        self._inputs.append(entry)
        return self

    def add_output(self, path: str, label: str = None):
        """
        Register an output file.
        Call this AFTER the file has been written so the SHA-256 is computed
        on the final content.
        """
        p = Path(path)
        entry = {"path": str(path), "label": label or str(path)}
        if p.is_file():
            entry["type"]       = "file"
            entry["sha256"]     = _sha256_file(path)
            entry["size_bytes"] = p.stat().st_size
            entry["row_count"]  = self._count_csv_rows(path)
        else:
            entry["type"]   = "not_found"
            entry["status"] = "NOT_FOUND — file was not produced"
        self._outputs.append(entry)
        return self

    def set_counts(self, counts: dict):
        """
        Record key statistics about the run.
        Example: {"total_files": 7911, "retained": 6800, "dropped_empty": 500}
        """
        self._counts = counts
        return self

    def write(self, output_dir: str = "outputs/manifests") -> str:
        """
        Write the manifest JSON file and return its path.
        Filename: {script_name}_{YYYYMMDD_HHMMSS}.json
        """
        timestamp_safe = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self._script_name}_{timestamp_safe}.json"
        output_path = Path(output_dir) / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)

        manifest = {
            "run_id":         self._run_id,
            "timestamp":      self._timestamp,
            "script_name":    self._script_name,
            "git_commit_sha": self._git_sha,
            "git_is_clean":   self._git_clean,
            "python_version": self._python,
            "random_seed":    self._random_seed,
            "parameters":     self._parameters,
            "notes":          self._notes,
            "counts":         self._counts,
            "inputs":         self._inputs,
            "outputs":        self._outputs,
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        print(f"  Manifest written -> {output_path}")
        return str(output_path)

    @staticmethod
    def _count_csv_rows(path: str) -> int:
        """Count data rows in a CSV (total lines minus 1 for header)."""
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                return sum(1 for _ in f) - 1
        except Exception:
            return -1
