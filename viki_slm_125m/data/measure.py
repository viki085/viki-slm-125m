"""Pure helpers for Phase 1 (feasibility, license ledger, measurements)."""

from __future__ import annotations

from typing import Iterable


def estimate_clean_tokens(total_rows: int, avg_clean_chars: float, chars_per_token: float) -> int:
    if chars_per_token <= 0:
        raise ValueError("chars_per_token must be positive")
    if total_rows < 0 or avg_clean_chars < 0:
        raise ValueError("total_rows and avg_clean_chars must be non-negative")
    return int(total_rows * avg_clean_chars / chars_per_token)


def license_from_card(info: dict) -> str | None:
    """License string from an HF dataset API payload, or None if undeclared."""
    card = info.get("cardData") or {}
    lic = card.get("license")
    if isinstance(lic, list):
        lic = ",".join(str(x) for x in lic)
    if lic:
        return str(lic)
    for tag in info.get("tags") or []:
        if tag.startswith("license:"):
            return tag.split(":", 1)[1]
    return None


def summarize_stack_edu(rows: Iterable[tuple[str, int, int]]) -> dict:
    """rows: (license_type, length_bytes, int_score) from stack-edu metadata."""
    files = perm_files = perm_hi = 0
    total_bytes = perm_bytes = 0
    for license_type, length_bytes, int_score in rows:
        files += 1
        total_bytes += length_bytes
        if license_type == "permissive":
            perm_files += 1
            perm_bytes += length_bytes
            if int_score >= 4:
                perm_hi += 1
    return {
        "files": files,
        "permissive_files": perm_files,
        "permissive_file_share": perm_files / files if files else 0.0,
        "permissive_byte_share": perm_bytes / total_bytes if total_bytes else 0.0,
        "permissive_int_score_4plus_files": perm_hi,
        "total_bytes": total_bytes,
        "permissive_bytes": perm_bytes,
    }


def summarize_fetch(n_ok: int, n_missing: int, n_error: int,
                    total_bytes: int, elapsed_s: float) -> dict:
    attempted = n_ok + n_missing + n_error
    return {
        "attempted": attempted,
        "ok": n_ok,
        "missing": n_missing,
        "errors": n_error,
        "files_per_sec": attempted / elapsed_s if elapsed_s > 0 else 0.0,
        "failure_rate": (n_missing + n_error) / attempted if attempted else 0.0,
        "avg_bytes_ok": total_bytes / n_ok if n_ok else 0.0,
        "elapsed_s": elapsed_s,
    }


def project_fetch_seconds(files_needed: int, files_per_sec_per_worker: float, workers: int) -> float:
    if files_per_sec_per_worker <= 0 or workers <= 0:
        raise ValueError("rate and workers must be positive")
    return files_needed / (files_per_sec_per_worker * workers)
