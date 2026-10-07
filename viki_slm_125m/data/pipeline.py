"""Pure helpers for the fetch + clean workers (Phase 2)."""

from __future__ import annotations

from typing import Iterable, Mapping

MIN_CODE_BYTES = 200
MAX_CODE_BYTES = 100_000


def worker_budget(total_tokens: int, n_workers: int) -> int:
    """Per-worker token cap; rounds up so the sum of caps is never below the total."""
    if n_workers <= 0:
        raise ValueError("n_workers must be positive")
    return -(-total_tokens // n_workers)


def owns_row(index: int, worker: int, n_workers: int) -> bool:
    """Round-robin partition of a shard's rows across workers."""
    if not 0 <= worker < n_workers:
        raise ValueError("worker must be in [0, n_workers)")
    return index % n_workers == worker


def accept_code_row(row: Mapping, *, min_score: int = 3, license_filter: str = "permissive",
                    min_bytes: int = MIN_CODE_BYTES, max_bytes: int = MAX_CODE_BYTES) -> bool:
    """Metadata filter applied BEFORE fetching file text from Software Heritage."""
    return (row.get("license_type") == license_filter
            and int(row.get("int_score", 0)) >= min_score
            and min_bytes <= int(row.get("length_bytes", 0)) <= max_bytes)


def est_tokens(chars: int, chars_per_token: float) -> int:
    return int(chars / chars_per_token)


def merge_reports(reports: Iterable[Mapping]) -> dict:
    merged: dict = {"streamed": 0, "kept": 0, "est_tokens": 0, "reasons": {}}
    for r in reports:
        merged["streamed"] += r["streamed"]
        merged["kept"] += r["kept"]
        merged["est_tokens"] += r["est_tokens"]
        for reason, n in r["reasons"].items():
            merged["reasons"][reason] = merged["reasons"].get(reason, 0) + n
    return merged


def decode_text(raw: bytes, encoding: str | None) -> str:
    """Decode with the recorded encoding, falling back to UTF-8; never raises."""
    for enc in (encoding, "utf-8"):
        if not enc:
            continue
        try:
            return raw.decode(enc, errors="ignore")
        except LookupError:
            continue
    return raw.decode("utf-8", errors="ignore")


def copy_tree_parallel(src: str, dst: str, workers: int = 32) -> int:
    """Copy a directory tree with a thread pool (network volumes copy much faster in parallel)."""
    import os
    import shutil
    from concurrent.futures import ThreadPoolExecutor

    if not os.path.isdir(src):
        raise FileNotFoundError(src)
    jobs = []
    for root, _, files in os.walk(src):
        rel = os.path.relpath(root, src)
        out_dir = os.path.join(dst, rel) if rel != "." else dst
        os.makedirs(out_dir, exist_ok=True)
        jobs += [(os.path.join(root, f), os.path.join(out_dir, f)) for f in files]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(lambda j: shutil.copyfile(*j), jobs))
    return len(jobs)
