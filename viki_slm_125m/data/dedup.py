"""Pure helpers for Phase 3: exact and near-duplicate removal, benchmark decontamination."""

from __future__ import annotations

import hashlib
import re
from typing import Iterable, Mapping, Sequence

import numpy as np

SHINGLE_K = 5
MINHASH_PERM = 32
MINHASH_THRESHOLD = 0.8
MINHASH_MAX_WORDS = 5_000
DECONTAM_NGRAM = 13
DECONTAM_MIN_RUN = 10  # consecutive matching windows (~22 words); filters shared boilerplate

_WS = re.compile(r"\s+")
_WORD = re.compile(r"[a-z0-9]+")


def normalize(text: str) -> str:
    return _WS.sub(" ", text.lower()).strip()


def words(text: str) -> list[str]:
    return _WORD.findall(normalize(text))


def _strip_comment_lines(text: str, markers: tuple[str, ...]) -> str:
    kept = [ln for ln in text.splitlines()
            if ln.strip() and not ln.strip().startswith(markers)]
    return "\n".join(kept)


def canonical_for_hash(text: str, kind: str) -> str:
    """Form used for exact-duplicate hashing: comment-insensitive for code and SQL."""
    if kind == "code":
        text = _strip_comment_lines(text, ("#",))
    elif kind == "sql":
        text = _strip_comment_lines(text, ("--", "#"))
    return normalize(text)


def exact_hash(text: str) -> int:
    """64-bit hash of the normalized text."""
    digest = hashlib.blake2b(normalize(text).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big")


def duplicate_indices(hash_arrays: Sequence[np.ndarray]) -> list[np.ndarray]:
    """For hash arrays in file order, return per-file indices of non-first occurrences."""
    if not hash_arrays:
        return []
    sizes = [len(a) for a in hash_arrays]
    total = sum(sizes)
    if total == 0:
        return [np.array([], dtype=np.int64) for _ in hash_arrays]
    flat = np.concatenate([a.astype(np.uint64) for a in hash_arrays])
    _, first = np.unique(flat, return_index=True)
    is_dup = np.ones(total, dtype=bool)
    is_dup[first] = False
    out, start = [], 0
    for size in sizes:
        out.append(np.flatnonzero(is_dup[start:start + size]).astype(np.int64))
        start += size
    return out


def word_ngrams(tokens: Sequence[str], n: int) -> set[int]:
    """Native hashes of word n-grams (valid within one process only)."""
    if len(tokens) < n:
        return set()
    return {hash(tuple(tokens[i:i + n])) for i in range(len(tokens) - n + 1)}


def record_text(record: Mapping, fields: Iterable[str]) -> str:
    """Join the string and list-of-string values of the named fields."""
    parts: list[str] = []
    for name in fields:
        value = record.get(name)
        if isinstance(value, str):
            parts.append(value)
        elif isinstance(value, (list, tuple)):
            parts.extend(v for v in value if isinstance(v, str))
    return "\n".join(parts)


MIN_DISTINCT_TOKENS = 6
MIN_ALPHA_TOKENS = 4


def informative_ngram(window: Sequence[str]) -> bool:
    """Reject windows that are mostly numbers or repeated tokens (tables, arrays, padding)."""
    return (len(set(window)) >= MIN_DISTINCT_TOKENS
            and sum(1 for t in window if not t.isdigit()) >= MIN_ALPHA_TOKENS)


def build_contamination_grams(texts_by_benchmark: Mapping[str, Iterable[str]],
                              n: int = DECONTAM_NGRAM) -> dict[int, str]:
    grams: dict[int, str] = {}
    for name, texts in texts_by_benchmark.items():
        for text in texts:
            toks = words(text)
            for i in range(len(toks) - n + 1):
                window = toks[i:i + n]
                if informative_ngram(window):
                    grams.setdefault(hash(tuple(window)), name)
    return grams


def find_contamination(text: str, grams: Mapping[int, str], n: int = DECONTAM_NGRAM,
                       min_run: int = DECONTAM_MIN_RUN) -> str | None:
    """Benchmark whose n-grams match `min_run` consecutive windows of the text, else None.

    A single shared 13-gram is usually common boilerplate (imports, standard calls); real
    contamination copies a long contiguous span.
    """
    if not grams:
        return None
    toks = words(text)
    run, run_name = 0, None
    for i in range(len(toks) - n + 1):
        hit = grams.get(hash(tuple(toks[i:i + n])))
        if hit is None:
            run, run_name = 0, None
            continue
        run = run + 1 if hit == run_name else 1
        run_name = hit
        if run >= min_run:
            return hit
    return None


def shingles(tokens: Sequence[str], k: int = SHINGLE_K) -> set[bytes]:
    if len(tokens) < k:
        return {" ".join(tokens).encode("utf-8")} if tokens else set()
    return {" ".join(tokens[i:i + k]).encode("utf-8") for i in range(len(tokens) - k + 1)}


def near_duplicate_ids(texts: Mapping[str, str] | Iterable[tuple[str, str]],
                       threshold: float = MINHASH_THRESHOLD,
                       num_perm: int = MINHASH_PERM) -> list[str]:
    """Ids (in input order) of documents that are near-duplicates of an earlier one.

    Accepts a mapping or an iterable of (id, text) pairs, so large corpora can be streamed.
    """
    from datasketch import MinHash, MinHashLSH

    pairs = texts.items() if isinstance(texts, Mapping) else texts
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    dups: list[str] = []
    for doc_id, text in pairs:
        m = MinHash(num_perm=num_perm)
        sh = shingles(words(text)[:MINHASH_MAX_WORDS])
        if sh:
            m.update_batch(sh)
        if lsh.query(m):
            dups.append(doc_id)
        else:
            lsh.insert(doc_id, m)
    return dups
