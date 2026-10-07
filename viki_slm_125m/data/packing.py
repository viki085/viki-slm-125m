"""Pure helpers for Phase 5: pack token streams into fixed windows, split train/val, index."""

from __future__ import annotations

from typing import Iterable, Iterator, Mapping, Sequence

import numpy as np

UINT16_LIMIT = 1 << 16


def pack_stream(docs_ids: Iterable[Sequence[int]], eos_id: int, seq_len: int) -> Iterator[np.ndarray]:
    """Concatenate documents (each followed by EOS) and yield full uint16 windows.

    The incomplete tail (fewer than seq_len tokens) is dropped.
    """
    buf: list[int] = []
    for ids in docs_ids:
        if ids and max(ids) >= UINT16_LIMIT:
            raise ValueError("token id does not fit in uint16")
        buf.extend(ids)
        buf.append(eos_id)
        while len(buf) >= seq_len:
            yield np.asarray(buf[:seq_len], dtype=np.uint16)
            del buf[:seq_len]


def route_window(index: int, val_every: int) -> str:
    """Every `val_every`-th window (starting at 0) goes to validation."""
    if val_every <= 0:
        raise ValueError("val_every must be positive")
    return "val" if index % val_every == 0 else "train"


def split_windows(windows: Iterable[np.ndarray], val_every: int) -> tuple[list[np.ndarray], list[np.ndarray]]:
    train: list[np.ndarray] = []
    val: list[np.ndarray] = []
    for i, w in enumerate(windows):
        (val if route_window(i, val_every) == "val" else train).append(w)
    return train, val


def build_index(shards: Sequence[Mapping], seq_len: int, dtype: str) -> dict:
    """Aggregate per-file window counts into totals, per-source stats and the realized mix.

    `epochs` (repeat factor applied by the trainer) weights the realized mix but never
    duplicates data on disk, so validation windows can never leak into repeated train data.
    """
    sources: dict[str, dict] = {}
    for s in shards:
        a = sources.setdefault(s["source"], {"train_windows": 0, "val_windows": 0,
                                             "epochs": s["epochs"], "files": 0})
        a["train_windows"] += s["train_windows"]
        a["val_windows"] += s["val_windows"]
        a["files"] += 1
    effective_total = 0.0
    for a in sources.values():
        a["train_tokens"] = a["train_windows"] * seq_len
        a["val_tokens"] = a["val_windows"] * seq_len
        a["effective_train_tokens"] = int(a["train_tokens"] * a["epochs"])
        effective_total += a["effective_train_tokens"]
    mix = {n: a["effective_train_tokens"] / effective_total for n, a in sources.items()} \
        if effective_total else {}
    return {
        "seq_len": seq_len, "dtype": dtype,
        "train_windows": sum(a["train_windows"] for a in sources.values()),
        "val_windows": sum(a["val_windows"] for a in sources.values()),
        "train_tokens": sum(a["train_tokens"] for a in sources.values()),
        "val_tokens": sum(a["val_tokens"] for a in sources.values()),
        "effective_train_tokens": int(effective_total),
        "sources": sources, "realized_mix": mix, "shards": list(shards),
    }
