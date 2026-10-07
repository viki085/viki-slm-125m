"""SFT dataset: load JSONL mixtures, encode with loss masks, token-budget batching, collation."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import torch

from viki_slm_125m.sft import sft_data


@dataclass(frozen=True)
class Encoded:
    ids: list[int]
    mask: list[int]          # 1 where the token is trained


def turns_from_record(rec: Mapping) -> list[sft_data.Turn]:
    return [sft_data.Turn(t["role"], t["text"], t.get("trained", True)) for t in rec["turns"]]


def load_mixture(sources: Mapping[str, tuple[str, int]],
                 encode: Callable[[str], tuple[list[int], list[tuple[int, int]]]],
                 max_len: int, seed: int) -> dict[str, list[Encoded]]:
    """sources: name -> (jsonl path, max examples). Examples that do not fit max_len are skipped."""
    out: dict[str, list[Encoded]] = {}
    rng = random.Random(seed)
    for name, (path, cap) in sources.items():
        with open(path, encoding="utf-8") as fh:
            records = [json.loads(line) for line in fh]
        rng.shuffle(records)
        kept: list[Encoded] = []
        for rec in records:
            if len(kept) >= cap:
                break
            ex = sft_data.encode_example(turns_from_record(rec), encode, max_len)
            if ex is not None:
                kept.append(Encoded(ex.ids, ex.mask))
        out[name] = kept
    return out


def token_budget_batches(lengths: Sequence[int], max_tokens: int, rng: random.Random,
                         bucket: int = 4096) -> list[list[int]]:
    """Batches of similar-length examples with (longest x count) <= max_tokens; deterministic per rng."""
    order = list(range(len(lengths)))
    rng.shuffle(order)
    batches: list[list[int]] = []
    for start in range(0, len(order), bucket):
        chunk = sorted(order[start:start + bucket], key=lambda i: lengths[i])
        cur: list[int] = []
        cur_max = 0
        for i in chunk:
            new_max = max(cur_max, lengths[i])
            if cur and new_max * (len(cur) + 1) > max_tokens:
                batches.append(cur)
                cur, new_max = [], lengths[i]
            cur.append(i)
            cur_max = new_max
        if cur:
            batches.append(cur)
    rng.shuffle(batches)
    return batches


def collate(examples: Sequence[Encoded], pad_id: int) -> dict[str, torch.Tensor]:
    width = max(len(e.ids) for e in examples)
    ids = torch.full((len(examples), width), pad_id, dtype=torch.long)
    attn = torch.zeros((len(examples), width), dtype=torch.long)
    labels = torch.full((len(examples), width), -100, dtype=torch.long)
    for r, e in enumerate(examples):
        n = len(e.ids)
        row = torch.tensor(e.ids, dtype=torch.long)
        ids[r, :n] = row
        attn[r, :n] = 1
        keep = torch.tensor(e.mask, dtype=torch.bool)
        labels[r, :n] = torch.where(keep, row, torch.tensor(-100))
    return {"input_ids": ids, "attention_mask": attn, "labels": labels}
