"""Batched text generation with left padding and stop tokens."""

from __future__ import annotations

import torch
from tokenizers import Tokenizer


@torch.no_grad()
def generate(model, tok: Tokenizer, prompts: list[str], stop_tokens: list[str], max_new: int,
             device: str, batch: int = 32, **gen_kwargs) -> list[str]:
    """One continuation per prompt (greedy unless gen_kwargs enable sampling), cut after the first stop token.

    Prompts are processed in length order for padding efficiency; results keep the input order.
    """
    pad = tok.token_to_id("<|pad|>")
    stops = [tok.token_to_id(t) for t in stop_tokens]
    out: list[str] = [""] * len(prompts)
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    for s in range(0, len(order), batch):
        idx = order[s:s + batch]
        enc = [tok.encode(prompts[i]).ids for i in idx]
        width = max(len(e) for e in enc)
        ids = torch.tensor([[pad] * (width - len(e)) + e for e in enc], device=device)
        attn = torch.tensor([[0] * (width - len(e)) + [1] * len(e) for e in enc], device=device)
        gen = model.generate(input_ids=ids, attention_mask=attn, max_new_tokens=max_new,
                             eos_token_id=stops, pad_token_id=pad, **{"do_sample": False, **gen_kwargs})
        for row, i in zip(gen, idx):
            new = row[width:].tolist()
            for k, t in enumerate(new):
                if t in stops:
                    new = new[:k + 1]
                    break
            out[i] = tok.decode(new, skip_special_tokens=False)
    return out
