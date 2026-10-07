"""Convert chat-style records (smoltalk) into SFT turns."""

from __future__ import annotations

import random
from typing import Iterable, Mapping, Sequence

from viki_slm_125m.sft.sft_data import Turn


def neutralise_specials(text: str) -> str:
    """Stop user-provided text from forming our reserved tokens (e.g. <|assistant|>)."""
    return text.replace("<|", "< |")


def messages_to_turns(messages: Sequence[Mapping]) -> list[Turn] | None:
    """[system?] user assistant (user assistant)* -> turns, or None if malformed."""
    turns: list[Turn] = []
    expected = "user"
    for i, m in enumerate(messages):
        role, content = m.get("role"), (m.get("content") or "").strip()
        if role == "system" and i == 0 and content:
            turns.append(Turn("system", neutralise_specials(content)))
            continue
        if role != expected or not content:
            return None
        turns.append(Turn(role, neutralise_specials(content)))
        expected = "assistant" if role == "user" else "user"
    if not turns or turns[-1].role != "assistant":
        return None
    return turns


def mostly_ascii(text: str, threshold: float = 0.97) -> bool:
    return bool(text) and sum(1 for c in text if ord(c) < 128) / len(text) >= threshold


def select_per_source(items: Iterable[Mapping], caps: Mapping[str, int], seed: int) -> list[Mapping]:
    """At most caps[source] items per source, chosen deterministically."""
    by_source: dict[str, list[Mapping]] = {}
    for it in items:
        by_source.setdefault(it["source"], []).append(it)
    rng = random.Random(seed)
    out: list[Mapping] = []
    for source in sorted(by_source):
        pool = by_source[source]
        rng.shuffle(pool)
        out.extend(pool[: caps.get(source, 0)])
    return out
