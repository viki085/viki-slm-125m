"""Pure helpers for Phase 4: sampling, training and checking the byte-level BPE tokenizer."""

from __future__ import annotations

from typing import Iterable, Iterator, Mapping, Sequence

from viki_slm_125m import config

MAX_DOC_CHARS = 20_000
TRAIN_STRIDE = 7
TRAIN_OFFSET = 0
EVAL_OFFSET = 3

# Share of tokenizer-training text per slice. Web-like slices total 30% so code, SQL and
# domain text are well represented in the merges.
TOKENIZER_SHARES: Mapping[str, float] = {
    "python": 0.30, "sql": 0.14, "notebooks": 0.12, "finance-sec": 0.14,
    "fineweb-edu": 0.18, "cosmopedia": 0.08, "math": 0.04,
}


def source_quotas(total: int, shares: Mapping[str, float]) -> dict[str, int]:
    if abs(sum(shares.values()) - 1.0) > 1e-6:
        raise ValueError("shares must sum to 1.0")
    return {name: int(total * share) for name, share in shares.items()}


def sample_docs(docs: Iterable[str], role: str, per_file_chars: int) -> Iterator[str]:
    """Deterministic, disjoint train/eval samples: every 7th doc (offset 0 train, 3 eval)."""
    if role == "train":
        offset = TRAIN_OFFSET
    elif role == "eval":
        offset = EVAL_OFFSET
    else:
        raise ValueError("role must be 'train' or 'eval'")
    used = 0
    for i, doc in enumerate(docs):
        if i % TRAIN_STRIDE != offset:
            continue
        piece = doc[:MAX_DOC_CHARS]
        yield piece
        used += len(piece)
        if used >= per_file_chars:
            return


def build_tokenizer(texts: Iterable[str], vocab_size: int, specials: Sequence[str]):
    """Train a byte-level BPE with single-digit splitting. Special tokens get ids 0..n-1."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    tok = Tokenizer(models.BPE(unk_token=config.SPECIAL_TOKENS["unk_token"]))
    tok.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Digits(individual_digits=True),
        pre_tokenizers.ByteLevel(add_prefix_space=False),
    ])
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size, special_tokens=list(specials), min_frequency=2,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False)
    tok.train_from_iterator(texts, trainer=trainer)
    return tok


def special_token_ids(tok, specials: Sequence[str]) -> dict[str, int]:
    ids = {}
    for s in specials:
        i = tok.token_to_id(s)
        if i is None:
            raise ValueError(f"special token missing from vocab: {s}")
        ids[s] = i
    return ids


def roundtrip_failures(tok, samples: Iterable[str]) -> list[str]:
    return [s for s in samples if tok.decode(tok.encode(s).ids) != s]


def chars_per_token(tok, texts: Sequence[str]) -> float:
    chars = sum(len(t) for t in texts)
    tokens = sum(len(tok.encode(t).ids) for t in texts)
    return chars / tokens if tokens else 0.0
