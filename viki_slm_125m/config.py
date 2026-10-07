"""Single source of truth for the Viki 125M SLM build (see the guide, section 7.1)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

PROJECT = "viki-slm-125m"
HF_REPO = "your-user/viki-slm-125m"  # set to your own HF namespace for release

VOLUME_NAME = "viki-slm-125m"
DATA_ROOT = "/data"
RAW_META_DIR = f"{DATA_ROOT}/raw_meta"    # Phase 1: license ledger, measurements
RAW_TEXT_DIR = f"{DATA_ROOT}/raw_text"    # Phase 2a: fetched code text
CLEAN_DIR = f"{DATA_ROOT}/clean"          # Phase 2b
CORPUS_DIR = f"{DATA_ROOT}/corpus"        # Phase 3
TOKENIZER_DIR = f"{DATA_ROOT}/tokenizer"  # Phase 4
TOKENS_DIR = f"{DATA_ROOT}/tokens"        # Phase 5
TRAIN_TOKENS_DIR = f"{TOKENS_DIR}/train"
VAL_TOKENS_DIR = f"{TOKENS_DIR}/val"
SFT_DIR = f"{DATA_ROOT}/sft"              # Phase 7
CKPT_DIR = f"{DATA_ROOT}/checkpoints"     # Phases 6 and 8
BASE_CKPT_DIR = f"{CKPT_DIR}/base"
SFT_CKPT_DIR = f"{CKPT_DIR}/sft"
RESUME_CKPT_PATH = f"{CKPT_DIR}/ckpt.pt"
METRICS_PATH = f"{CKPT_DIR}/metrics.jsonl"
EVAL_DIR = f"{DATA_ROOT}/eval"            # Phase 9

SEQ_LEN: int = 2_048


@dataclass(frozen=True)
class ModelConfig:
    """Maps 1:1 to transformers.LlamaConfig. ~124.3M params with tied embeddings."""

    vocab_size: int = 32_768
    hidden_size: int = 768
    intermediate_size: int = 2_560        # SwiGLU inner
    num_hidden_layers: int = 12
    num_attention_heads: int = 12         # head dim 64
    num_key_value_heads: int = 12         # == heads -> MHA
    max_position_embeddings: int = SEQ_LEN
    rope_theta: float = 10_000.0
    rms_norm_eps: float = 1e-5
    hidden_act: str = "silu"              # SwiGLU
    tie_word_embeddings: bool = True
    attention_bias: bool = False

    def to_llama_kwargs(self) -> dict:
        return {
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "max_position_embeddings": self.max_position_embeddings,
            "rope_theta": self.rope_theta,
            "rms_norm_eps": self.rms_norm_eps,
            "hidden_act": self.hidden_act,
            "tie_word_embeddings": self.tie_word_embeddings,
            "attention_bias": self.attention_bias,
        }

    def approx_params(self) -> int:
        h, i = self.hidden_size, self.intermediate_size
        kv = self.num_key_value_heads * (h // self.num_attention_heads)
        attn = h * h + 2 * (h * kv) + h * h
        mlp = 3 * h * i
        per_layer = attn + mlp + 2 * h
        return self.vocab_size * h + self.num_hidden_layers * per_layer + h


MODEL = ModelConfig()

SPECIAL_TOKENS: Mapping[str, str] = {
    "bos_token": "<|bos|>",
    "eos_token": "<|eos|>",
    "pad_token": "<|pad|>",
    "unk_token": "<|unk|>",
}
EXTRA_CHAT_TOKENS: tuple[str, ...] = ("<|user|>", "<|assistant|>", "<|system|>")
AGENT_TOKENS: tuple[str, ...] = (
    "<|schema|>", "<|/schema|>", "<|sql|>", "<|/sql|>",
    "<|result|>", "<|/result|>", "<|python|>", "<|/python|>",
    "<|output|>", "<|/output|>", "<|insight|>", "<|/insight|>",
    "<|think|>", "<|/think|>",
)


def all_special_tokens() -> tuple[str, ...]:
    """Every reserved token, in the order they are added to the tokenizer."""
    return (*SPECIAL_TOKENS.values(), *EXTRA_CHAT_TOKENS, *AGENT_TOKENS)


@dataclass(frozen=True)
class Source:
    name: str
    kind: str                       # "text" | "light" | "code" | "sql" | "notebook"
    hf_id: str
    token_budget: int               # stop at ~this many clean tokens (chars/4 proxy)
    text_field: str = "text"
    split: str = "train"
    config_name: str | None = None
    epochs: float = 1.0             # >1 up-samples scarce slices at tokenize time
    needs_fetch: bool = False       # metadata only; text comes from Software Heritage
    license_filter: str | None = None  # e.g. "permissive" for stack-edu


# Active slices, budgets from the guide section 3.4 (ids verified against the HF API).
# Phase 1 `measure` overrides these budgets with measured yields.
DATA_MIX: tuple[Source, ...] = (
    Source("python", "code", "HuggingFaceTB/stack-edu", 2_400_000_000,
           config_name="Python", needs_fetch=True, license_filter="permissive"),
    # Phase 1: only ~0.4B permissive SQL tokens exist; up-sample 2 epochs (~0.8B seen).
    Source("sql", "sql", "HuggingFaceTB/stack-edu", 400_000_000, epochs=2.0,
           config_name="SQL", needs_fetch=True, license_filter="permissive"),
    Source("notebooks", "notebook", "HuggingFaceTB/issues-kaggle-notebooks", 640_000_000,
           config_name="kaggle"),
    Source("fineweb-edu", "text", "HuggingFaceFW/fineweb-edu", 1_760_000_000,
           config_name="sample-10BT"),
    Source("cosmopedia", "light", "HuggingFaceTB/smollm-corpus", 640_000_000,
           config_name="cosmopedia-v2"),
    Source("finance-sec", "text", "PleIAs/SEC", 720_000_000),
    Source("math", "light", "HuggingFaceTB/finemath", 400_000_000,
           config_name="finemath-4plus"),
)

# Planned slices whose sources are not yet chosen (resolved in Phase 1).
PENDING_SOURCES: Mapping[str, int] = {
    "supply-chain": 320_000_000,   # thin public data; synthetic text + Wikipedia/open texts
    "datasci-docs": 320_000_000,   # permissively licensed library/SQL manuals
}

TARGET_TOKENS: int = 8_000_000_000
CHARS_PER_TOKEN: float = 4.0


def planned_tokens(include_pending: bool = False) -> int:
    total = sum(int(s.token_budget * s.epochs) for s in DATA_MIX)
    if include_pending:
        total += sum(PENDING_SOURCES.values())
    return total


# Held OUT of training; Phases 3 and 7 strip anything resembling these.
EVAL_HOLDOUT: tuple[str, ...] = (
    "openai/openai_humaneval", "google-research-datasets/mbpp",
    "xlangai/spider", "xlangai/DS-1000",   # BIRD dev id: verify in Phase 1
)


@dataclass(frozen=True)
class CleanConfig:
    # prose chain (reference.md)
    min_line_chars: int = 40
    max_nonalnum_ratio: float = 0.30
    min_doc_chars: int = 600
    repetition_top_k: int = 10
    max_repetition_ratio: float = 0.50
    ngram_n: int = 4
    lang_sample_chars: int = 5_000
    # code chain (guide 7.2)
    code_min_chars: int = 200
    code_max_chars: int = 100_000
    code_max_avg_line: int = 100
    code_max_line: int = 1_000
    code_min_alnum_frac: float = 0.25
    code_max_comment_frac: float = 0.50
    sql_max_insert_rows: int = 50


CLEAN = CleanConfig()

VAL_EVERY_N_WINDOWS: int = 100          # every 100th window -> val (99/1 split)
TOKENS_DTYPE: str = "uint16"


@dataclass(frozen=True)
class TrainConfig:
    seq_len: int = SEQ_LEN
    global_batch_tokens: int = 524_288
    lr: float = 6e-4
    min_lr: float = 6e-5
    warmup_tokens: int = 200_000_000
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    beta1: float = 0.9
    beta2: float = 0.95
    ckpt_every_steps: int = 500
    log_every_steps: int = 20
    eval_every_steps: int = 1_000
    seed: int = 1337


TRAIN = TrainConfig()

PRETRAIN_GPU = "H100"
PRETRAIN_GPU_COUNT = 8
PRETRAIN_GPU_USD_PER_HOUR: float = 3.95   # assumed Modal H100 list price; verify before spend
BUDGET_CAP_USD: float = 150.0        # whole project (Standard tier + contingency)
TRAIN_BUDGET_CAP_USD: float = 90.0   # pretraining only; training aborts + checkpoints here

STAGES: tuple[str, ...] = (
    "setup", "measure", "fetch_clean", "dedup", "tokenizer", "tokenize",
    "pretrain", "sft_data", "sft", "eval", "release",
)


if __name__ == "__main__":
    p = MODEL.approx_params()
    print(PROJECT)
    print(f"model: {p:,} params (~{p / 1e6:.1f}M) | vocab {MODEL.vocab_size} | "
          f"{MODEL.num_hidden_layers}L/{MODEL.hidden_size}d/{MODEL.num_attention_heads}h")
    print(f"planned tokens: active {planned_tokens() / 1e9:.2f}B, "
          f"with pending {planned_tokens(True) / 1e9:.2f}B "
          f"(target {TARGET_TOKENS / 1e9:.1f}B)")
    print(f"stages: {' -> '.join(STAGES)}")
