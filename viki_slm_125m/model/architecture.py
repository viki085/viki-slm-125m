"""The 125M architecture: a Llama-style decoder (RoPE, RMSNorm, SwiGLU, tied embeddings).

Hyper-parameters live in config.ModelConfig (12 layers, 768 hidden, 12 heads, 2,560 FFN, 32,768 vocab,
2,048 context = 124,275,456 parameters). The layers come from transformers' Llama implementation.
"""

from __future__ import annotations

from transformers import LlamaConfig, LlamaForCausalLM

from viki_slm_125m import config


def build_model(model_cfg: config.ModelConfig = config.MODEL) -> LlamaForCausalLM:
    """A freshly initialised model for the given configuration."""
    cfg = LlamaConfig(**model_cfg.to_llama_kwargs())
    cfg._attn_implementation = "sdpa"
    return LlamaForCausalLM(cfg)


def count_parameters(model) -> int:
    """Unique parameters (tied embeddings are counted once)."""
    return sum(p.numel() for p in model.parameters())
