"""Loading trained weights and the tokenizer."""

from __future__ import annotations

import os

import torch
from tokenizers import Tokenizer

from viki_slm_125m import config
from viki_slm_125m.model.architecture import build_model

DEFAULT_TOKENIZER = "artifacts/tokenizer/tokenizer.json"


def load_model(path: str, device: str, model_cfg: config.ModelConfig = config.MODEL,
               dtype: torch.dtype = torch.bfloat16):
    """Model with the weights from a training checkpoint ({"model": state_dict, ...}), in eval mode."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"checkpoint not found: {path}")
    model = build_model(model_cfg)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True)["model"])
    # casting the model would also round the non-trainable rotary frequencies; training kept them in fp32
    rotary = {n: b.clone() for n, b in model.named_buffers() if n.endswith("inv_freq")}
    model = model.to(dtype).to(device).eval()
    for name, freq in rotary.items():
        owner = model.get_submodule(name.rsplit(".", 1)[0])
        setattr(owner, "inv_freq", freq.to(device))
    return model


def load_tokenizer(path: str = DEFAULT_TOKENIZER) -> Tokenizer:
    if not os.path.exists(path):
        raise FileNotFoundError(f"tokenizer not found: {path}")
    return Tokenizer.from_file(path)
