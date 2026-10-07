"""VikiSLM: a small facade over the model, tokenizer and the prompt formats used in fine-tuning."""

from __future__ import annotations

from typing import Optional

import torch

from viki_slm_125m.eval.evaluator import build_prompt, extract_block
from viki_slm_125m.model.checkpoint import DEFAULT_TOKENIZER, load_model, load_tokenizer
from viki_slm_125m.model.generation import generate
from viki_slm_125m.sft import sft_data

SQL_STOPS = ["<|/sql|>", "<|eos|>"]
CHAT_REPETITION = {"repetition_penalty": 1.15, "no_repeat_ngram_size": 4}


class VikiSLM:
    """Load once, then ask for SQL or free text. This is the model only: it does not run the SQL it writes
    (see viki_slm_125m.app.ui_server for the execute / vote / repair loop)."""

    def __init__(self, model, tokenizer, device: str):
        self.model, self.tokenizer, self.device = model, tokenizer, device

    @classmethod
    def load(cls, checkpoint: str, tokenizer: str = DEFAULT_TOKENIZER, device: Optional[str] = None) -> "VikiSLM":
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
        return cls(load_model(checkpoint, device, dtype=dtype), load_tokenizer(tokenizer), device)

    def sql_for(self, schema: str, question: str, max_new: int = 220) -> Optional[str]:
        """The SQL the model writes for a question about a schema (CREATE TABLE text), or None."""
        user = f"<|schema|>\n{schema.strip()}\n<|/schema|>\n{question.strip()}"
        prompt = build_prompt(sft_data.SYSTEM_PROMPTS[0], user)
        text = generate(self.model, self.tokenizer, [prompt], SQL_STOPS, max_new, self.device)[0]
        return extract_block(text, "sql")

    def chat(self, message: str, system: str = "", max_new: int = 300) -> str:
        """Free-text reply. Unreliable for factual questions: the model is tuned for SQL and pandas code."""
        prompt = "<|bos|>" + (f"<|system|>{system.strip()}" if system.strip() else "") \
            + f"<|user|>{message.strip().replace('<|', '< |')}<|assistant|>"
        text = generate(self.model, self.tokenizer, [prompt], ["<|eos|>"], max_new, self.device, **CHAT_REPETITION)[0]
        return text.replace("<|eos|>", "").strip()
