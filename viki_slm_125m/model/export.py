"""Export a training checkpoint as a self-contained Hugging Face model folder.

Usage: python -m viki_slm_125m.model.export [checkpoint] [--out models/viki-slm-125m] [--dtype bfloat16]
The folder loads with AutoModelForCausalLM / AutoTokenizer.from_pretrained and needs no project code.
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
from transformers import PreTrainedTokenizerFast

from viki_slm_125m import config
from viki_slm_125m.model.architecture import build_model, count_parameters
from viki_slm_125m.model.checkpoint import DEFAULT_TOKENIZER

MODEL_CARD = """---
language: en
library_name: transformers
pipeline_tag: text-generation
tags: [text-to-sql, python, pandas, small-language-model, llama]
datasets:
- gretelai/synthetic_text_to_sql
- xlangai/spider
---
# Viki SLM 125M

A {params} parameter Llama-style decoder (12 layers, 768 hidden, 12 heads, 2,048 context, 32,768 BPE vocab),
pretrained from scratch on Python, SQL, notebooks and web text, then fine-tuned (SFT v5, step {step}) to
**write SQL for a given schema and pandas code for CSV files**. It is a Python and SQL coding model, not a
general chat or knowledge model.

## Prompt format
```
<|bos|><|system|>You help users analyze data. Given a schema and a question, write a single SELECT query for SQLite, wait for the result, then summarize the findings clearly.<|user|><|schema|>
CREATE TABLE singer (id INTEGER, name TEXT, age INTEGER);
<|/schema|>
How many singers are older than 30?<|assistant|>
```
The model replies with an optional `<|think|>...<|/think|>` plan, then `<|sql|>QUERY<|/sql|>`. Run the query, append
`<|result|>TABLE<|/result|><|assistant|>`, and it writes `<|insight|>...<|/insight|>`. For pandas tasks the
schema lists CSV files and the model writes `<|python|>...<|/python|>`, then explains the `<|output|>`.
Stop on `<|/sql|>`, `<|/python|>` or `<|eos|>`.

## Use
```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained("{folder}"); model = AutoModelForCausalLM.from_pretrained("{folder}").eval()
ids = tok(prompt, return_tensors="pt", add_special_tokens=False)
stops = [tok.convert_tokens_to_ids(t) for t in ("<|/sql|>", "<|eos|>")]
out = model.generate(**ids, max_new_tokens=220, eos_token_id=stops, pad_token_id=tok.pad_token_id, do_sample=False)
print(tok.decode(out[0][ids["input_ids"].shape[1]:]))
```
For best SQL accuracy, sample 8 queries (temperature 0.7), execute them and keep the majority result; retry with the
database error when none runs (see the project's `viki_slm_125m/app/ui_server.py`).

## Measured results (project evaluation, execution against real databases)
| Test | Result |
|---|---|
| Spider dev (1,034 questions), greedy / 8-sample voting | 22.1% / 33.0% |
| gretel unseen domains (600), greedy / voting | 58.0% / 60.2% |
| Held-out finance and supply-chain templates, greedy | 61.4% (66.7% rounding-tolerant); first four 89%, three newer 24% |
| pandas tasks on unseen data (template-style questions) | 100% |
| Refusal / clarification / missing data on unseen schemas | 100% |

## Limitations
- Wrong tables or columns are common on schemas with many tables; always read the generated SQL.
- Not reliable for free-text questions (statistics, machine-learning or domain concepts): answers are fluent but
  often wrong. Use it only for SQL and pandas code.
- Consistently misses some patterns (for example ascending order for "lowest" or "shortest").
- Trained from scratch with 7.5B pretraining tokens; no supply-chain text in pretraining.

## Data and licences
Pretraining and SFT sources are listed in the project's DATA_CARD.md. SFT includes the Spider training split
(cc-by-sa-4.0, attribution and share-alike apply), gretelai/synthetic_text_to_sql (apache-2.0) and generated
finance and supply-chain examples.
"""


def export_hf(checkpoint: str, tokenizer_path: str, out_dir: str, model_cfg: config.ModelConfig = config.MODEL,
              dtype: torch.dtype = torch.bfloat16) -> str:
    """Write config, weights (safetensors), tokenizer, generation config and a model card to out_dir."""
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"tokenizer not found: {tokenizer_path}")
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = build_model(model_cfg)
    model.load_state_dict(state["model"])
    model = model.to(dtype).eval()
    fast = PreTrainedTokenizerFast(
        tokenizer_file=tokenizer_path, model_max_length=model_cfg.max_position_embeddings, padding_side="left",
        model_input_names=["input_ids", "attention_mask"],
        additional_special_tokens=list(config.EXTRA_CHAT_TOKENS + config.AGENT_TOKENS), **config.SPECIAL_TOKENS)
    gen = model.generation_config
    gen.bos_token_id = fast.bos_token_id
    gen.eos_token_id = fast.eos_token_id
    gen.pad_token_id = fast.pad_token_id
    model.config.bos_token_id, model.config.eos_token_id = fast.bos_token_id, fast.eos_token_id
    model.config.pad_token_id = fast.pad_token_id
    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir, safe_serialization=True)
    fast.save_pretrained(out_dir)
    card = MODEL_CARD.format(params=f"{count_parameters(model) / 1e6:.1f}M", step=state.get("step", "?"),
                             folder=out_dir.replace("\\", "/"))
    with open(os.path.join(out_dir, "README.md"), "w", encoding="utf-8") as fh:
        fh.write(card)
    return out_dir


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint", nargs="?", default="artifacts/sft_v5/best.pt")
    p.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    p.add_argument("--out", default="models/viki-slm-125m")
    p.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    args = p.parse_args(argv)
    export_hf(args.checkpoint, args.tokenizer, args.out, dtype=getattr(torch, args.dtype))
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
