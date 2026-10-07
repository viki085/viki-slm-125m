# Page 4: The tokenizer

**Runs on:** Modal, CPU. **Code:** `viki_slm_125m/data/tokenizer_lib.py`, `modal_app.py::tokenizer`.
**Output:** `artifacts/tokenizer/` (`tokenizer.json`, config files). **Report:** `reports/phase4_tokenizer_report.json`.

## Why train our own
A general-purpose tokenizer spends many tokens on code: indentation, symbols, long identifiers. A tokenizer fitted to
our corpus represents Python and SQL in fewer tokens, so the model sees more code per training step and has more of its
2,048-token context left for schema and answer.

## What was built
* **Type:** byte-level BPE (every byte can be represented, so there are no unknown characters).
* **Vocabulary:** 32,768 tokens, including 21 reserved tokens.
* **Training data:** a sample of 1.2 billion characters (about 300M tokens) drawn from the cleaned corpus,
  weighted by slice so code and SQL are well represented.
* **Digits are split one per token.** Numbers are built from single digits, which helps arithmetic-like patterns and
  keeps numbers in query results consistent.
* **Runs of indentation are single tokens**, so deeply indented Python costs few tokens.
* **Reserved tokens** are guaranteed to be one token each, so the format never fragments:

| Group | Tokens |
|---|---|
| Control | `<|bos|>` (id 0), `<|eos|>` (1), `<|pad|>` (2), `<|unk|>` (3) |
| Roles | `<|user|>` (4), `<|assistant|>` (5), `<|system|>` (6) |
| Agent steps | `<|schema|>` `<|/schema|>` `<|sql|>` `<|/sql|>` `<|result|>` `<|/result|>` `<|python|>` `<|/python|>` `<|output|>` `<|/output|>` `<|insight|>` `<|/insight|>` `<|think|>` `<|/think|>` |

Putting the agent tokens in the vocabulary from the start (even though pretraining text never contains them) means
fine-tuning only has to give them meaning, not add new embeddings.

## Checks
| Check | Result |
|---|---|
| Encode then decode 2,100 documents | 0 failures |
| Every reserved token is exactly one token | yes |
| Ids match the Hugging Face tokenizer wrapper | yes |
| Digit splitting and a 12-space indent test | pass |

**Compression** (characters per token; higher is better) against the GPT-2 tokenizer on the same text:

| Slice | Ours | GPT-2 | Ours vs GPT-2 |
|---|---|---|---|
| Python | 3.49 | 2.21 | 1.58 times better |
| SQL | 2.67 | 2.38 | 1.12 times |
| Notebooks | 3.12 | 2.52 | 1.24 times |
| Finance (SEC) | 4.64 | 5.15 | 0.90 (slightly worse on this prose) |
| FineWeb-edu | 4.31 | 4.64 | 0.93 |
| Cosmopedia | 4.92 | 5.12 | 0.96 |
| Math | 3.14 | 3.30 | 0.95 |

The trade-off is deliberate: a little worse on general prose, clearly better on the code this model is for.

## Hand-off
The tokenizer is frozen. Changing it later would mean re-tokenizing the corpus and re-training the base model.
