# Model Card: Viki SLM 125M

## What it is
A 124,275,456-parameter Llama-style decoder (12 layers, 768 hidden, 12 heads, 2,048 context, 32,768 BPE vocabulary),
pretrained from scratch on 7.5B tokens of Python, SQL, notebooks, finance filings and educational web text (cloud cost
including data processing: $40.83), then fine-tuned (SFT v5) as a **Python and SQL coding model**.

## Intended use
- Write a SQL query from a schema and a question, then explain the executed result in a sentence or two.
- Write pandas code for CSV files and explain its printed output.
- Refuse write requests, ask for clarification, or say data is missing when the schema cannot answer.
- Run inside an agent loop that executes the SQL or code, votes over several samples and retries on errors.

## Not intended for
- Answering general or technical knowledge questions (statistics, machine-learning or finance concepts). It produces
  fluent but often wrong answers. Use a larger model for those.
- Unattended use: always show the generated SQL or code next to its result.

## Where the files are
- Ready-to-load Hugging Face folder: `models/viki-slm-125m/` (weights in bfloat16 safetensors, tokenizer, config,
  model card with prompt format and usage). Re-create it with `python -m viki_slm_125m.model.export`.
- Training checkpoints: `artifacts/` (base model `base-e1`, SFT versions `sft` to `sft_v5`).
- Model code: `viki_slm_125m/model/` (architecture, checkpoint loading, generation, `VikiSLM` facade, export).

## Measured results (SFT v5; execution against real databases)
| Test | Result |
|---|---|
| Spider dev (1,034 questions), greedy / 8-sample voting | 22.1% / 33.0% |
| gretel unseen domains (600), greedy / voting | 58.0% / 60.2% |
| Held-out finance and supply-chain templates, greedy | 61.4% (66.7% rounding-tolerant); first four 89%, three newer 24% |
| pandas tasks on unseen data (template-style questions) | 100% |
| Refusal / clarification / missing data, unseen schemas | 100% |
| Insights whose numbers match the query result | 99.5% |
Full history and caveats: `DECISIONS.md`.

## Known limitations
- Wrong tables or columns on schemas with many tables (about two thirds of Spider dev questions still fail).
- Misses some patterns consistently, for example ascending order for "lowest" or "shortest".
- No supply-chain text in pretraining; chat answers are unreliable.

## Data and licences
See `DATA_CARD.md`. SFT used the Spider training split (cc-by-sa-4.0: attribution and share-alike apply; a release
decision is pending), gretelai/synthetic_text_to_sql (apache-2.0) and generated finance and supply-chain examples.
