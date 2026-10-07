# Page 11: Export and final delivery

**Code:** `viki_slm_125m/model/export.py`, `model/checkpoint.py`, `model/assistant.py`, `pyproject.toml`.

## 1. Packaging the model
`python -m viki_slm_125m.model.export` converts the training checkpoint (`artifacts/sft_v5/best.pt`, step 2,040) into
a standard Hugging Face folder, `models/viki-slm-125m/`:

| File | Content |
|---|---|
| `model.safetensors` | The 124,275,456 weights in bf16 (249 MB; tied embeddings stored once) |
| `config.json`, `generation_config.json` | Architecture, plus the start, end and padding token ids |
| `tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json` | The 32,768-token tokenizer with all reserved tokens registered |
| `README.md` | Model card: prompt format, stop tokens, usage code, measured results, limitations, data licences |

It loads with `AutoModelForCausalLM.from_pretrained` and `AutoTokenizer.from_pretrained`, with no project code.

**Checks.** The export was compared with the project's own pipeline: 30 of 30 greedy generations were identical, and
tensor-by-tensor comparison showed identical weights. Unit tests cover the files written, the tokenizer's special tokens,
the generation config and the model card.

**A real bug found by that check.** Casting a model to bf16 also rounded its non-trainable rotary position frequencies,
while training kept them in fp32. `load_model` now keeps them in fp32. The effect was small (mean logit error to fp32
0.0418 against 0.0402); the v5 re-evaluation moved gretel from 57.3% to 58.0% and left the domain templates unchanged.

## 2. Packaging the code
* The importable package is `viki_slm_125m` (hyphens are not valid in import names); the project name `viki-slm-125m` is
  set in `pyproject.toml`, which also lists dependencies and the static page as package data.
* Model code is in `viki_slm_125m/model/`: `architecture.py` (build the model), `checkpoint.py` (load weights and
  tokenizer), `generation.py` (batched generation), `assistant.py` (a `VikiSLM` class with `sql_for` and `chat`),
  `export.py`.
* 342 tests pass (overall coverage about 96% when last measured).

## 3. Delivery checklist
| Deliverable | Location |
|---|---|
| Trained model, ready to load | `models/viki-slm-125m/` |
| Training checkpoints (base, SFT v1 to v5) | `artifacts/` |
| Tokenizer | `artifacts/tokenizer/` |
| Model and pipeline code | `viki_slm_125m/`, `modal_app.py` |
| Playground | `python -m viki_slm_125m.app.ui_server` |
| Fine-tuning datasets and benchmark data | `data/sft/`, `data/external/` |
| All evaluation reports and the Modal bill | `reports/` |
| Documentation | `README.md`, `docs/` (this guide, DATA_CARD, DECISIONS, MODEL_CARD, plans) |
| Cloud data export (for training on another service) | `Desktop/viki-slm-125m-modal-export/` (see below) |

## 4. The cloud data export
So that training can continue on another provider, the data on the Modal volume was downloaded into its own folder,
`Desktop/viki-slm-125m-modal-export/downloaded/` (uncompressed, same layout as the volume), and compressed into
separate zip files with checksums in `Desktop/viki-slm-125m-modal-export/zips/`:

| Zip | Content | Needed for |
|---|---|---|
| `tokenizer_and_meta.zip` | Tokenizer and the Phase 1 measurements and licence ledger | everything |
| `checkpoints_base-e1.zip` | Final base checkpoint, the pre-decay snapshot (step 12,870), metrics | resuming or extending pretraining |
| `tokens.zip` | The packed training and validation tokens (7.03B + 71M) and their index | pretraining |
| `corpus.zip` | The cleaned, deduplicated text | re-tokenizing or adding data |

Not exported, on purpose: `clean/` (the cleaning output before deduplication, superseded by `corpus/`), the pilot and
trial checkpoints, and temporary files. `RESTORE.md` in that folder explains how to unpack and resume training.
The fine-tuning data was never on the volume; it is under `data/sft/` in the project.

## 5. Open decisions before a public release
1. **Spider licence.** The fine-tuning data includes Spider's training split (CC BY-SA 4.0: attribution and share-alike).
   Decide whether that applies to the weights and what the licence of the release will be.
2. **Pretraining data licences.** Stack-Edu files were filtered to permissive licences; the Kaggle notebooks declare no
   single licence (see the licence ledger).
3. **What the release claims.** The model card says Python and SQL coding model, not a knowledge model, and lists its limits.
4. **Hosting.** The playground runs locally; publishing needs a Hugging Face repo (`HF_REPO` in `config.py`) and a host.

## 6. State at delivery
Base model `base-e1` (7.5B tokens, $40.83 cloud cost including data processing), fine-tuned to SFT v5: Spider dev 33.0% with
voting (22.1% greedy), 58.0% on open-ended gretel SQL, 100% on pandas and behaviour tests of the trained styles, and
weak on general knowledge. The next gains would come from more real multi-table SQL data and, for knowledge, more
pretraining on supply-chain and data-science text (page 12).
