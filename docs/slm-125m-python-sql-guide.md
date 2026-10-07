# AGENT BRIEF: Build the Viki 125M SLM (Python + SQL Coding Model)

You are an AI coding agent. Follow this brief top to bottom to build, from nothing, an open-source 125M-parameter small language model that writes Python and SQL code, is usable as the code-writing component of an agentic analytics application, and understands finance/banking and supply-chain schemas.

This guide mirrors the structure of `reference.md` (the legal/financial 125M data pipeline) and extends it from "data pipeline only" to the full model lifecycle: data, tokenizer, pretraining, supervised fine-tuning for agentic behaviour, evaluation, and release.

> STATUS OF NUMBERS: Parameter counts below were computed. Token counts, costs and dataset sizes are PLANNING ESTIMATES. Every phase has a `measure` step. Run it and let measured values override the estimates before committing spend. Dataset ids marked `[VERIFY]` must be confirmed to exist (and have a usable license) in Phase 0 before you rely on them.
>
> REVISION 2: The datasets in the HuggingFaceTB family plus three community links were checked against the Hugging Face API (metadata, sizes, licenses, columns). Findings are in section 3.5 and the tables in 3.4, Phase 1 and Phase 6 were updated. One important correction: `python-edu` and `stack-edu` contain file METADATA ONLY (no source text); the code must be fetched separately (section 3.5, Phase 1).

* * *

## 1. PROJECT GOAL (from idea.md)

The model must:

1. Understand a user's natural-language data question.
2. Generate a correct, executable SQL script for it (given a schema in the prompt).
3. Read the returned result and explain insights clearly and concisely.
4. Write Python (pandas, numpy, matplotlib, scikit-learn) into notebook cells to do basic analysis, and react to execution output (errors, tracebacks) by fixing its own code.
5. Act as a subject-matter expert (SME) on supply chain management and finance/banking.
6. Be openly released so anyone can build their own agentic application on it.

Not in scope: the agentic application itself (orchestrator, SQL executor, notebook runner). This brief produces the model and the interface contract the application will rely on (section 4).

* * *

## 2. HOW TO WORK (rules)

1. Go phase by phase. Run one phase, show the result, then continue. Never chain all phases into one silent run.
2. `config.py` is the single source of truth. Every other file imports from it.
3. Pretraining is the only GPU-expensive step. All data phases run on CPU. Print cost with `modal billing report` after each phase.
4. Fan out heavy work one worker per shard (preemption-safe). Never use one giant container.
5. Decontaminate against every benchmark you will report (section 9). A contaminated benchmark score is worthless.
6. Code must be execution-verified wherever possible: SQL is run against SQLite/DuckDB, Python is run in a sandbox. Unverified synthetic code does not enter SFT data.
7. Licenses matter: this model is released openly. Only use permissively licensed data, record every source and license in `DATA_CARD.md`.
8. Never commit secrets (`.env.local` is git-ignored).

* * *

## 3. KEY DESIGN DECISIONS

### 3.1 Pretrain vs fine-tune the domain knowledge: DECISION

The decision left open in idea.md is made here, with reasons.

| Capability | Where it is learned | Why |
|---|---|---|
| Python, SQL, notebook syntax and idioms | Pretraining (heavy share) | Code fluency needs billions of tokens; fine-tuning cannot create it at 125M scale. |
| General English fluency, reasoning prose | Pretraining (edu web text) | Needed to write clear insights. |
| Finance/banking and supply-chain vocabulary and concepts | Pretraining (small, deliberate share, ~10%) PLUS SFT | A 125M model has little capacity. Vocabulary and concepts should be in the base weights and the tokenizer; SFT then teaches how to answer questions. |
| Text-to-SQL, tool calling, notebook loop, self-repair | SFT (supervised fine-tuning) | These are behaviours and formats, best taught with curated instruction data. |
| Preference/robustness | Optional light DPO | Only after SFT metrics are stable. |

Honest limit: a 125M model will be a competent SQL/Python generator on well-specified, schema-grounded tasks, and a shallow SME. It will not match large models on open-ended reasoning or deep domain advice. Design the agentic app to give it schemas, retrieved domain snippets (RAG) and execution feedback; do not rely on it to recall facts.

### 3.2 Architecture (LLaMA-style decoder)

| Setting | Value | Note |
|---|---|---|
| Layers | 12 | |
| Hidden size | 768 | |
| Attention heads | 12 (head dim 64) | kv heads = 12 (MHA). GQA with 4 kv heads is a valid alternative if inference memory matters; re-balance `intermediate_size`. |
| FFN inner (SwiGLU) | 2,560 | Reduced from 3,072 to offset the larger vocab. |
| Vocab | 32,768 | Larger than the legal model's 16K: code, SQL keywords, identifiers and numbers need the room. |
| Context length | 2,048 | Doubled vs reference: schemas + result tables + notebook history do not fit in 1,024. |
| Embeddings | tied | |
| Params | 124.3M (computed) | `approx_params()` in config.py |

### 3.3 Pretraining token budget

Reference used 2.19B tokens. For a code-capable 125M model plan for roughly 8B training tokens (about 64 tokens/param, far past the 20 tok/param compute-optimal point, because small models keep improving with extra data and a smaller model is cheaper to serve). Tokens come from unique data plus at most 2 to 3 epochs over the scarce domain slices. Adjust after the Phase 0 `measure` step.

### 3.4 Planned pretraining mix (token share of ~8B)

| Slice | Share | ~Tokens | Source (all `[VERIFY]` in Phase 0) |
|---|---|---|---|
| Python code (educational quality) | 30% | 2.4B | `HuggingFaceTB/stack-edu` config `Python` (25.3M files, score 3 to 5) or its stricter subset `HuggingFaceTB/smollm-corpus` config `python-edu` (score >= 4). METADATA ONLY: file text must be fetched from Software Heritage (see 3.5). Filter `license_type == "permissive"`. |
| Jupyter / data-science notebooks | 8% | 0.64B | `HuggingFaceTB/issues-kaggle-notebooks` config `kaggle` (580K Kaggle notebooks, ~1.7B tokens, text included, already in markdown + code blocks; verified). Take a quality-filtered ~0.64B slice. License field is empty: see 3.5. |
| SQL (queries, DDL, schemas) | 10% | 0.8B | `HuggingFaceTB/stack-edu` config `SQL` (2.5M files; only ~398K are `permissive`, ~2.1M are `no_license`), fetched from Software Heritage; plus `b-mc2/sql-create-context`, `gretelai/synthetic_text_to_sql` `[VERIFY]` as seed. The SQL config is 9.62B tokens in total (SmolLM2 tokenizer); if permissive files are proportional by count (~16%), expect roughly 1.5B tokens, enough for this slice's 0.8B budget but with many files being data dumps, so the cleaning step (7.2) will cut it further. Measure the real figure in Phase 0. |
| General educational web text | 22% | 1.76B | `HuggingFaceFW/fineweb-edu` (config `sample-10BT`) |
| Textbook-style synthetic explanations | 8% | 0.64B | `HuggingFaceTB/smollm-corpus` config `cosmopedia-v2` (odc-by; text included; verified) for stats/ML/data topics |
| Finance and banking text | 9% | 0.72B | `PleIAs/SEC` (10-K style filings, as in reference), finance Wikipedia articles, public central-bank/regulator documents, finance Q&A seeds (`gbharti/finance-alpaca`, `Josephgflowers/Finance-Instruct-500k`) |
| Supply chain text | 4% | 0.32B | Wikipedia articles under logistics/operations/procurement/inventory categories, open textbooks (CC-licensed), public government/industry reports. This domain is thin in public datasets, so supplement with teacher-generated synthetic explanations (section 8.2). Expect to up-sample (2 to 3 epochs). |
| Math / statistics prose | 5% | 0.4B | `HuggingFaceTB/finemath` configs `finemath-4plus` / `infiwebmath-4plus` (odc-by, text included; verified), open stats textbooks. Use a small slice; this is a data-science model, not a math model. |
| Data-science docs | 4% | 0.32B | permissively licensed docs for pandas, numpy, scikit-learn, matplotlib, SQL dialect manuals (SQLite/PostgreSQL/DuckDB). Check each license. |

Rules:

* The numbers are a starting point. Finance and supply-chain slices will likely cap below plan (see section 8 of reference gotchas: small domain sources cap the mix). Use the reference's "legal-first" principle: take ALL of the scarce domain data, fill the rest with abundant code/web data, and report the realized mix.
* Do NOT use `70/20/10`-style fantasy ratios. Report the realized mix after tokenization.

### 3.5 Dataset verification results (checked against the Hugging Face API)

| Dataset | Verdict | Findings |
|---|---|---|
| `HuggingFaceTB/stack-edu` | USE (pretraining, code) with a fetch step | Ungated. Configs include `Python` (25.3M files), `SQL` (2.5M), `Markdown` (20.7M). Columns are `blob_id, language, repo_name, path, length_bytes, score, int_score, detected_licenses, license_type`: **no file text**. Text must be downloaded from the Software Heritage S3 bucket by `blob_id` (follow the download snippet in the dataset README; content is gzip-compressed). Quality scores are 3 to 5. For SQL, `license_type` is `permissive` for ~398K files and `no_license` for ~2.1M. For an openly released model, train only on `permissive` unless you accept the legal risk of `no_license` code. |
| `HuggingFaceTB/smollm-corpus` | USE (pretraining) | Ungated, odc-by. `cosmopedia-v2` and `fineweb-edu-dedup` ship text. `python-edu` is metadata only (blob_id, score >= 4), same fetch step as stack-edu; it is a subset of the same idea, so use one of them, not both. |
| `HuggingFaceTB/issues-kaggle-notebooks` | USE (pretraining, notebooks) | Ungated. Config `kaggle`: 580K notebooks, 5.2 GB text, ~1.7B tokens, text included, markdown + code blocks. Derived from StarCoder2-Extras / Meta Kaggle Code, with PII and dedup filtering already applied by the authors. Card has no license field: record upstream terms (Kaggle code is user-authored) in the ledger before release. Skip config `issues` (GitHub issue chatter, not useful for this goal). Best fit for the data-science notebook slice. |
| `HuggingFaceTB/finemath` | OPTIONAL (small slice) | Ungated, odc-by. Text included. Large (149 GB). Take a small slice only. |
| `HuggingFaceTB/smoltalk` | USE (SFT, selected configs) | Ungated, 1.1M conversations, designed for small models (the authors tuned SmolLM2-135M/360M on a subset). Card has no single license: new subsets (smol-magpie-ultra, smol-constraints, smol-rewrite, smol-summarize) are Apache-2.0, other subsets keep their original licenses. Useful configs for this project: `self-oss-instruct` (50.7K executed-filtered Python instructions), `apigen-80k` (87.5K function-calling examples, teaches structured tool calls), `systemchats-30k` (varied system prompts), `smol-rewrite` and `smol-summarize` (summarisation helps insight writing), `everyday-conversations` (2.4K, keeps chat manners). Skip math configs (`metamathqa`, `numina-cot`) and `longalign` (context too long for 2,048). The smaller, declared-Apache-2.0 variant `HuggingFaceTB/smol-smoltalk` (485K rows, aimed at 135M/360M models) is the safer starting point. |
| `AlicanKiraz0/Agentic-Chain-of-Thought-Coding-SFT-Dataset` | REFERENCE ONLY | MIT, but fewer than 1,000 rows, distilled by Minimax-M2 from GitHub crawl. Too small to train on. Its JSON schema (`task, context, plan, cot, actions, final_answer`) is a useful model for how to structure our own `<|think|>` plan + tool-action examples. |
| `Crownelius/Complete-FABLE.5-traces-2M` | DO NOT USE | A mirror of 28 community datasets of Claude agent traces (228,968 rows, not 2M despite the name). Rows are raw Claude Code session JSON (tool calls, file paths, UUIDs) in a single `row_json` string, far longer and noisier than a 2,048-token model can use. The card's own header says the data is not the uploader's and belongs to the original uploaders; it declares MIT but provenance and the terms covering model-generated outputs are unclear. It does not contain SQL/data-analysis loops in our format. Skipping avoids both the format mismatch and a license/provenance risk for an open release. |

Other useful HuggingFaceTB datasets found: `smoltalk2` (8.6M rows, SFT/Preference/Mid configs; large, license unclear, optional later for preference tuning), `stackexchange_2025_md` (Q&A markdown; check license per site; candidate for stats/data-science Q&A), `finemath`, `cosmopedia`, `python-edu-annotations`. Organisation page: https://huggingface.co/HuggingFaceTB

Implications: (1) the notebook slice is now concrete (Kaggle), (2) the Python and SQL slices need a Software Heritage fetch stage that the reference pipeline does not have, (3) SQL volume is limited by licensing, which strengthens the need for verified synthetic SQL (Phase 6), and (4) smoltalk gives us cheap general instruction-following and function-calling data so synthetic generation can focus on the domain.

* * *

## 4. INTERFACE CONTRACT (what the agentic app will rely on)

Define this early. SFT data, the tokenizer's special tokens and evaluation all depend on it.

### 4.1 Special tokens

Base: `<|bos|> <|eos|> <|pad|> <|unk|>`.
Chat: `<|system|> <|user|> <|assistant|>`.
Agent tokens (added to the tokenizer so each is a single token):

```
<|schema|> <|/schema|>        database schema block (DDL) given to the model
<|sql|> <|/sql|>              SQL the model wants executed
<|result|> <|/result|>        execution result returned by the app (table / error)
<|python|> <|/python|>        notebook code cell the model wants executed
<|output|> <|/output|>        cell stdout / traceback / figure note returned by the app
<|insight|> <|/insight|>      final plain-language analysis for the user
<|think|> <|/think|>          short planning scratchpad (optional; strip in app if unwanted)
```

### 4.2 The agent turn loop

```
system:     role + dialect + safety rules ("read-only SELECT unless told otherwise")
user:       <|schema|>DDL...<|/schema|>  question
assistant:  <|think|>plan<|/think|> <|sql|>SELECT ...<|/sql|>
[app]       <|result|>rows or ERROR text<|/result|>
assistant:  (if error) corrected <|sql|>  | (if ok) <|insight|>findings<|/insight|>
```

Same loop for `<|python|>` / `<|output|>`. The model must learn to stop after a `<|/sql|>` or `<|/python|>` so the app can execute (set these as stop strings).

### 4.3 Safety defaults trained into SFT

* Default to read-only SQL (`SELECT`, `WITH`). Refuse or ask before `DROP`, `DELETE`, `UPDATE`, `ALTER` unless the system prompt allows it.
* Never invent columns/tables not in `<|schema|>`; if unsure, ask or say what is missing.
* Cite which result rows an insight is based on.

* * *

## 5. PREREQUISITES: accounts and tokens

1. **Modal** account (same as reference): `pip install modal`, `modal token new`, `modal profile current`.
2. **HuggingFace** token with WRITE role (needed to push the final model/datasets; some datasets such as The Stack are gated and need an accepted license plus read token).
3. **Teacher LLM API key** (for synthetic data in Phase 6; keep behind an env var, never in code). Check the teacher's terms permit using outputs to train a released model.
4. `.env.local` (git-ignored):

```bash
MODAL_TOKEN_ID=ak-XXXXXXXX
MODAL_TOKEN_SECRET=as-XXXXXXXX
HUGGINGFACE_TOKEN=hf_XXXXXXXX
TEACHER_API_KEY=XXXXXXXX
```

5. Persistent volume: `modal volume create viki-slm-125m`

* * *

## 6. FILE LAYOUT

```
config.py              single source of truth (model, mix, budgets, paths, thresholds)
cleaning.py            deterministic cleaning: text, code, SQL, notebooks
dedup.py               exact + MinHash dedup, benchmark decontamination
modal_app.py           Modal app: image, Volume, one function per phase
sql_verify.py          run generated SQL against SQLite/DuckDB; compare results
sandbox.py             restricted Python executor for verifying code samples
sft_data.py            build/format SFT conversations (agent loop format)
train.py               pretraining loop (PyTorch + FSDP/DDP) 
sft.py                 supervised fine-tuning
eval/                  HumanEval, MBPP, Spider, BIRD-dev subset, DS-1000, domain QA, agent loop evals
DATA_CARD.md           every source, license, filter, and known bias
MODEL_CARD.md          release documentation
```

On-Volume layout:

```
/data/raw_meta/                         Phase 0 measurements and license ledger
/data/clean/<slice>/shard-XX.txt        Phase 1
/data/corpus/<slice>/shard-XX.txt       Phase 2 (deduped + decontaminated)
/data/tokenizer/                        Phase 3 (32K byte-level BPE + agent tokens)
/data/tokens/{train,val}/*.bin          Phase 4 (uint16 windows of 2048)
/data/tokens/index.json                 Phase 4
/data/sft/{train,val}.jsonl             Phase 6
/data/checkpoints/{base,sft}/           Phases 5 and 7
/data/eval/                             Phase 8 results
```

Sanity check, no Modal needed: `python3 config.py` should print the project name and `model: ~124.3M params | vocab 32768 | 12L/768d/12h`.

* * *

## 7. THE SOURCE FILES

Write these following the reference's patterns (it is the proven skeleton: one worker per shard, `config.py` imports everywhere, `add_local_python_source` AFTER all `pip_install`/`apt_install`). Below are the parts that DIFFER from the reference. Reuse the reference's `dedup.py`, MinHash/LSH flow, tokenizer trainer and tokenizer-shard packer, changing only the values given here.

### 7.1 config.py (key differences)

```python
PROJECT = "viki-slm-125m"
VOLUME_NAME = "viki-slm-125m"
SEQ_LEN = 2_048

@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 32_768
    hidden_size: int = 768
    intermediate_size: int = 2_560        # SwiGLU inner -> ~124.3M params
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    num_key_value_heads: int = 12
    max_position_embeddings: int = 2_048
    rope_theta: float = 10_000.0
    rms_norm_eps: float = 1e-5
    hidden_act: str = "silu"
    tie_word_embeddings: bool = True
    attention_bias: bool = False

AGENT_TOKENS = (
    "<|schema|>", "<|/schema|>", "<|sql|>", "<|/sql|>",
    "<|result|>", "<|/result|>", "<|python|>", "<|/python|>",
    "<|output|>", "<|/output|>", "<|insight|>", "<|/insight|>",
    "<|think|>", "<|/think|>",
)

@dataclass(frozen=True)
class Source:
    name: str
    kind: str              # "text" | "code" | "sql" | "notebook"
    hf_id: str
    token_budget: int
    text_field: str
    split: str = "train"
    config_name: str | None = None
    epochs: float = 1.0    # >1 up-samples scarce domain slices at tokenize time

# Fill DATA_MIX from section 3.4 after Phase 0 confirms ids and sizes.

EVAL_HOLDOUT = (            # decontaminate against ALL of these
    "openai/openai_humaneval", "google-research-datasets/mbpp",
    "xlangai/spider", "birdsql/bird_sql_dev_20251106",   # [VERIFY] ids
    "xlangai/DS-1000",
)
```

Training hyperparameters (starting point for the 125M run; tune with a short sweep):

```python
@dataclass(frozen=True)
class TrainConfig:
    seq_len: int = 2_048
    global_batch_tokens: int = 524_288      # 256 sequences
    lr: float = 6e-4
    min_lr: float = 6e-5
    warmup_tokens: int = 200_000_000
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    beta1: float = 0.9
    beta2: float = 0.95
    seed: int = 1337
    # Consider WSD (warmup-stable-decay) schedule so you can resume/extend
    # and run an end-of-training "annealing" phase on the highest-quality
    # code + SQL + domain data (see Phase 5).
```

### 7.2 cleaning.py (code-aware cleaning, differs from prose cleaning)

The reference chain (line filter, boilerplate strip, repetition, English, OCR) is built for prose and would damage code (short lines, symbols). Use per-kind chains:

* **text slices**: reference chain, minus the OCR gate, plus a PII scrub.
* **code (python)**: keep line structure and indentation. Drop files that: fail `ast.parse` (Python), are auto-generated (header markers), have lines averaging > 100 chars or max line > 1,000, have alphanumeric fraction < 25%, are mostly data blobs/base64, contain more than 50% comments-only lines, or are tiny (< 200 chars). Strip licence headers only if clearly boilerplate. Remove files with secrets (API-key regexes) and redact emails/IPs.
* **sql**: keep only statements that parse (`sqlglot`), normalize whitespace but preserve keywords; drop dumps of INSERT rows beyond a cap per file (they teach memorisation, not SQL); drop files that are mostly data.
* **notebooks**: convert `.ipynb` to a linear text form: markdown cells as `# %% [markdown]` blocks, code cells as `# %%` blocks. Drop cell outputs larger than 2 KB (keep short outputs, they teach output interpretation). Strip embedded images/base64.
* **finance / supply chain text**: reference prose chain; additionally strip tables of pure numbers beyond a ratio, and remove boilerplate (cover pages, signature blocks, exhibit indexes).
* **PII/secret scrub (all kinds)**: emails, phone numbers, API keys, private keys, tokens, account numbers. This is a release-safety requirement for an open model trained on code and filings.

Every function stays pure and deterministic, returning a `CleanResult(kept, text, reason, raw_chars, clean_chars)` like the reference so the drop report works unchanged.

### 7.3 dedup.py

Reuse reference helpers (`normalize`, `exact_hash`, `word_ngrams`, `shingles`). Changes:

* Code is deduplicated on a whitespace-normalized, comment-insensitive form, with MinHash (5-line shingles) across the whole code slice (not only one source like the reference's case-law-only LSH).
* Decontamination: 13-gram word overlap (as reference) for prose benchmarks. For code and SQL benchmarks also drop any file containing a benchmark problem's docstring/question text or its canonical solution (exact normalized match on the solution body or 10-line window).

### 7.4 modal_app.py (extra functions beyond the reference)

* `smoke_test`, `measure_sources`, `clean_shard`, `minhash_shard`, `build_near_dups`, `write_corpus_shard`, `train_tokenizer`, `tokenize_shard`, `write_token_index`: same structure as reference; swap in `config.DATA_MIX`, 2,048 windows, and uint16 (vocab 32,768 fits).
* `train_tokenizer`: set `vocab_size=config.MODEL.vocab_size`; special tokens = `SPECIAL_TOKENS + EXTRA_CHAT_TOKENS + AGENT_TOKENS`; train on a stratified sample so SQL and code are not drowned out (cap web at 30% of tokenizer training text, force a minimum share of SQL/Python/finance/supply-chain). Use a pre-tokenizer that splits digits individually (numbers matter for data work) and keeps whitespace runs as tokens (indentation).
* `train_pretrain` (GPU): see Phase 5.
* `run_sft`, `build_sft_data`, `verify_sql_batch`, `run_eval`: see Phases 6 to 8.

* * *

## 8. RUN THE PIPELINE (phase by phase)

Before every command: `source .env.local && export MODAL_TOKEN_ID MODAL_TOKEN_SECRET`

### Phase 0: feasibility, licenses, measurement (about 0 cost)

1. Smoke test: stream 10 docs per slice, clean, print excerpts (as reference `modal run modal_app.py`).
2. `measure`: estimate true clean-token yield per slice. Compare against the plan in 3.4.
3. Confirm every `[VERIFY]` dataset id exists, is streamable, and has an acceptable license. Write `/data/raw_meta/license_ledger.json` (id, license, gated?, row count, est. tokens).
4. Revise `DATA_MIX` budgets from measured yields (take all of any scarce slice; fill with abundant slices; set `epochs` for up-sampled domain slices).
5. Decide the domain-data plan: if public supply-chain text is below ~150M tokens, schedule the synthetic generation in Phase 6.

Gate: do not proceed until the ledger exists and mix budgets are updated.

### Phase 1: stream + clean (about 0 to a few dollars CPU)

Phase 1a, fetch code text (new, needed for Python and SQL). Verified folder layout: `Python/` = 5 parquet shards (~500 MB each, 2.5 GB total, 21.8B tokens once fetched); `SQL/` = 1 parquet shard (261 MB, 9.62B tokens). The dataset README's reference script uses `boto3` against bucket `softwareheritage`, key `content/<blob_id>`, gzip-decoded UTF-8, and reports about 6 hours on a 16-core `us-east-1` machine for a full download. Data license: follow the Stack v2 licensing (README points to `bigcode/the-stack-v2-train-full-ids`). Because we only need a fraction, do NOT download everything. For `stack-edu` (`Python`, `SQL`) stream the metadata parquet, filter `license_type == "permissive"` and `length_bytes` between 200 B and ~100 KB, then fetch each file's text from the Software Heritage S3 bucket by `blob_id` (anonymous S3 access; follow the README snippet). Fan out one worker per metadata shard, write `/data/raw_text/<slice>/shard-XX.jsonl.gz`, and use retries with backoff (rate limits are likely). Decode with the row's `src_encoding`. Kaggle notebooks, cosmopedia, fineweb-edu and finemath already contain text and skip this step. Measure fetch throughput on 10K files first; 25M Python files will not fit a small budget, so sample by `int_score` (prefer 4 and 5) up to the token budget.

Phase 1b, clean: `modal run modal_app.py::clean` fans out one worker per parquet shard per slice, applying the per-kind chains from 7.2. Expect a drop report per slice. Note: code slices have a lower keep rate than prose (30 to 60% is normal); sanity-check by reading samples of what was dropped and kept.

### Phase 2: dedup + decontaminate (about 0 CPU)

`modal run modal_app.py::dedup`. Stages: MinHash signatures per shard, LSH near-dup pass, per-shard writer (near-dup + exact-dup + benchmark contamination). Print counts removed per reason. Contamination removal counts for HumanEval/MBPP/Spider/BIRD/DS-1000 must be reported in `DATA_CARD.md`.

### Phase 3: tokenizer (about 0 CPU)

`modal run modal_app.py::tokenizer`. Verify and record:

* `vocab_size == 32768`; all agent tokens are single tokens.
* Round-trip exact on code, SQL and unicode samples.
* Fertility (tokens per word / per char) on Python, SQL, finance text, supply-chain text. Compare with the GPT-2 tokenizer as a sanity baseline; code and SQL should be clearly more efficient.
* Digits tokenized individually; indentation of 4/8/12 spaces is cheap.

### Phase 4: tokenize + pack (about 0 CPU)

`modal run modal_app.py::tokenize`. Append `<|eos|>` after each document, pack to 2,048-token uint16 windows, send every 100th window to val (99/1 split, as reference). For up-sampled slices honour `epochs`. Keep a per-slice val set too (val loss per domain is far more informative than one global number). Expect `index.json` with the realized token mix; report it, it will differ from section 3.4.

Optional code-specific packing improvement: when packing code, avoid mixing unrelated files mid-window without an `<|eos|>` separator (already done), and consider masking attention across document boundaries in the trainer.

### Phase 5: pretraining (GPU, the main cost)

Do this only after Phases 0 to 4 are verified and a human approves spend.

* Hardware: 8x H100 (reference config) or fewer GPUs for longer. Rough planning figure: about 6 x 125M x 8B = 6e18 FLOPs; at a realistic ~35 to 45% MFU this is a few GPU-hours-times-eight, i.e. low tens of dollars. Re-estimate from a measured tokens/sec on a 200-step trial run before committing, and set `BUDGET_CAP_USD` in config.py with a hard stop.
* Framework: PyTorch + `torch.compile` + FlashAttention/SDPA, bf16, FSDP or DDP, gradient checkpointing off (model is small).
* Schedule: warmup, then stable, then decay (WSD). Final ~10% of tokens ("annealing" stage): switch the data mix to the highest-quality slices (execution-verified code, SQL, finance/supply-chain, textbook text) and decay LR to `min_lr`. This stage gives disproportionate gains for domain skills in small models.
* Checkpoint every 500 steps to the Volume; training must be resumable (preemption-safe). Log train loss, per-domain val loss (every 1,000 steps), tokens/sec, grad norm.
* Acceptance gates: loss curve smooth (no unrecovered spikes), per-domain val loss decreasing for every slice, and a generation smoke test (see Phase 8) at each checkpoint.

### Phase 6: SFT data creation (free-first; paid teacher ON HOLD)

> REVISION 3: A paid teacher LLM is not required for the first version. Prefer public datasets, programmatic template-and-execute generation, Kaggle-notebook mining, and an optional free-GPU small open-weight teacher. See `cost-and-time-plan.md` section 1.2. Where the subsections below say "teacher", read "free small teacher or programmatic generator". Decide on any paid generation only after Phase 8 evaluation shows a specific gap.

The pretrained base predicts text; SFT turns it into an agent. Build conversations in the section 4 format. Target 150K to 400K high-quality examples; quality and execution verification matter more than count.

6.0 General and tool-calling base (about 10% of SFT; free, from verified datasets)
* From `HuggingFaceTB/smol-smoltalk` / `smoltalk`: `self-oss-instruct` (Python), `apigen-80k` (function calling; reformat its JSON tool calls into our `<|sql|>`/`<|python|>` style or keep a small share as-is for generic tool use), `systemchats-30k`, `smol-summarize`, `smol-rewrite`, `everyday-conversations`. Drop examples over ~1,500 tokens. Record per-config licenses in the ledger.
* Do not use `Crownelius/Complete-FABLE.5-traces-2M` (see 3.5). `AlicanKiraz0/Agentic-Chain-of-Thought-Coding-SFT-Dataset` is too small to train on; copy its plan/cot/actions structure when designing `<|think|>` examples.

6.1 Text-to-SQL (largest share, about 30%)
* Seed from permissive public text-to-SQL sets (`xlangai/spider` train split, `b-mc2/sql-create-context`, `gretelai/synthetic_text_to_sql`, BIRD train) `[VERIFY licenses]`. Keep Spider/BIRD dev/test out entirely.
* Add synthetic schemas in the target domains: supply chain (orders, suppliers, shipments, inventory, warehouses, lead times, BOMs, demand forecasts) and finance/banking (customers, accounts, transactions, loans, repayments, GL entries, risk ratings, FX rates). Teacher generates DDL + sample rows + questions + gold SQL.
* Verify every example by executing the gold SQL on a generated SQLite/DuckDB database (`sql_verify.py`). Drop non-executing or empty-result examples. Where two independently generated SQLs return identical results, keep the shorter one.
* Include: joins, aggregation, window functions, CTEs, date arithmetic, NULL handling, ranking, running totals, cohort/retention queries, and DISTINCT/GROUP BY traps.
* Dialects: default to ANSI/SQLite plus DuckDB/PostgreSQL variants flagged in the system prompt.

6.2 SQL self-repair (about 10%)
* Take wrong/failing SQL (mutate good SQL: wrong column, missing GROUP BY, bad join) plus the real database error text in `<|result|>`; target is the corrected `<|sql|>`. Teaches the execution-feedback loop.

6.3 Insight generation (about 15%)
* Input: question + SQL + `<|result|>` table (various sizes, including empty and truncated results). Target: concise `<|insight|>` that states the finding, quantifies it from the table, flags caveats (small sample, NULLs, truncated rows), and never invents numbers not in the result. Teacher writes; a checker script verifies that every number in the insight appears in (or is directly computable from) the result.

6.4 Python / notebook analysis (about 20%)
* Tasks over small generated CSV/DataFrames: cleaning, groupby, merge, pivot, time series resample, outlier detection, simple regression/classification with scikit-learn, charts with matplotlib. Format as `<|python|>` cells with `<|output|>` results.
* Execute every cell in `sandbox.py` (restricted, timeouts, no network, memory cap). Keep only examples that run. Include traceback-repair pairs (about 20% of this slice).
* Public seeds: `[VERIFY]` data-science instruction sets with permissive licenses. Do not use DS-1000 for training.

6.5 Domain SME Q&A (about 10%)
* Supply chain: inventory models (EOQ, safety stock, reorder point), ABC/XYZ analysis, forecasting accuracy (MAPE/WAPE), bullwhip effect, lead time variability, OTIF, S&OP, procurement, logistics and incoterms, supplier risk, MRP/ERP concepts.
* Finance/banking: financial statements, ratios, interest/amortization/time value of money, credit risk (PD/LGD/EAD), liquidity, capital adequacy and Basel concepts, AML/KYC basics, payments and settlement, accounting entries, FP&A variance analysis.
* Teacher-generated, then filtered: self-consistency check (ask the same question several ways, keep agreeing answers), and a numeric-answer checker for computational questions (compute with Python and compare).
* Ground in retrieved text when possible (answers that quote a provided passage reduce hallucination).
* Mark these answers to include a short "assumptions" statement and avoid regulatory/legal certainty; add disclaimers training examples for advice-like questions.

6.6 Multi-turn agent traces (about 5%)
* Full loops: question -> SQL -> error -> fixed SQL -> result -> Python plot/stat -> insight. Include "ask a clarifying question" and "cannot answer from this schema" examples (about 5% of all SFT data) so the model learns to refuse to guess.

SFT hygiene:
* Loss only on assistant tokens (and on `<|sql|>`/`<|python|>` content), not on `<|result|>`/`<|output|>` blocks (those come from the app).
* Dedup SFT data and decontaminate against Spider/BIRD/HumanEval/MBPP/DS-1000 dev/test.
* Hold out 2% as `sft_val` and a separate domain-schema test set that never appears in training (new schemas, new tables).
* Record teacher model and terms in `DATA_CARD.md`.

### Phase 7: supervised fine-tuning (GPU, small cost)

* Full-parameter fine-tune of the 125M base (cheap enough; LoRA unnecessary).
* 2 to 3 epochs, LR about 2e-5 to 5e-5 cosine, packed sequences with attention masking across examples, batch about 128 sequences of 2,048.
* Mix in 5 to 10% of pretraining-style code/SQL/domain text to reduce forgetting.
* Early-stop on `sft_val` loss and on the execution metrics from Phase 8 (loss alone is not enough for code).
* Optional Phase 7b: rejection-sampling self-improvement. Sample N SQL/Python answers per prompt from the SFT model, execute, keep verified-correct ones, fine-tune again (execution-based filtering is cheap and effective). Optional DPO using verified-correct vs incorrect pairs.

### Phase 8: evaluation (gate for release)

Report all of these, base vs SFT, with the exact prompt format and decoding settings used (greedy, plus pass@k with temperature):

| Skill | Benchmark | Metric |
|---|---|---|
| Python generation | HumanEval, MBPP | pass@1, pass@10 |
| Data-science Python | DS-1000 (pandas/numpy/sklearn/matplotlib subsets) | pass@1 |
| Text-to-SQL | Spider dev (execution accuracy) | EX |
| Text-to-SQL, harder | BIRD dev subset | EX |
| Domain text-to-SQL | internal supply-chain + finance schema test set (section 6.1, unseen schemas) | EX |
| Self-repair | internal set: fix SQL/Python given the error | fix rate within 2 turns |
| Insight faithfulness | internal set | % of numeric claims supported by the result |
| Domain SME | internal Q&A set + finance/supply-chain multiple-choice sets `[VERIFY]` | accuracy |
| General LM | HellaSwag, ARC-Easy, PIQA (sanity vs other ~125M models) | accuracy |
| Agent loop | scripted environment: schema + question + real DB, up to 4 turns | task success rate, avg turns, invalid-format rate |
| Safety | prompts requesting destructive SQL, secret leakage | refusal/ask rate |

Reality check for expectations: a 125M model is expected to be far below large models on HumanEval/DS-1000 and BIRD. Success is relative to other models of this size and, above all, to the agent-loop success rate on the target domains with schema in context. Publish the numbers honestly, including weak ones.

Also run a qualitative review of 50 random agent transcripts by hand before release.

### Phase 9: release

1. Convert to a standard `LlamaForCausalLM` checkpoint with `safetensors`, plus tokenizer, `generation_config.json` (stop strings for `<|/sql|>`, `<|/python|>`, `<|eos|>`), and a chat template implementing section 4.
2. Verify the HF checkpoint reproduces the training-framework logits (max abs diff tiny) on test prompts.
3. Optional: GGUF / quantized (Q8, Q4) exports for CPU and edge agentic use, and ONNX if needed.
4. Write `MODEL_CARD.md`: intended use, the agent interface contract, training data summary, benchmark table with caveats, limitations (shallow SME, hallucination risk, no guarantee of SQL correctness; always validate/sandbox), license, and safety guidance (read-only DB credentials, sandboxed code execution, human review for consequential decisions, not financial or supply-chain advice).
5. Write `DATA_CARD.md`: every source, license, filter, dedup/decontamination counts, PII scrub description.
6. Push to HuggingFace (`HF_REPO = "<your-namespace>/viki-slm-125m"`), with the base and the SFT ("-instruct") variants as separate repos. Release code for the data pipeline and an example agent loop (SQL executor + notebook runner) so others can build apps.
7. Choose the license consistent with all training data and teacher terms (Apache-2.0 is the usual target; confirm nothing in the data ledger forbids it).

* * *

## 9. EXPECTED RESULTS AND COST (estimates; replace with measured values)

| Phase | What | Compute | Estimated cost |
|---|---|---|---|
| 0 | Feasibility, license ledger | CPU | about $0 |
| 1 | Stream + clean | CPU, fanned out | $1 to $10 (code slices are larger than the reference corpus) |
| 2 | Dedup + decontaminate | CPU | $1 to $5 |
| 3 | Tokenizer | CPU | about $0 |
| 4 | Tokenize + pack | CPU | $1 to $5 |
| 5 | Pretraining ~8B tokens | 8x H100, a few hours | about $25 to $60 (measure first; hard cap in config) |
| 6 | Synthetic SFT data | Teacher API + CPU verification | $30 to $300 depending on teacher and volume |
| 7 | SFT | 1 to 8 GPUs, under 1 hour | $2 to $10 |
| 8 | Evaluation | GPU/CPU | $2 to $10 |

Final artifacts: ~8B-token tokenized corpus on the Volume (with per-slice val), base checkpoint, SFT checkpoint, eval report, model card, data card, HF release.

* * *

## 10. GOTCHAS (do not relearn these)

1. Scarce domain data caps the mix. Finance and especially supply-chain public text is small; take all of it, up-sample modestly (at most 2 to 3 epochs), and fill the rest from abundant code/web data. Over-repeating tiny data causes memorization.
2. Modal image rule: all `pip_install`/`apt_install` BEFORE `add_local_python_source`.
3. Prose cleaning destroys code. Use the per-kind chains in 7.2; never apply the line-length filter or alphanumeric-ratio filter to code or SQL.
4. Token counts in the data phases are a chars/4 proxy; for code the ratio is closer to 3 chars/token. Only Phase 4 gives true counts, so trust it for budget decisions.
5. Execution-verify synthetic SFT data. LLM-written SQL that has never been executed is wrong often enough to teach the model bad habits.
6. Teacher-data circularity: if the teacher also wrote the eval items, scores are inflated. Keep internal eval sets generated differently from training data, with unseen schemas.
7. Numbers: tokenize digits individually, and include arithmetic-style/numeric reasoning data; otherwise insights misquote figures. Always train insight data with a faithfulness checker.
8. Do not rely on the model to remember facts. Design the app around schema-in-prompt, retrieved domain snippets and execution feedback.
9. Context budget: 2,048 tokens fills fast. Truncate result tables (top-N rows plus summary stats) in the app, and train SFT examples with truncated results so behaviour matches deployment.
10. Preemption: keep every heavy step fanned out and resumable (checkpoint every 500 steps in pretraining).
11. Never commit `.env.local`, API keys, or scraped private data. Run the PII/secret scrub before anything is uploaded or released.
12. `python-edu` and `stack-edu` hold only file metadata. Plan and budget the Software Heritage fetch stage (Phase 1a), and filter by `license_type` before fetching, not after.
13. Dataset names can mislead: `Complete-FABLE.5-traces-2M` has ~229K rows, not 2M, and is raw agent-session JSON. Always check row counts, columns and license in Phase 0 before relying on a dataset.
14. To change how much data you keep, edit `token_budget`/`epochs` per slice in `config.py`, then re-run Phase 1 and every phase after it in order. Changing the tokenizer invalidates Phases 4 to 7.

* * *

## 11. OPEN DECISIONS FOR THE OWNER

These do not block starting Phase 0, but should be settled before Phase 5 spend:

1. **Teacher model for synthetic data** and its terms of use (affects release license and cost).
2. **Release license** (Apache-2.0 vs a more restrictive one) after the data ledger is complete.
3. **Compute budget cap** for pretraining (config `BUDGET_CAP_USD`).
4. **Primary SQL dialect** to optimize first (SQLite/DuckDB for easy verification vs PostgreSQL for realism).
5. **GQA vs MHA** (MHA chosen here for simplicity; GQA lowers KV-cache memory for serving the agent).
6. **Domain emphasis**: if only one of supply chain or finance can be done well first, which one? (Finance has far more public data; supply chain will lean on synthetic data.)
