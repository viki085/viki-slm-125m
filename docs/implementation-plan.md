# Implementation Plan: Viki 125M SLM

Source of truth for design: `slm-125m-python-sql-guide.md` (the "guide"). This plan orders the work, defines deliverables and exit gates, and marks where a human decision or spend approval is required. Guide section numbers are in brackets.

Working rules: pure functions are written test-first (pytest, 80%+ coverage); `config.py` is the single source of truth; every heavy step is fanned out one worker per shard; each phase ends with a gate and a short report before the next begins; GPU spend needs explicit approval.

> Cost and speed optimization: see `cost-and-time-plan.md`. It changes the data phases (P2 to P5) to a fused, tokenizer-first, massively parallel design and adds a small pilot pretrain before P6.

## Dependency overview

```
P0 Scaffold+Decisions -> P1 Feasibility -> P2 Data fetch+clean -> P3 Dedup/decontam
   -> P4 Tokenizer -> P5 Tokenize/pack -> P6 Pretrain (GPU $)
                         \                         \
                          P7 SFT data (parallel from P2) -> P8 SFT -> P9 Eval -> P10 Release
```

P7 (SFT data) can start as soon as P1 finishes and runs in parallel with P2 to P6, which is the main schedule saving.

---

## Phase 0: Scaffold and decisions (local, no cost)

Tasks
1. Create repo skeleton: `config.py`, `cleaning.py`, `dedup.py`, `modal_app.py`, `sql_verify.py`, `sandbox.py`, `sft_data.py`, `tests/`, `DATA_CARD.md`, `MODEL_CARD.md`, `requirements.txt`. (`.env.local` and `.gitignore` already exist.)
2. Write `config.py` [7.1]: model (verify `approx_params()` is ~124.3M), agent tokens, `Source` dataclass, paths, `TrainConfig`.
3. Test-first: `tests/test_config.py` (param count, token uniqueness, paths).
4. Fill `.env.local` with Modal and HuggingFace tokens; run `modal token`/`modal profile current`; create Volume `viki-slm-125m`.
5. Settle the open decisions [11] that block later phases: teacher model and its terms, release license, GPU budget cap, primary SQL dialect (recommend SQLite/DuckDB), MHA vs GQA (recommend MHA as written).

Deliverables: skeleton, passing config tests, authenticated Modal, decisions recorded in `DECISIONS.md`.
Gate: `python config.py` prints ~124.3M; `modal volume ls` works.

## Phase 1: Feasibility and license ledger (CPU, about $0)

Tasks
1. Smoke test: stream 10 docs per slice (Kaggle notebooks, cosmopedia-v2, fineweb-edu, PleIAs/SEC, finemath, stack-edu metadata) and print excerpts.
2. Verify every `[VERIFY]` dataset id (text-to-SQL seeds, finance Q&A, DS instruct sets); record id, license, gated flag, rows, columns in `/data/raw_meta/license_ledger.json`.
3. `measure` step: estimated clean tokens per slice; measure stack-edu `permissive` token share for Python and SQL; time a 10K-file Software Heritage fetch to get throughput and failure rate.
4. Decide supply-chain data strategy: quantify public supply-chain text; if under ~150M tokens, commit to synthetic generation in P7.
5. Revise `DATA_MIX` budgets and `epochs` from measurements.

Deliverables: license ledger, measurement report, updated `config.DATA_MIX`.
Gate: ledger complete; no slice depends on an unverified or unlicensed source; mix totals ~8B tokens or the target is revised.

## Phase 2: Fetch and clean (CPU, about $2 to $15)

Tasks (tests first for each cleaner)
1. `cleaning.py` per-kind chains [7.2]: text, python (ast.parse, autogen, line-length, secrets), sql (sqlglot parse, INSERT cap), notebook normalization, finance/supply-chain text, PII/secret scrub. Unit tests with good/bad fixtures per rule.
2. Phase 1a code fetch worker: stack-edu Python and SQL, filter `permissive`, size bounds, prefer `int_score` 4 and 5, S3 fetch with retries and backoff, one worker per metadata shard, outputs to `/data/raw_text/`.
3. `modal_app.py`: image (pip before `add_local_python_source`), `clean_shard` fan-out, drop report aggregation.
4. Run per slice, review samples of kept and dropped documents by hand.

Deliverables: `/data/clean/<slice>/shard-XX.txt`, `phase_clean_report.json`.
Gate: keep rates sane (code 30 to 60%); manual sample review shows no code mangled; secrets scrub verified by a planted-secret test.

## Phase 3: Dedup and decontamination (CPU, about $1 to $5)

Tasks
1. `dedup.py`: exact hash, MinHash/LSH across the whole code slice, comment-insensitive code normalization; tests with planted duplicates.
2. Benchmark decontamination [7.3]: build n-gram and solution-window sets for HumanEval, MBPP, Spider, BIRD, DS-1000 (verify ids in P1); test with planted contaminated files.
3. Fan-out writer; per-reason drop counts.

Deliverables: `/data/corpus/<slice>/`, `phase_dedup_report.json`, contamination counts in `DATA_CARD.md`.
Gate: planted duplicates and contaminated samples are removed; counts per benchmark recorded.

## Phase 4: Tokenizer (CPU, about $0)

Tasks
1. Stratified training sample (web capped at 30%; minimum SQL/Python/finance/supply-chain shares).
2. Train 32,768 byte-level BPE; digit-splitting; whitespace-run tokens; add chat and agent tokens [4.1].
3. Tests: agent tokens are single ids; round-trip exact on code/SQL/unicode; fertility report versus GPT-2 on Python, SQL, finance, supply-chain text.

Deliverables: `/data/tokenizer/`, fertility report.
Gate: vocab == 32768, all special tokens single-id, round-trip passes, code/SQL fertility clearly better than baseline. Changing the tokenizer later invalidates P5 to P8.

## Phase 5: Tokenize and pack (CPU, about $1 to $5)

Tasks
1. `tokenize_shard`: append `<|eos|>`, pack 2,048 uint16 windows, 99/1 split, honour `epochs`, write per-slice val sets.
2. `index.json` with realized token mix.
3. Test: window length, dtype, no id above vocab, deterministic split, EOS boundaries.

Deliverables: `/data/tokens/{train,val}/`, `index.json`.
Gate: realized mix reported; train token count within 10% of plan; per-slice val sets exist. Re-run the cost estimate for P6 here, from the real token count.

## Phase 6: Pretraining (GPU, about $25 to $60) - REQUIRES SPEND APPROVAL

Tasks
1. `train.py`: LlamaForCausalLM from config, SDPA/FlashAttention, bf16, DDP/FSDP, WSD schedule, document-boundary attention masking, resumable checkpoints every 500 steps, per-domain val loss, tokens/sec, grad norm.
2. Unit tests on CPU with a tiny model: loss decreases on a toy set, checkpoint resume reproduces loss, schedule values.
3. 200-step trial on 1 GPU: measure tokens/sec, re-estimate total cost; set `BUDGET_CAP_USD` with hard stop.
4. Human approves spend. Run full pretraining; annealing stage on the best slices in the final ~10%.
5. Generation smoke test at each checkpoint (code completion, SQL completion, finance and supply-chain sentence completion).

Deliverables: `/data/checkpoints/base/`, loss curves, per-domain val-loss table.
Gate: smooth curve with no unrecovered spikes; every slice's val loss decreases; smoke generations are coherent.

## Phase 7: SFT data (free-first, about $0 to $10; paid teacher ON HOLD) - runs in parallel from P1

Revision: no paid teacher at scale. Build data from public datasets, programmatic (template + execution) generation, Kaggle-notebook mining, and an optional free-GPU small open-weight teacher, as detailed in `cost-and-time-plan.md` section 1.2. Items below that say "teacher" apply only to the optional free small teacher; every example still passes the execution and numeric checkers. Re-evaluate paid generation only after Phase 9 shows a gap.

Tasks [6.0 to 6.6]
1. `sql_verify.py` and `sandbox.py` first, with tests (read-only enforcement, timeouts, memory cap, no network).
2. Pull general and tool-calling data from smol-smoltalk/smoltalk (6.0) and public text-to-SQL seeds.
3. Synthetic generators: supply-chain and finance schemas with sample data, questions, gold SQL; execute and keep only verified examples; self-repair pairs; insight data with numeric-faithfulness checker; Python/notebook tasks executed in the sandbox; domain SME Q&A with consistency and numeric checks; multi-turn traces; clarify/refuse examples.
4. Formatting to the section 4 contract; loss masks; dedup; decontaminate against Spider/BIRD/HumanEval/MBPP/DS-1000.
5. Hold out `sft_val` and an unseen-schema domain test set.
6. Spend a small pilot budget first (about 1K examples), inspect by hand, then scale.

Deliverables: `/data/sft/{train,val}.jsonl`, unseen-schema test set, teacher usage log.
Gate: 100% of SQL/Python training examples execute; faithfulness checker pass rate above threshold; hand review of 100 random examples is clean.

## Phase 8: Supervised fine-tuning (GPU, about $2 to $10)

Tasks
1. `sft.py`: assistant-only loss masks, packed sequences with masking, 2 to 3 epochs, LR 2e-5 to 5e-5, 5 to 10% pretraining-style replay.
2. Early stopping on `sft_val` loss and execution metrics.
3. Optional: rejection-sampling round with verified-correct outputs; optional DPO.

Deliverables: `/data/checkpoints/sft/`.
Gate: valid-format rate above 98% (balanced tags, stops after `</sql>`/`</python>`), execution accuracy improving on the unseen-schema test set.

## Phase 9: Evaluation (about $2 to $10)

Tasks [8, Phase 8 table]
1. Harnesses: HumanEval, MBPP, DS-1000, Spider dev EX, BIRD dev subset, internal domain SQL, self-repair, insight faithfulness, domain Q&A, HellaSwag/ARC/PIQA, scripted agent-loop environment, safety prompts.
2. Run base and SFT; fixed prompt formats and decoding recorded.
3. Hand review of 50 random agent transcripts.
4. Write results honestly, including weak ones.

Deliverables: `/data/eval/` results, eval report.
Gate: agent-loop success rate on target domains meets the target agreed in P0; no benchmark contamination found on audit.

## Phase 10: Release (about $0)

Tasks
1. Convert to HF `LlamaForCausalLM` + safetensors; tokenizer, `generation_config.json` stop strings, chat template.
2. Logit-parity test between training checkpoint and HF checkpoint.
3. Optional GGUF/quantized exports; test they still follow the format.
4. `MODEL_CARD.md`, `DATA_CARD.md`, example agent loop (SQL executor + notebook runner), license chosen from the ledger.
5. Push base and instruct repos to HuggingFace after human review.

Gate: parity test passes, example agent loop runs end to end on a clean machine, licenses reconciled.

---

## Critical path and timeline (rough, one person plus agents)

| Block | Phases | Effort |
|---|---|---|
| Foundations | P0, P1 | 2 to 3 days |
| Data | P2 to P5 | 1 to 2 weeks (fetch throughput is the main unknown) |
| SFT data (parallel) | P7 | 1 to 2 weeks |
| Training | P6, P8 | 2 to 4 days incl. trial and review |
| Eval and release | P9, P10 | 3 to 5 days |

## Top risks and mitigations

| Risk | Mitigation |
|---|---|
| Software Heritage fetch is slow or rate-limited | Measure in P1; sample by score and license before fetching; retries with backoff; cap at the token budget |
| Permissive-licensed SQL/Python yield is below plan | Report in P1; shrink mix or add synthetic code; do not fall back to `no_license` without explicit decision |
| Thin supply-chain data | Synthetic generation with verified schemas; up-sample at most 2 to 3 epochs |
| Benchmark contamination | Decontaminate at P3 and in SFT; planted-sample tests; audit in P9 |
| Teacher output terms conflict with open release | Decide in P0; log teacher and terms in DATA_CARD |
| Training instability or wasted GPU spend | CPU unit tests, 200-step trial, hard budget cap, resumable checkpoints |
| 2,048 context too small in the app | Truncate result tables in SFT data to match deployment |
| Silent data bugs after a tokenizer or mix change | Pipeline is ordered and idempotent; re-run all downstream phases after any change |

## Immediate next actions

1. Confirm the open decisions in P0 (teacher model, license, budget cap, SQL dialect).
2. Start P0: write `config.py` and its tests.
3. Start P1 measurements in parallel with P0's decisions.
