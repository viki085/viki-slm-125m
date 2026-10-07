# Cost and Time Plan: Viki 125M SLM

Companion to `slm-125m-python-sql-guide.md` and `implementation-plan.md`. Goals: (1) know the total cost before spending, (2) minimize wall-clock time of dataset creation plus pretraining.

> All figures are ESTIMATES from published list prices and arithmetic, not measurements. Assumed prices (verify on modal.com/pricing and your teacher provider before approving spend): Modal H100 about $3.95 per GPU-hour (so 8 GPUs about $31.60/hour), Modal CPU about $0.047 per core-hour. The 200-step trial in Phase 6 and the Phase 1 measurements replace these numbers with real ones.

## 1. Total cost

### 1.1 Three budget tiers (REVISED: paid teacher on hold, free-first SFT data)

Decision: the paid teacher (commercial API or rented-GPU teacher at scale) is ON HOLD. SFT data is built with the free-first path in section 1.2. The CPU data-creation line was also corrected to scale from the reference run ($0.18 for 2.2B prose tokens), allowing 3 to 5x on the code share.

Pretraining token budget is the main lever (cost and time scale linearly with it). Throughput assumed 1.2M to 3M tokens/second on one 8xH100 node (about 1.0 GFLOP/token for this model, 25 to 40% MFU), plus 20% overhead for trial runs and restarts.

| Tier | Pretrain tokens | Data creation (CPU) | Pretraining | SFT data (free-first) | SFT training | Eval | Debug/misc | With 25% contingency |
|---|---|---|---|---|---|---|---|---|
| Lean | 4B | $1 to 2 | $14 to 35 | $0 to 5 | $1 to 3 | $2 to 5 | $5 to 10 | **$29 to 75** |
| **Standard (recommended)** | 8B | $1 to 4 | $28 to 70 | $0 to 10 | $1 to 5 | $2 to 8 | $5 to 15 | **$46 to 140** |
| Extended | 12B | $2 to 6 | $42 to 105 | $0 to 10 | $1 to 5 | $2 to 10 | $5 to 15 | **$65 to 189** |

Now pretraining is the dominant cost (about 60 to 75% of the total). Previous totals with a paid teacher were $115 to 298 (Standard); the free-first path saves roughly $70 to 160.

### 1.2 Free-first SFT data (replaces the paid teacher)

No paid LLM is needed to reach a first working agent model. Use these sources, in this order, and measure the result before considering any paid spend:

| Source | Cost | What it provides | Limits |
|---|---|---|---|
| **Public datasets** (already verified or to verify in Phase 1): `smol-smoltalk`/`smoltalk` configs, `xlangai/spider` train, `b-mc2/sql-create-context`, `gretelai/synthetic_text_to_sql` `[VERIFY: license, domain coverage]`, finance Q&A seeds `[VERIFY]` | $0 | General instruction following, tool calling, Python instructions, 100K+ text-to-SQL pairs | Spider/BIRD dev and test must stay out; domain coverage thin for supply chain |
| **Programmatic generation, no LLM** | CPU only, about $0 to 3 | Hand-write about 40 to 60 supply-chain and finance schemas; generate sample data with code; question templates with slots produce gold SQL by construction; execute SQL to get results; build insight text with templates from computed numbers (faithful by construction); inject errors for repair pairs; clarify/refuse examples from rules | Less linguistic variety; mitigate with many paraphrase templates and by letting the model paraphrase questions later (below) |
| **Mining Kaggle notebooks** (`HuggingFaceTB/issues-kaggle-notebooks`, already in our corpus) | CPU only | Markdown-cell-to-code-cell pairs become Python task examples; run cells in the sandbox and keep those that execute | Needs a runnable data context; keep only self-contained cells |
| **Free GPU small teacher** (optional): a 7B open-weight coder/instruct model (check its license is Apache-2.0) run with vLLM on free Kaggle GPUs (about 30 GPU-hours per week), Colab free tier, or Modal's monthly free credits (check the current amount) | $0 within free quotas | Paraphrased questions, domain Q&A drafts, richer insights, for about 20K to 30K examples (roughly 15M output tokens; about 6 to 10 free-GPU hours on a T4-class card) | Slower and weaker than a large teacher; every output still passes the execution and numeric checkers |
| **Self-improvement** (Phase 8 option): sample from the SFT model, execute, keep verified-correct outputs, fine-tune again | a few GPU minutes | Grows verified data without any outside teacher | Needs a decent first SFT model |

Expected trade-off (honest): the free-first path yields strong, verified SQL and Python behaviour and correct format/loop handling, but weaker natural-language variety and a shallower domain Q&A than a large paid teacher. Domain SME knowledge relies more on pretraining data (finance filings, supply-chain text) and on RAG in the app.

When to reconsider paid generation: only after Phase 9 evaluation shows a specific gap (e.g. low domain Q&A accuracy or poor paraphrase robustness). Then spend a small, targeted amount (about $20 to 60 for 10K to 20K examples of that gap) rather than a broad run. That decision stays ON HOLD until then.

### 1.3 Line items (Standard tier)

| Item | Estimate | Basis |
|---|---|---|
| Phase 1 measurement and smoke tests | $1 to 3 | 10K-file fetch test, 2K docs per slice |
| Code fetch from Software Heritage + clean | $0.5 to 2 | about 3B tokens kept, about 20 GB fetched; scaled from the reference run with 3 to 5x for code parsing |
| Other text slices (fineweb-edu, cosmopedia, Kaggle, SEC, finemath) | $0.3 to 1 | streaming, light cleaning |
| Dedup + decontamination | $0.2 to 0.5 | see 2.2: only where needed |
| Tokenizer + tokenize/pack | $0.2 to 0.5 | |
| **Pilot pretrain (0.3B tokens, 1 GPU)** | about $1.5 to 3 | catches pipeline bugs before the big run |
| **Full pretraining (8B tokens, 8xH100)** | $28 to 70 | 0.7 to 1.9 GPU-node hours x $31.6, +20% |
| SFT data (free-first path, CPU verification) | $0 to 10 | section 1.2; paid teacher ON HOLD |
| SFT training | $1 to 5 | ~100M tokens x 3 epochs, minutes |
| Evaluation | $2 to 8 | |
| Debugging and reruns | $5 to 15 | |
| Storage | about $0 | Modal Volume and public HF hosting; confirm current Modal volume pricing |
| **Contingency** | +25% | |

### 1.4 Cost controls (enforced in code and process)

1. Hard budget cap in `config.BUDGET_CAP_USD`; the pretraining script aborts and saves a checkpoint at the cap.
2. Spend gates: Phase 6 (pretrain) and Phase 7 (teacher at scale) require explicit approval after the trial run and the pilot batch respectively.
3. Pilot before scale: 1K teacher examples reviewed by hand before generating 100K; 0.3B-token pilot pretrain before the 8B run.
4. `modal billing report` after every phase, logged in `spend_log.md`.
5. Check workspace credits (Modal gives monthly free credits on its starter plan) and concurrency limits before launch.

## 2. Minimizing wall-clock time (dataset creation + pretraining)

### 2.1 Where the time goes in the naive plan

Naive sequence: fetch all -> clean -> dedup (MinHash across everything) -> train tokenizer on the whole corpus -> tokenize -> train. Each arrow is a full pass over the data on the Volume. The biggest delays are the code fetch (README reports about 6 hours on 16 cores for a full download), global MinHash, and tokenizer training on the full corpus.

### 2.2 Speed-ups (ranked by impact)

1. **Fetch only what you need, massively parallel, in the right region.** We need about 3B code tokens of 31B available (about 10%). Filter metadata (permissive license, size bounds, `int_score` 4 to 5) BEFORE fetching. Fan out 64 to 128 containers, many threads each, pinned to `us-east-1` (where the Software Heritage bucket lives: lower latency, no cross-region transfer). Estimated 10 to 20 minutes instead of hours. Over-fetch by 1.3x to cover missing files.
2. **Fuse the passes: fetch -> clean -> exact-dedup -> tokenize -> write `.bin` inside one worker per shard.** Removes three full read/write passes over the Volume. Needs the tokenizer first (next item).
3. **Train the tokenizer early on a small stratified sample (about 200 to 300M tokens), in parallel with the bulk fetch.** BPE vocab quality saturates well before the whole corpus; 32K merges from 1 GB of mixed text is enough. About 10 minutes on 16 cores instead of a full-corpus pass. Bulk shards then tokenize the moment they are cleaned.
4. **Skip global near-dup MinHash for sources that are already deduplicated** (Stack-Edu/StarCoder2 data, fineweb-edu, cosmopedia, Kaggle set). Do exact hash per shard and benchmark decontamination everywhere; run MinHash only on SEC/finance, supply-chain, and synthetic text. Decontamination is not optional and stays.
5. **Pilot-validated freeze.** A 0.3B-token, one-GPU pilot (about 20 to 30 minutes) validates tokenizer, packing, loss curve and generation before the big run, avoiding a discovered-late bug that costs a full rerun. It runs while the bulk tokenization finishes.
6. **One well-tuned single-node run.** 8xH100 on one node (no multi-node networking), `torch.compile`, bf16, SDPA/FlashAttention, fused AdamW, memmap loader, large micro-batch, no gradient checkpointing. The 200-step trial (about 3 to 5 minutes) fixes real tokens/sec and the final cost.
7. **Overlap everything that is independent.** SFT data generation (teacher + verification) runs from Phase 1 onward in parallel with Phases 2 to 6, so it adds no wall-clock time. Eval harness code is written while pretraining runs.
8. **Warm infrastructure.** Pin dependency versions, build the Modal image once and reuse it, keep decisions (teacher, license, mix) frozen after Phase 1 so nothing downstream reruns.
9. **Optional: start training on the first ready shards** (streaming data loader that waits for new `.bin` files) if the data phase stays longer than about 3 hours. Adds complexity, so default is off.

### 2.3 Resulting critical path (Standard tier, machine time after code is written)

| Step | Naive | Optimized |
|---|---|---|
| Measure + license ledger | 1 to 2 h | 1 to 2 h (parallel workers) |
| Tokenizer | 1 to 2 h on full corpus | about 10 to 20 min on sample, overlapped with fetch |
| Fetch + clean + dedup + tokenize | 10 to 20 h sequential | about 1.5 to 3 h (fused, 64 to 128 workers) |
| Pilot pretrain | not done | about 0.5 h (overlapped with last tokenization) |
| Pretraining 8B tokens | 0.7 to 1.9 h | 0.7 to 1.9 h (already one node; shrink only by cutting tokens) |
| **Total machine time to a base checkpoint** | about 14 to 26 h | **about 4 to 8 h** |
| SFT data (parallel) | adds 1 to 2 days if sequential | 0 added (overlapped) |

Calendar time is dominated by writing and testing the pipeline code (about 1 week), not machine time. Highest-value use of that week: tests first for the pure functions (cleaners, dedup, packing) so reruns are not needed.

### 2.4 Further time cuts, at a quality price

| Lever | Time saved | Cost to quality |
|---|---|---|
| Lean tier (4B tokens) | about half of pretrain and data time | weaker code/SQL; fine for a first end-to-end version |
| Drop finemath and extra web, shrink fineweb share | small | slightly weaker general prose |
| Cap per-slice epochs at 1 | none (already small) | avoid over-repeating thin domain data |
| Skip annealing stage | none (already inside budget) | loses disproportionate domain gains; not recommended |

Recommended path: build the whole pipeline on the Lean tier first (about $60 to 160 all-in, one working day of machine time), verify end to end, then rerun the pretraining stage at 8B tokens with the same tokenized shards extended, because the pipeline is already proven.

## 3. Decisions needed to lock the plan

1. Tier: Lean first then Standard (recommended), or Standard directly.
2. Teacher: ON HOLD. Free-first SFT data (section 1.2) is the plan; revisit a small targeted paid spend only after Phase 9 shows a gap.
3. Budget cap in dollars (suggest $150 total for Standard with contingency, and a pretraining-only cap of $90).
4. Confirm Modal workspace limits (container and GPU concurrency) and credit balance, since 100+ parallel CPU containers and 8 GPUs are assumed.

## 4. Risks to the numbers

| Risk | Effect | Mitigation |
|---|---|---|
| Real pretraining throughput below 1.2M tok/s | cost up to about 1.5x | 200-step trial, then re-decide tokens vs cap |
| Software Heritage rate limits | fetch time up | measure in Phase 1; reduce concurrency per worker, add workers; widen sample |
| Permissive yield below plan | less code data | shrink code slice or add verified synthetic code |
| Workspace GPU/CPU concurrency limit below assumption | data phase longer | request limit increase early; accept longer wall-clock |
| Free-first data gives weak domain Q&A or paraphrase robustness | quality gap | measure in Phase 9; then a small targeted paid top-up (about $20 to 60) |
| Free GPU quotas too slow or unavailable | SFT-data time up | programmatic and public data alone still cover SQL/Python loop; free teacher is optional |
| Price changes | any line item | recheck pricing before each spend gate |
