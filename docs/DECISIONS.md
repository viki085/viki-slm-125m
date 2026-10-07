# Decisions

| Decision | Status | Value / note |
|---|---|---|
| Model | decided | 124.3M params, vocab 32,768, 12L/768d/12h MHA, context 2,048 |
| Tier | proposed | Lean (4B) first, then Standard (8B); confirm before Phase 6 |
| Teacher LLM | ON HOLD | free-first SFT data (cost-and-time-plan.md 1.2) |
| Budget cap | proposed | $150 total, $90 pretraining (config.py); confirm |
| Release license | open | decide after the license ledger (Phase 1) |
| Primary SQL dialect | proposed | SQLite/DuckDB (easy execution checks) |
| GQA vs MHA | decided | MHA |

## Pretraining epoch 1 (done)
- Run `base-e1`: 14,300 steps, 7.5B effective tokens, 8x H100, 44.6 min, node cost $23.69, no spikes, no budget stop.
- Final validation loss: sql 0.701, notebooks 1.055, python 1.291, finance-sec 1.654, math 2.012, cosmopedia 2.080, fineweb-edu 3.124.
- Checkpoints on the Volume: `checkpoints/base-e1/ckpt.pt` (final), `ckpt.pt.predecay` (step 12,870, for continuing).
- Smoke test: Python/SQL fluent and structurally right; finance text plausible; supply-chain text wrong (no supply-chain data in the corpus).
- GPU spend to date: about $26.2 by node-time estimate (pilot 0.91, 8-GPU trial 1.58, epoch 1 23.69). The actual Modal bill is higher and also includes data processing: see "Actual Modal bill" below.

## SFT v1 (done, local RTX 4060 Ti)
- 84,299 examples (gretel 40,000; general 25,000; domain 11,449; repair 7,850), 3 epochs, 1,120 steps, 54 min, $0.
- Final val loss (trained tokens): gretel 0.139, repair 0.010, domain 0.013, general 1.092. Best checkpoint: step 1,100 (`artifacts/sft/best.pt`).
- Eval (greedy, SQL run against real databases):
  - gretel unseen domains (600): execution accuracy 54.7%, SQL runs 92.7%, format 99.3%, insight faithful 98.7%.
  - domain unseen templates (240): execution accuracy 21.7% (fin_balance_country 87%, the other three templates 0%).
  - refusal / missing-info / clarification: 100% on in-distribution templates, but fails on new schemas (custom prompts): "Delete all employees..." became a SELECT, "best employee" was not clarified.
- Known gaps: new SQL constructs (strftime, 3-table joins with CASE, HAVING), refusal/clarify generalization.

## SFT v2 (done, local RTX 4060 Ti)
- Changes vs v1: diverse-schema refusal/clarify/missing-data data (11,291 examples from gretel schemas, 5,875 write-request refusals) and 15 extra SQL templates (domain SQL 7,200 -> 10,646; 4 eval templates still held out).
- 98,954 examples (gretel 40,000; general 25,000; domain 14,813; refusal 11,291; repair 7,850), 3 epochs, 1,277 steps, ~60 min, $0. Checkpoint: `artifacts/sft_v2/best.pt`. Report: `eval_sft_v2_report.json`.
- Eval (v1 -> v2):
  - gretel unseen domains (600): execution accuracy 54.7% -> 57.0%; SQL runs 93.7%; insight faithful 98.9%.
  - domain unseen templates (240): 21.7% -> 19.6%; SQL runs only 57.1%.
  - unseen-schema behaviour (100 per kind, domains never trained on): refusal 0% -> 100%, missing-data 0% -> 100%, clarification 0% -> 100%; wrote SQL instead 99-100% -> 0%. These checks test shape and wording, not answer quality.
- Per-template failures (held-out templates, all 0%): the model gets joins and grouping right but misses semantics.
  - sc_monthly_orders: filters `order_date = '2024'` instead of `strftime('%Y', order_date) = '2024'`.
  - sc_avg_po_value_country: averages `quantity` instead of the derived `quantity * unit_price`.
  - fin_late_rate_segment: invents a column `late_payment`; the rate needs `paid_date > due_date` via a 3-table join.
- Takeaway: refusal generalization is fixed; derived metrics and date-function filters are the remaining weakness. Decision on a targeted v3: pending.

## SFT v3 (done, local RTX 4060 Ti)
- Changes vs v2: 17 new SQL templates (derived values, strftime year/month filters, CASE rates), 14 new pandas templates (Python examples 3,000 -> 5,000), and every Python example now lists all of the domain's CSV files (as the playground does). Domain SQL 14,000, domain train 20,078. Chat/general data unchanged.
- 104,219 examples, 3 epochs, 1,424 steps, ~65 min, $0. Checkpoint `artifacts/sft_v3/best.pt`. Reports: `eval_sft_v3_report.json`, `eval_python_v3_report.json`.
- Eval (v1 -> v2 -> v3):
  - domain unseen templates (240): 21.7% -> 19.6% -> **91.2%**. Per template: monthly orders, avg PO value by country and late rate by segment 0% -> 100%; balance by country 87% -> 78% -> 65%.
  - gretel unseen domains (600): 54.7% -> 57.0% -> 56.2% (flat; SQL runs 92.2%, insight faithful 98.9%).
  - unseen-schema refusal / missing / clarify: 100% / 100% / 100% (unchanged from v2).
  - Python (pandas, unseen seeds, seen templates, 58 items): v2 17.2% (runs 36.2%, correct code tag 0%) -> v3 **100%** (runs 100%, correct tag 100%).
- Caveats: the 91.2% is not a clean unseen test. The new templates teach the same semantic patterns as the held-out ones (year filter, derived value, CASE rate), so it measures pattern transfer to similar questions. The Python 100% is in-distribution (templates seen). The flat gretel number is the better guide to open-ended SQL. balance_by_country regressed.

## SFT v4 (done, local RTX 4060 Ti)
- Changes vs v3: 25 training + 3 held-out plain-SQL templates (top-N by an existing column, simple WHERE filters, AVG/SUM/COUNT/MAX of existing columns); domain SQL 18,000; repair 11,000 (8,000 gretel + 3,000 domain) including 2,530 "rewrite" traces where the whole select list is wrong and the fix is a full rewrite from the schema. New builder order: sql, general, domain, repair (reads domain_train), refusal.
- 111K examples, 3 epochs, 1,574 steps, ~75 min, $0. Checkpoint `artifacts/sft_v4/best.pt`. Reports: `reports/eval_sft_v4_report.json`, `reports/eval_python_v4_report.json`. v3 data snapshot: `data/sft_v3/`.
- Eval (v3 -> v4):
  - original 4 held-out templates: 100/100/100/65 -> 100/100/100/67 (unchanged).
  - gretel unseen domains (600): 56.2% -> 55.7% (flat). Python 100% -> 100%. Refusal/missing/clarify 100% -> 100%.
  - 3 NEW held-out plain-SQL templates (60 each): shortest lead time 15%, lowest loan rate 0%, average warehouse capacity 58%.
- What the new held-out misses show:
  - ASC vs DESC: "shortest lead time" always came out `ORDER BY ... DESC`. Every top-N template I wrote was DESC, so the model never saw ascending order. Real gap in the data.
  - "lowest interest rate" picked the wrong column (principal) or an invented aggregate.
  - avg capacity: logic was right, mismatch is `ROUND(...,2)` vs gold `ROUND(...,1)`; the strict judge counts it wrong. Real accuracy is higher than 58%.
- Playground, v4: "Which 5 suppliers have the highest average rating?" is fixed (that exact question is in training). "Highest reorder point" returned the unit_cost column; "total amount of all loans" invented a column `amount` (should be principal) and self-repair repeated the same query. Plain-select answers still follow the nearest trained template rather than the wording.
- Next candidates: ASC templates ("lowest/shortest/smallest/oldest") and many more column-selection paraphrases; make the judge treat rounding-only differences as a match.

## Inference-only improvements and external baseline (on SFT v4)
- Rounding-tolerant judge added (`match_rounding`; reported next to the strict score, strict numbers unchanged). Execution voting added (`viki_slm_125m/sft/sql_vote.py`: greedy + 8 samples at T=0.7, run all, majority result wins, ties go to greedy); used by the playground, repair loop is the fallback when no candidate runs. Token-level schema-constrained decoding was NOT built (needs a grammar engine over BPE tokens); execution filtering removes invented columns instead.
- Greedy -> vote(8): gretel unseen domains 55.7% -> 59.3% (runs 92.8% -> 98.2%); held-out domain templates 62.9% -> 67.1% strict, 68.1% -> 72.4% tolerant (runs 87.9% -> 99.8%). Consistent mistakes are not fixed by voting (e.g. "shortest lead time" got DESC from 9/9 candidates).
- Spider dev (1,034 questions, 20 databases, real data; mirror HAL-9001/spider-databases, cc-by-sa-4.0, evaluation only): greedy 4.6%, vote(4) 10.1%, vote(8) 12.3% execution accuracy (queries run: 26% -> 67%). No question was too long for the context window.
- Why Spider is so low (200-question greedy sample): 55% reference a column that does not exist in the chosen table, 7% a nonexistent table, 20% give no parsable SQL (mostly the model drops the opening `<|sql|>` tag after the think block, some refuse), 9% run but are wrong, 2% correct. Pattern: with 4-7 tables, quoted identifiers and foreign keys the model picks a wrong table (often `singer_in_concert`) and invents columns. SFT SQL data is mostly 1-3 table gretel schemas plus two template domains; Spider-style multi-table schemas are out of distribution.
- Decision pending: Spider train (cc-by-sa-4.0, share-alike) is the most direct fix; b-mc2/sql-create-context is also Spider/WikiSQL-derived. Licence implication for released weights needs a decision.

## SFT v5: Spider train added (done, local RTX 4060 Ti)
- Changes vs v4: Spider train split (cc-by-sa-4.0; 8,659 records -> 6,414 verified training traces from 136 databases + 459 validation traces from 10 whole held-out databases; the 20 Spider dev databases excluded) added to the mix, repeated 2x per epoch. Builder: `viki_slm_125m/sft/builders/build_spider_sft.py`, helpers `viki_slm_125m/sft/spider_source.py`. Data: `data/external/spider_raw` (HF mirror HAL-9001/spider-databases, evaluation + training use).
- ~124K training examples incl. repeats, 3 epochs, 2,040 steps, ~100 min, $0. Checkpoint `artifacts/sft_v5/best.pt` (final step, lowest mean val loss 0.2374); snapshot `best_step1300.pt` scored identically on Spider dev greedy (22.1%). Spider val loss flattened at ~0.22 after step 1,000 (10 databases, noisy).
- Eval (v4 -> v5):
  - Spider dev (1,034 q, real databases): greedy 4.6% -> 22.1%; vote(4) 10.1% -> 30.2%; vote(8) 12.3% -> **33.0%**. Queries that run: greedy 26% -> 38%; vote(8) 67% -> 67% (the gain is table/column choice among runnable queries).
  - gretel unseen domains: greedy 55.7% -> 57.3%; vote(8) 59.3% -> 60.2%; runs 98.2%.
  - held-out domain templates (420): greedy 62.9% -> 61.4% strict (68.1% -> 66.7% tolerant). Original four: 100/100/100 unchanged, fin_balance_country 67% -> 57%. New plain-SQL held-outs unchanged: shortest lead time 15%, lowest loan rate 0%, avg capacity 58% strict / 95% tolerant.
  - Voting no longer helps on the domain templates (v4: 62.9% -> 67.1%; v5: 61.4% -> 61.4%). Checked: samples differ from greedy (32 of 60 items) but the failures are consistent across samples, not random.
  - Python (pandas) 100%; refusal/missing/clarify 100%; insight faithfulness 99.6%.
- Not addressed by v5: ASC ordering, "lowest/shortest", wrong-column choice on finance/supply-chain wording, chat/domain knowledge.
- The dataset AlicanKiraz0/Agentic-Chain-of-Thought-Coding-SFT-Dataset was assessed and rejected: 429 examples, median ~10.3K tokens each (none fit the 2,048 context), general software engineering (0 pandas/dataframe prompts, 6 SQL), tool outputs are placeholders, not verified execution.

## Project restructure for release (after SFT v5)
- Package renamed `viki` -> `viki_slm_125m` (hyphens cannot appear in import names; the project name is `viki-slm-125m` in `pyproject.toml`). Model code consolidated in `viki_slm_125m/model/` (architecture, checkpoint, generation, `VikiSLM`, Hugging Face export). Exported model: `models/viki-slm-125m/` (bf16 safetensors, 30/30 identical greedy outputs vs the project pipeline).
- Found while exporting: `load_model` cast the whole model to bf16, which also rounded the rotary `inv_freq` buffer (training kept it fp32). Fixed (`inv_freq` stays fp32). Effect is small: mean logit error to fp32 0.0418 -> 0.0402 on 350-token Spider prompts. Re-check of v5 greedy: domain templates 61.4% (unchanged), gretel 57.3% -> 58.0%. Earlier reports used the old loading path; differences are within noise.
- Model described as a "Python and SQL coding model" (README, page, model card, guide title). Training system prompts ("You are a data analyst assistant ...") were left unchanged on purpose: they are part of the fine-tuning data and changing them without retraining would move the model off its training prompt. `docs/idea.md` (original brief) left as written.


## Actual Modal bill (read from `modal billing report`, saved in `reports/modal_billing_raw.json`)
- 17 cloud apps, $40.83 in total, all billed on 2026-10-06 (UTC). Oct 7 is not in the report (end date exclusive).
- Data processing (CPU-only apps: feasibility, fetch and clean, dedup, tokenizer, tokenise): **$11.37** (the cleaning fan-out is about $7.68 of it).
- GPU apps: epoch-1 pretraining $25.86 (H100 $23.93, container CPU $1.15, memory $0.78); 8 x H100 trial $1.96; 1-GPU pilot $1.62; smoke tests $0.03. GPU apps total $29.46.
- Base model including data processing: **$40.83**. The earlier estimates ($23.69 for epoch 1, about $26 for all GPU work) counted only H100 node-time at the assumed $3.95 per GPU-hour and left out container CPU and memory, and all data processing.
- Mapping of the three smaller GPU apps to pilot / trial / smoke is inferred from their size; the totals do not depend on it.
