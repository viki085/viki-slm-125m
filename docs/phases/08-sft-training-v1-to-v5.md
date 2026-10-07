# Page 8: Fine-tuning, five rounds

**Runs on:** one local GPU (RTX 4060 Ti). **Cost:** $0. **Code:** `viki_slm_125m/sft/train.py`,
`sft_dataset.py`. **Output:** `artifacts/sft/` (v1) and `artifacts/sft_v2` to `sft_v5`.

## The method
Full fine-tuning (all weights) of `base-e1` on the mixed datasets of page 7.

| Setting | Value |
|---|---|
| Learning rate | 3e-5 with 3% warmup, constant, then cosine decay to 3e-6 over the last half |
| Epochs | 3 |
| Batch | Padded micro-batches of 2,048 tokens grouped by length, 32 accumulated, about 65,000 tokens per step |
| Optimiser | AdamW, no weight decay, gradient clip 1.0, bf16 mixed precision, no `torch.compile` |
| Loss | Cross-entropy on assistant tokens only |
| Model selection | Validation loss per source every 100 steps; `best.pt` is the checkpoint with the lowest mean |
| Speed | About 21,000 tokens per second, about 4.2 GB of GPU memory |

**Memory lesson:** the first smoke test with 8,192-token micro-batches peaked at 9.6 GB and spilled to system memory
(2.6K tokens per second). Micro-batches of 2,048 with 32 accumulation steps gave 4.2 GB and 21K tokens per second, a
roughly eight-fold speed-up for the same effective batch.

## Round by round
Each round was driven by the previous round's evaluation (page 9).

| Round | Examples, steps, time | What was changed | What the evaluation then showed |
|---|---|---|---|
| **v1** | 84,299, 1,120 steps, 54 min | First data: gretel, general, domain (11.4K), repair (7.9K) | gretel SQL 54.7%. On our new question shapes only 21.7% (three of four held-out templates at 0%). Wrote SELECT for "delete all employees" on a new schema; 0% refusal on unseen schemas |
| **v2** | 98,954, 1,277 steps, 60 min | Added 11,291 refusal / clarify / missing-data examples built on many different schemas; 15 more SQL templates | Refusal, clarification and missing-data answers on unseen schemas went from **0% to 100%**. SQL on the held-out templates stayed poor: right joins, wrong meaning (a year compared as `= '2024'`; `quantity` averaged instead of quantity x price; an invented `late_payment` column) |
| **v3** | 104,219, 1,424 steps, 65 min | 17 SQL templates for those gaps (derived values, `strftime` filters, `CASE` rates); 14 pandas templates; every pandas example lists all CSV files | Held-out templates 19.6% to **91.2%**; pandas **17.2% to 100%** (its code block now opens with the right tag). gretel flat at 56.2%. **Caveat:** the new templates teach the same patterns the held-out ones test, so 91.2% measures transfer to similar questions |
| **v4** | about 111K, 1,574 steps, 75 min | 28 plain-SQL templates (top-N by an existing column, filters, simple aggregates); repair examples rewritten from 7,850 to 11,000, including 2,530 whole-query rewrites | The three new held-out templates exposed that **"shortest" and "lowest" always came out as descending order** (every top-N template I wrote was descending), plus wrong-column choices. gretel still flat. Spider dev, measured at last, was **4.6% greedy / 12.3% voting**: the real weakness is multi-table schemas |
| **v5** | 123,655, 2,040 steps, about 100 min | Added the Spider training split (6,414 examples from 136 databases, twice per epoch) | Spider dev **4.6% to 22.1% greedy and 12.3% to 33.0% with voting**. gretel 58.0%, Python 100%, refusal 100% unchanged; the new held-out templates and finance wording were not helped |

## Choosing the v5 checkpoint
Validation loss on Spider flattened around 0.22 after step 1,000 and drifted up slightly (0.215 to 0.229); the other
sources kept improving, and the mean was lowest at the final step. The step-1,300 snapshot and the final checkpoint were
both scored on all of Spider dev greedy and **tied at 22.1%**. The final checkpoint (lowest mean validation loss) is the
delivered model. The validation set for Spider is only 10 databases, so its loss is noisy; the downstream score decided.

## Rejected idea: an agentic coding dataset
`AlicanKiraz0/Agentic-Chain-of-Thought-Coding-SFT-Dataset` was assessed: 429 examples averaging about 10,300 tokens
(none fit the 2,048 context), general software tasks (no pandas, 6 SQL mentions), and tool outputs that are placeholders
rather than real results. It was not used.

## What the sequence taught
1. Evaluate on something the training data cannot memorise, or the score is flattering (v3).
2. Fixing a measured gap with data works, and only for the shape of the gap that was fixed (v2, v3, v5).
3. A real external benchmark (Spider) changed the plan more than any internal test.
4. Fine-tuning is cheap enough on a local GPU that five rounds cost nothing but time.
