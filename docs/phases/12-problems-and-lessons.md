# Page 12: Problems met and lessons

A compact record of what went wrong, how it was found and fixed, and what to do differently. Details are on the page
of the phase where each problem happened.

## Problems, fixes and how they were found
| Phase | Problem | How found | Fix |
|---|---|---|---|
| 1 | SQL tokens overestimated (token share assumed equal to file share) | Measured bytes and tokens per file | SQL seen twice; budgets from measurements |
| 1 | Modal rejected `us-east-1` | Run failed | Use `us-east` |
| 1 to 6 | Git Bash rewrote `/raw_meta` volume paths into Windows paths | A stray `C:` entry appeared | `MSYS_NO_PATHCONV=1` |
| 2 | "Comment-heavy" rule dropped converted notebooks | Pilot on 400 rows | Exempt `# In[ ]` files |
| 2 | First full cleaning run stopped from the dashboard | Missing shards | Deleted 8 partial files; accepted the shortfall |
| 3 | Decontamination removed 21% of notebooks for shared boilerplate | Spot-checking removed files | Require 10 consecutive 13-word windows plus an informative-n-gram test |
| 6 | GPU image missing a module | Pilot crashed on import | Add to image sources |
| 6 | `transformers` import failed under the meta device | Pilot error | Import at module top |
| 7 | Generated code lost its newline escapes | Syntax errors in files | Write files with the editor tool, not shell heredocs |
| 7 | Only 300 domain SQL examples after the repeat cap | Dataset report | Paraphrases, more templates |
| 8 | First fine-tuning run spilled GPU memory | 2.6K tokens per second | 2,048-token micro-batches with accumulation (21K tokens per second) |
| 9 | Held-out templates scored 0% for v1 and v2 | Per-template breakdown | Added the missing SQL patterns (v3), then measured with a harder set (v4) |
| 9 | 91.2% overstated generalisation | Writing newer, different held-out templates | Reported the old and new sets separately |
| 9 | Spider (an outside test) showed 4.6% where internal tests showed 55% to 91% | First real external run | Added Spider training data (v5) |
| 10 | Self-repair repeated the same wrong query | Playground testing | Voting with sampling; repair as a fallback |
| 10 | Chat answers looped | A pasted output | Repetition penalty and n-gram ban |
| 11 | Exported model's logits differed from the evaluated model | Export parity check | Cast to bf16 had rounded the rotary frequencies; fixed |

## Lessons
1. **Measure on data the model cannot memorise.** Internal tests built from the same templates as training flatter the
   model. Use an outside benchmark early (Spider should have been run after v1, not after v4).
2. **Fix gaps with data in the shape of the gap, then test a different shape.** Every round improved exactly what it
   targeted; the next test found the next gap.
3. **Execution is the cheapest and most honest check.** It filtered training data, scored the model, and now powers
   voting and repair.
4. **Know what the model is for before claiming it.** The brief said "expert data scientist"; the evidence supports
   "Python and SQL coding model". Concept questions were never trained or tested.
5. **Pretraining decides knowledge; fine-tuning decides behaviour.** Supply-chain and data-science text were never in the
   corpus, so no amount of fine-tuning produced domain explanations.
6. **Record cost from the bill, not from estimates.** The pretraining estimate of $23.69 became $40.83 once data
   processing and container CPU were counted.
7. **Cheap loops win.** Local fine-tuning at 55 to 100 minutes per round allowed five rounds for no cost.

## If the work continues (in order of expected value)
1. More real multi-table text-to-SQL data (BIRD, sql-create-context) and ascending-order and column-choice examples;
   re-run Spider.
2. Run HumanEval and DS-1000 so general Python skill has a number.
3. A data-science concept test set and, if wanted, retrieval from a curated reference so the model answers from supplied text.
4. More pretraining on a corpus that adds supply-chain and data-science text, resuming from `ckpt.pt.predecay`
   (about $25 per extra pass).
