# Page 2: Fetch and clean

**Runs on:** Modal, CPU, 162 workers in parallel across the seven slices (the largest data-processing cost, about $7.68 of the $11.37).
**Code:** `viki_slm_125m/data/cleaning.py`, `viki_slm_125m/data/pipeline.py`, `modal_app.py::clean_pilot` and `::clean`.
**Output:** `clean/<slice>/*.jsonl` on the Modal volume. **Report:** `reports/phase2_clean_report.json`.

## What the phase does
For each of the seven slices: stream the records, fetch the file text where needed, apply a cleaning chain chosen for
that kind of data, and write the survivors. Each worker handles one shard, so the whole job takes minutes.

```
dataset rows ─► [code only: fetch file text from Software Heritage] ─► cleaning chain ─► kept / dropped (with reason)
```

## Cleaning chains (deterministic rules, `cleaning.py`)
| Kind | Slices | Rules |
|---|---|---|
| Prose ("text") | FineWeb-edu, SEC | Language filter (English), minimum length 600 characters, line-level filters (short lines, lines that are mostly symbols), repetition filter (a document is dropped if its most common n-grams cover over 50%) |
| Light prose | Cosmopedia, math | Same without line filtering, so equations and textbook structure survive |
| Python | Python | Must parse with `ast` (drops Python 2); 200 to 100,000 characters; average line under 100 characters; at least 25% alphanumeric; comments under 50% of lines |
| SQL | SQL | Must parse with `sqlglot`; very long `INSERT` data blocks truncated (limit 50 rows) |
| Notebooks | Kaggle | Python parse and comment rules, with an exemption for converted notebooks (`# In[ ]` markers); files with secrets dropped |
| Everyone | all | Emails and public IPv4 addresses replaced with placeholders; files with private keys or high-confidence API tokens dropped |

## Results
| Slice | Streamed | Kept | Clean tokens (characters / 4) |
|---|---|---|---|
| Python | 2,626,870 | 2,380,371 | 2.34B |
| SQL | 354,802 | 262,880 | 0.29B |
| Notebooks | 273,578 | 273,194 | 0.60B |
| SEC filings | 29,355 | 28,998 | 0.72B |
| FineWeb-edu | 1,533,573 | 1,482,366 | 1.76B |
| Cosmopedia | 683,488 | 682,413 | 0.64B |
| Math | 270,739 | 253,060 | 0.33B |
| **Total** | | | **6.68B** |

Main drop reasons: 7.5% of Python failed to parse (mostly Python 2), 1.6% were comment-heavy; 25% of SQL files did not
parse with `sqlglot`; 23 notebook files contained secrets.

## Problems met and how they were fixed
| Problem | Fix |
|---|---|
| The "comment-heavy" rule dropped converted notebooks, because their `# In[ ]` markers look like comments | Exempt files that carry those markers |
| A clean pilot (`clean_pilot`, 400 rows per slice) was run first to catch rule bugs before spending on the full run | Rules corrected, then the full run |
| **The first full run was stopped from the Modal dashboard** before every worker finished | 8 partially written files were deleted; you chose to **accept the shortfall** and **keep the strict filter** rather than re-run |
| `sqlglot` raised unexpected exceptions for some dialect-specific SQL | Catch the exception class broadly and count the file as unparseable |
| Windows console could not print some characters (`charmap` error) | `PYTHONIOENCODING=utf-8` for all scripts |

## Known gaps
The supply-chain and data-science-docs slices were never added. Token counts here are a characters-divided-by-four
estimate; real counts came in about 6% higher once the tokenizer existed (page 5).

## Hand-off
Seven cleaned slices of about 6.7B estimated tokens go to deduplication (page 3).
