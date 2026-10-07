# Page 3: Deduplication and benchmark decontamination

**Runs on:** Modal, CPU. **Code:** `viki_slm_125m/data/dedup.py`, `modal_app.py::dedup`.
**Output:** `corpus/<slice>/*.jsonl`. **Report:** `reports/phase3_dedup_report.json`.

## Why
* Duplicated text wastes training and encourages memorisation.
* If benchmark problems (HumanEval, MBPP, Spider, BIRD, DS-1000) leak into pretraining, the later scores are inflated and
  cannot be trusted. Evaluation later in the project depends on this step being right.

## What was done
1. **Exact duplicates:** every document is hashed across the whole corpus (5.36M documents) and repeats are dropped.
2. **Near-duplicates:** MinHash with locality-sensitive hashing (5-word shingles, similarity threshold 0.8), applied only
   to the SEC filings, where boilerplate repeats heavily with small changes.
3. **Decontamination:** the benchmark problems and solutions (HumanEval, MBPP, Spider validation, BIRD dev, DS-1000)
   were turned into sets of 13-word windows. A training document was removed if it shared **at least 10 consecutive
   windows** (about 22 words) with a benchmark item.
4. A second pass wrote the surviving documents into `corpus/`.

## Results
| Slice | Documents kept | Tokens (estimate) | Exact duplicates | Contaminated removed |
|---|---|---|---|---|
| Python | 2,378,735 | 2.34B | 1,286 | 350 (MBPP 210, DS-1000 109, HumanEval 30, Spider 1) |
| SQL | 261,613 | 0.29B | 1,267 | 0 |
| Notebooks | 273,143 | 0.60B | 2 | 49 (DS-1000 36, MBPP 11, HumanEval 2) |
| SEC filings | 27,040 | 0.66B | 988 | 0 |
| FineWeb-edu | 1,471,632 | 1.75B | 10,734 | 0 |
| Cosmopedia | 680,519 | 0.64B | 1,890 | 4 |
| Math | 253,049 | 0.33B | 0 | 11 |
| **Total** | | **6.61B** | 16,167 | **414** |

## The bug that mattered
The first decontamination rule removed a document if it shared **any single** 13-word window with a benchmark item.
That wrongly deleted **21% of the notebooks**, because common boilerplate (`import pandas as pd` lines, standard plotting
calls) appears in benchmark solutions too.

Fix: require 10 consecutive matching windows, plus an "informative n-gram" test that ignores windows made only of
imports and common calls. The write step was re-run and the counts above come from the fixed version.

## Limits to remember
* A window needs 13 words, so very short benchmark items (one-line questions) cannot be detected this way.
* This decontaminates pretraining only. The **fine-tuning** data used Spider's *training* split on purpose (page 7),
  and the Spider *dev* databases were excluded from it so the benchmark stayed clean.

## Hand-off
A corpus of about 6.6B tokens, free of duplicates and benchmark text, goes to tokenizer training (page 4).
