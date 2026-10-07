# Page 1: Feasibility, measurements and licences

**Runs on:** Modal, CPU. **Cost:** a few cents to well under a dollar (part of the $11.37 data-processing total).
**Code:** `viki_slm_125m/data/measure.py`, `viki_slm_125m/data/ledger.py`, `modal_app.py::stage1`.
**Report:** `docs/phase1-report.md`, `reports/phase1_measurements.json`, `reports/license_ledger.json`.

## Why this phase exists
Before downloading billions of tokens, find out whether the plan is possible: is there enough permissively licensed
data, can the code files be fetched fast enough, and what licences apply?

## What was done
1. **Chose the sources** for each corpus slice and verified the dataset ids against the Hugging Face API.
2. **Measured the text sources:** sampled 2,000 documents from each and estimated the usable tokens.
3. **Measured Stack-Edu** (a filtered GitHub code set): what share of files carry a permissive licence.
4. **Tested the Software Heritage fetch.** Stack-Edu lists files but not their text; the text must be fetched from the
   Software Heritage archive. Tested at 64 threads in one container.
5. **Built a licence ledger:** the declared licence of every candidate dataset, so the release can be decided on facts.

## Results
**Software Heritage fetch:** anonymous access works. 10,000 Python files in about 21 seconds (477 files per second),
10,000 SQL files at 454 per second, no failures. Fetching 3.9M Python files would take about 130 seconds on 64
containers: not a bottleneck.

**Permissive share of Stack-Edu (300K-row sample):**

| Config | Permissive files | Estimated usable tokens |
|---|---|---|
| Python | 18.0% (about 4.6M files) | about 5.5B (budget 2.4B: feasible) |
| SQL | 15.9% (about 399K files) | **about 0.4B (budget was 0.8B: short)** |

**Text sources (raw upper bound, before cleaning):** Kaggle notebooks 1.26B, FineWeb-edu 12.0B, Cosmopedia 36.6B,
FineMath 8.4B, SEC filings 1.24B. All comfortably above their budgets.

**Licences:** clean for an open release: SEC (CC0), FineWeb-edu / Cosmopedia / FineMath (ODC-By),
gretel synthetic SQL (Apache-2.0), smoltalk (Apache-2.0). **Needs a decision:** Spider is CC BY-SA 4.0 (share-alike);
Stack-Edu and the Kaggle notebooks declare no single licence (Stack-Edu follows each file's own licence, which is why
only the "permissive" files were kept).

## Decisions made from the evidence
* **SQL is scarce.** Only about 0.4B permissive SQL tokens exist. Decision: use all of it and **see it twice**
  ("2 epochs"), giving about 0.8B seen. The mix keeps SQL at 12.3% of what the model sees.
* Keep only permissive Python and SQL files.
* The two slices with no public source (supply-chain text, data-science documentation) stayed "pending" and were never
  filled. This is the origin of the later weakness on supply-chain and concept questions.

## Problems met here
| Problem | Fix |
|---|---|
| Modal rejected the region name `us-east-1` | Use `us-east` |
| Git Bash rewrote `/raw_meta` volume paths into `C:/...` | Set `MSYS_NO_PATHCONV=1` and delete the stray entry |
| An early estimate assumed token share equals file share, overstating SQL | Replaced by measuring bytes and tokens per file |

## Exit gate
Data budgets confirmed or adjusted, licence ledger written, fetch speed proven. Next: fetch and clean (page 2).
