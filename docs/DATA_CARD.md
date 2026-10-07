# Data Card

Status: in progress. Updated after each data phase. Licenses come from `license_ledger.json`; measurements from `phase1-report.md`.

## Stage 2 (clean) results, after removing 8 unfinished workers

| Slice | Source | License | Est. clean tokens (chars/4 proxy) | Notes |
|---|---|---|---|---|
| python | HuggingFaceTB/stack-edu (Python), `permissive` files only | per-file (The Stack v2) | 2.34B | 7.5% syntax errors (mostly Python 2), 1.6% comment-heavy dropped |
| sql | HuggingFaceTB/stack-edu (SQL), `permissive` files only | per-file (The Stack v2) | 0.29B | 25% failed sqlglot parse; to be up-sampled 2 epochs |
| notebooks | HuggingFaceTB/issues-kaggle-notebooks (kaggle) | upstream Kaggle terms, to be recorded | 0.60B | 23 files with secrets dropped |
| finance-sec | PleIAs/SEC | cc0-1.0 | 0.72B | |
| fineweb-edu | HuggingFaceFW/fineweb-edu sample-10BT | odc-by | 1.76B | |
| cosmopedia | HuggingFaceTB/smollm-corpus cosmopedia-v2 | odc-by | 0.64B | |
| math | HuggingFaceTB/finemath (finemath-4plus) | odc-by | 0.33B | |
| **Total** | | | **6.68B** | Pending slices: supply-chain, datasci-docs |

Known gaps: supply-chain and data-science-docs slices not yet sourced. Token figures are proxies; real counts come from Phase 5.
PII handling: emails and public IPv4 addresses replaced with placeholders; files containing private keys or high-confidence API tokens are dropped.
Decontamination and deduplication counts: to be added in Phase 3.

## Stage 3 (dedup + decontamination) results

Exact duplicates removed globally (5.36M docs hashed): 16,167. MinHash near-duplicates (SEC only): 1,958.
Benchmark decontamination: a document is removed when it shares at least 10 consecutive 13-word windows (about 22 words) with a HumanEval, MBPP, Spider (validation), BIRD (dev) or DS-1000 problem or solution. A first version flagged any single shared window and wrongly removed 21% of notebooks for shared import boilerplate; this was fixed and verified on samples.

| Slice | Docs kept | Est. tokens | Exact dups | Near dups | Contaminated removed |
|---|---|---|---|---|---|
| python | 2,378,735 | 2.34B | 1,286 | 0 | 350 (mbpp 210, ds1000 109, humaneval 30, spider 1) |
| sql | 261,613 | 0.29B | 1,267 | 0 | 0 |
| notebooks | 273,143 | 0.60B | 2 | 0 | 49 (ds1000 36, mbpp 11, humaneval 2) |
| finance-sec | 27,040 | 0.66B | 988 | 970 | 0 |
| fineweb-edu | 1,471,632 | 1.75B | 10,734 | 0 | 0 |
| cosmopedia | 680,519 | 0.64B | 1,890 | 0 | 4 |
| math | 253,049 | 0.33B | 0 | 0 | 11 |
| **Total** | | **6.61B** | | | |

Limitation: contamination matching needs at least 13 words, so very short benchmark items (e.g. one-line questions) are not detectable this way.

## Stage 5 (tokenized, packed) results

Tokenizer: 32,768 byte-level BPE (phase4_tokenizer_report.json). Windows of 2,048 uint16 tokens, EOS after every document, every 100th window to validation (per-file, so every slice has its own validation set). Window-level split: a document spanning a train/val boundary leaks slightly; accepted trade-off.

| Slice | Train tokens (real) | Val tokens | Epochs | Share of effective mix |
|---|---|---|---|---|
| python | 2.671B | 27.0M | 1 | 35.6% |
| sql | 0.461B | 4.7M | 2 | 12.3% |
| notebooks | 0.766B | 7.8M | 1 | 10.2% |
| fineweb-edu | 1.601B | 16.2M | 1 | 21.4% |
| cosmopedia | 0.514B | 5.2M | 1 | 6.9% |
| finance-sec | 0.601B | 6.1M | 1 | 8.0% |
| math | 0.420B | 4.3M | 1 | 5.6% |
| **Total** | **7.033B** | **71.2M** | | **7.49B effective** |

Real counts are about 6% above the chars/4 proxy overall (code is token-heavy, web text slightly lighter). Not yet included: supply-chain and data-science-docs slices.
