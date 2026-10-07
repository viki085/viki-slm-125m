# Phase 1 Report: Feasibility, Licenses, Measurements

Raw data: `license_ledger.json`, `phase1_measurements.json` (also on the Volume under `/raw_meta/`). Run cost: a few cents of CPU (confirm with `modal billing report`).

## Software Heritage fetch (Modal, region `us-east`, 64 threads, one 4-CPU container)
| Config | Files fetched | Failures | Throughput | Avg file |
|---|---|---|---|---|
| Python | 10,000 | 0 | 477 files/s | 2,020 chars |
| SQL | 10,000 | 0 | 454 files/s | 3,247 chars |

Anonymous (unsigned) S3 access works. Projected: 3.9M Python files (about 2.4B tokens) in about 130 seconds on 64 containers. The fetch is NOT a bottleneck. Note: `region="us-east-1"` is rejected by Modal; `"us-east"` works (region selection may carry a price multiplier; check billing).

## Stack-Edu permissive share (300K-row shuffled sample per config)
| Config | Permissive files | Permissive bytes | Permissive and score >= 4 |
|---|---|---|---|
| Python | 18.0% (about 4.6M files) | 26.6% | about 2.1% of all files (about 0.53M files) |
| SQL | 15.9% (about 399K files) | 20.6% | 4.5% of all files |

Estimated yields (code assumed 3.3 chars/token; a real tokenizer is not available until Phase 4):
* Python: about 5.5B tokens of permissive files in total, so the 2.4B budget is feasible. Prefer score >= 4, then fill with score 3.
* SQL: about 0.4B tokens of permissive files after a 100 KB size cap. This is BELOW the 0.8B budget; my earlier ~1.5B estimate wrongly assumed token share equals file share (permissive files are smaller and many `no_license` files are huge dumps).

## Text sources (2,000 docs sampled; raw upper bounds before cleaning)
| Source | Avg chars/doc | Raw tokens available | Planned budget | Verdict |
|---|---|---|---|---|
| notebooks (Kaggle) | 8,680 | 1.26B | 0.64B | OK |
| fineweb-edu sample-10BT | 4,976 | 12.0B | 1.76B | OK |
| cosmopedia-v2 | 3,741 | 36.6B | 0.64B | OK |
| finemath-4plus | 5,023 | 8.4B | 0.40B | OK |
| SEC (PleIAs, CC0) | 101,885 | 1.24B | 0.72B | OK (could raise budget; cleaning will cut about 10%) |

## Licenses (from the ledger)
Clean for an open release: PleIAs/SEC (cc0-1.0), fineweb-edu / smollm-corpus / finemath (odc-by), smol-smoltalk (apache-2.0), gretelai/synthetic_text_to_sql (apache-2.0), Finance-Instruct-500k (apache-2.0), finance-alpaca (mit), sql-create-context (cc-by-4.0), humaneval (mit).
Needs a decision: `xlangai/spider` is cc-by-sa-4.0 (share-alike; used only for training if we accept that, never its dev/test); `stack-edu` and `issues-kaggle-notebooks` declare no license (stack-edu follows The Stack v2 per-file licenses, so we filter `permissive`; Kaggle upstream terms to be recorded). `Salesforce/xlam-function-calling-60k` is gated (auto-approval, needs the HF token). All `[VERIFY]` ids exist.

## Open issue before Phase 2
SQL pretraining slice will be about 0.4B tokens, half the plan. Options: (a) accept a smaller SQL share (about 5% instead of 10%), (b) up-sample SQL 2 epochs (about 0.8B seen), (c) (a) plus rely on programmatic SQL text (Phase 7 data also used as pretraining text). Recommended: (b), since SQL is a core skill and 2 epochs over clean data is safe.
