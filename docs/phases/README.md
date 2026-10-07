# Phase-by-phase guide

This folder walks through the whole build, one page per phase, in the order the work was done. Each page says what the
phase was for, what was done, the numbers that came out, what went wrong and how it was fixed, and what the next phase
received.

| Page | Phase | Where it ran | Output handed on |
|---|---|---|---|
| [00](00-overview-and-decisions.md) | Goals, constraints and decisions | local | the plan and its budget |
| [01](01-feasibility-and-licences.md) | Feasibility, measurements, licences | cloud CPU | data budgets, licence ledger |
| [02](02-fetch-and-clean.md) | Fetch and clean | cloud CPU | 7 clean slices, about 6.7B tokens |
| [03](03-dedup-and-decontamination.md) | Dedup and benchmark decontamination | cloud CPU | corpus of 5.2M documents, 6.6B tokens |
| [04](04-tokenizer.md) | Tokenizer | cloud CPU | 32,768-token BPE tokenizer |
| [05](05-tokenize-and-pack.md) | Tokenize and pack | cloud CPU | 7.03B train tokens, 71M validation tokens |
| [06](06-pretraining.md) | Pretraining | cloud 8 x H100 | base model `base-e1` |
| [07](07-sft-data.md) | Fine-tuning data | local CPU | six verified datasets and test sets |
| [08](08-sft-training-v1-to-v5.md) | Fine-tuning, five rounds | local GPU | SFT v5 |
| [09](09-evaluation.md) | Evaluation | local GPU | measured results and their caveats |
| [10](10-agent-loop-and-playground.md) | Agent loop and playground | local | the runnable system |
| [11](11-release-and-delivery.md) | Export and delivery | local | `models/viki-slm-125m/`, docs, tests |
| [12](12-problems-and-lessons.md) | Problems met and lessons | all | a checklist for the next round |

## The build in one picture

```
   corpus sources                                              your own generators + public SQL data
 (code, notebooks, SEC, web, math)                              (databases, templates, Spider, gretel)
        │                                                                   │
  01 measure ─► 02 clean ─► 03 dedup ─► 04 tokenizer ─► 05 pack             07 build + verify by execution
        └────────────────────────┬──────────────────┘                       │
                                 ▼                                          │
                       06 pretrain (8 x H100, 7.5B tokens)                  │
                                 │  base model                              │
                                 └───────────────► 08 fine-tune v1 ... v5 ◄─┘
                                                          │
                              09 evaluate (real databases, Spider) ──► decide the next round (back to 07)
                                                          │
                                           10 agent loop + playground ─► 11 export and deliver
```

## Cost by phase (from the Modal billing report)

| Phases | What | Billed |
|---|---|---|
| 01 to 05 | Data processing, CPU only (the cleaning fan-out is about $7.68 of it) | **$11.37** |
| 06 | 1-GPU pilot $1.62; 8 x H100 trial $1.96; epoch-1 run $25.86; smoke tests $0.03 | **$29.46** |
| 01 to 06 | **Base model, data processing included** | **$40.83** |
| 07 to 11 | Data building, fine-tuning, evaluation, packaging (local) | $0 |

Source: `modal billing report`, saved as `reports/modal_billing_raw.json`; all 17 cloud apps were billed on 2026-10-06
(UTC). Which individual app belongs to which small stage is inferred from its size; the totals do not depend on that.
The epoch-1 run was first estimated at $23.69 from node time alone; the bill adds container CPU and memory.

## Working method used throughout

* **Gates.** Each cloud stage needed explicit approval before money was spent. Each ended with a short report.
* **Test first.** Pure functions (cleaning rules, packing, loss masks, SQL checks, voting) were written with tests first;
  the suite now has 342 tests.
* **Verify by execution.** Training data and evaluation both rely on running SQL and Python, not on reading answers.
* **Keep the evidence.** Every round's result is in [../DECISIONS.md](../DECISIONS.md); run reports are in `reports/`.
