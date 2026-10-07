# Page 7: Building the fine-tuning data

**Runs on:** a local CPU. **Cost:** $0. **Code:** `viki_slm_125m/sft/` (formats, verification) and
`viki_slm_125m/sft/builders/` (one script per dataset). **Output:** `data/sft/*.jsonl`.

## Why a separate data phase
The base model only continues text. To act as an agent it must learn a fixed format (schema, question, SQL, result,
insight) and behaviours (refuse, clarify, repair). With no paid teacher model available, the rule was:
**build data by code, and keep an example only if running it proves it correct.**

## The training format
Each example is a short conversation: a system message, the user turn (`<|schema|> ... <|/schema|>` plus the question),
the assistant's `<|think|>` plan and `<|sql|>` or `<|python|>` block, the **tool result** inserted by the application
(`<|result|>` or `<|output|>`), and a final `<|insight|>` answer.

* **Loss masking:** only assistant tokens are trained. System text, user text, schemas and tool results are masked, so
  the model learns to write answers, not to imitate prompts or fabricate results.
* **A per-turn "trained" flag** lets a deliberately wrong query appear in context without being learned (repair data).
* **Insights are faithful by construction:** they are generated from the actual result rows, and a check rejects any
  insight containing a number that is not in the result.

## The verification tools
| Tool | What it enforces |
|---|---|
| `sql_verify.py` | Read-only gate (one statement, no writes), a timeout, row cap, and result formatting. The same code runs in training data, evaluation and the playground |
| `sandbox.py` | Python from templates or the model is checked with an AST allow-list (only pandas, numpy, math and similar imports; no `exec`, `open`, dunder access, URLs or absolute paths) and run in a fresh interpreter with a temp directory and a timeout. It limits mistakes; it is not a security boundary |
| `sft_data.numbers_supported` | Every number in an insight must appear in the result |

## The six datasets

| Dataset | Builder | Size used in v5 | What it contributes |
|---|---|---|---|
| **gretel** | `build_sql_sft` | 40,000 (of 50,212) | `gretelai/synthetic_text_to_sql`: SQL over many industries. Each query is executed on a database rebuilt from its own schema; rows without results are dropped. Four domains (oceanography, ethical fashion, rural development, arts and culture) are held out as an unseen test |
| **domain** | `build_domain_sft` | 24,025 | Our finance and supply-chain databases (below), SQL and pandas, plus refusal-style examples |
| **spider** | `build_spider_sft` | 6,275 loaded of 6,414 built | Real multi-table schemas with foreign keys (page 8, v5) |
| **repair** | `build_repair_sft` | 10,789 | A broken query (masked), the real database error, then the corrected query |
| **refusal** | `build_refusal_sft` | 11,291 | Declines write requests, asks to clarify vague questions, says when data is missing |
| **general** | `build_general_sft` | 25,000 (of 49,501) | Everyday conversation, code instructions, summarising and rewriting (smoltalk subsets), to keep language and formatting skills |

## The finance and supply-chain generators
`domain/domain_data.py` creates two SQLite databases with seeded random data:

* **Supply chain:** suppliers, products, warehouses, inventory, purchase_orders, shipments, demand_forecast.
* **Finance:** customers, accounts, transactions, loans, repayments, gl_entries.

Question templates with slots produce the gold SQL **by construction** (for example "Which {n} suppliers have the
highest rating?" fills a query), then each instance is executed and kept only if it returns rows:

* **86 SQL templates** (46 supply-chain, 40 finance) and **29 pandas templates** (15 and 14), built up over rounds v1 to v4.
* **7 SQL templates held out** from training and used only for evaluation.
* **Paraphrasing:** opening phrases ("Can you tell me", "Quick question") and alternative wordings; at most 12 copies of
  the same question and query, so no single pattern dominates.
* Every pandas example runs in the sandbox on real CSV exports of the generated tables; its printed output becomes
  the result the model reads.
* From v3 every pandas example lists **all** of the domain's CSV files, so the model must choose the right ones.

## Repair, refusal and clarification
* **Repair kinds:** misspelled column, misspelled table, syntax error (all from v1), and "rewrite" (v4), where the first
  query uses a column from the wrong table and the fix is a whole new select list.
* **Refusal data (v2):** 5,875 real "delete or update" requests built on gretel's own schemas, 2,954 ambiguous questions
  ("who is the best employee?") answered with a clarifying question that lists the schema's real numeric columns, and
  2,462 questions about concepts the schema lacks. Test prompts (100 per kind) come from domains never trained on.

## Spider conversion (v5)
8,659 training records from the Spider train and train-others splits, each re-run on its real database:
**6,414 kept** from 136 databases; 1,723 dropped (error or no rows or too many rows) and 63 as repeats. 459 more
examples from 10 whole held-out databases form a validation set of unseen schemas. **The 20 Spider dev databases are
excluded entirely**, so the Spider dev benchmark stays clean.

## Test sets built at the same time
| Test | Contents |
|---|---|
| gretel unseen domains | 600 questions from the four held-out domains |
| Domain templates | 420 questions (60 per held-out template), seven templates |
| pandas | 58 items with unseen database seeds |
| Behaviour | 300 prompts (100 each: refuse, missing, clarify) on unseen domains |
| Spider dev | 1,034 questions (page 9) |

## Problems met and how they were fixed
| Problem | Fix |
|---|---|
| Generated Python lost its `\n` escapes through the shell | Write code files with the editor tool, not heredocs |
| Only 300 domain SQL examples survived the repeat cap | Added paraphrasing, a repeat limit of 12, and more templates |
| A comprehension used `d` twice (walrus bug); unescaped braces in templates | Rewritten as a loop; doubled the braces |
| First SFT smoke run spilled memory (9.6 GB, 2.6K tokens per second) | Smaller micro-batches (page 8) |

## Build order
`build_sql_sft` then `build_general_sft`, `build_domain_sft`, `build_spider_sft`, `build_repair_sft` (reads gretel and
domain output), `build_refusal_sft`.
