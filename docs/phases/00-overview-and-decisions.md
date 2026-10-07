# Page 0: Goals, constraints and decisions

## The goal
Build, from nothing and in the open, a 125-million-parameter language model that can serve as the code-writing part of
an agentic analytics application. In the original brief (`docs/idea.md`) it was described as an expert data scientist
and coder in Python and SQL with working knowledge of supply chain and finance. During the build the evidence narrowed
that to what the model can actually do, so the delivered claim is: **a Python and SQL coding model** that turns a
schema plus a question into SQL, CSV files plus a question into pandas code, and a query result into a short
explanation.

What the agent application needs from it:

1. **SQL generation.** A read-only query for a given schema, including finance and supply-chain schemas.
2. **Insight.** A plain-language reading of the executed result, using only numbers that are in it.
3. **Python.** pandas code for CSV files.
4. **Safe behaviour.** Refuse writes, ask for clarification, or say data is missing when the schema cannot answer.

## Constraints that shaped every choice
| Constraint | Consequence |
|---|---|
| Keep cost low (cap $150 for the project, $90 for pretraining) | Short, planned cloud runs; fine-tuning on a local PC |
| **Free-first** training data; a paid teacher model was put on hold | SFT data built by code, public datasets and execution checks (page 7) |
| Open-source release | Only data with known licences; a licence ledger was built first (page 1) |
| One person plus an AI coding agent | Everything scripted, tested and re-runnable; every cloud stage gated |

## Decisions
| Decision | Choice | Why |
|---|---|---|
| Model size | 124.3M parameters, 12 layers, 768 hidden, 12 heads, 2,048 context | Small enough to train for tens of dollars and fine-tune on a consumer GPU |
| Attention | Full multi-head, not grouped-query | At this size the memory saving is negligible |
| Vocabulary | 32,768 byte-level BPE, trained on our own corpus | Code-heavy text needs a code-aware tokenizer (page 4) |
| SQL dialect | SQLite | Free, in-process, so every query can be executed and checked |
| Pretraining tokens | 7.5B ("Lean to Standard" tier in `cost-and-time-plan.md`) | Fits the budget; one pass over the corpus |
| Fine-tuning hardware | Local RTX 4060 Ti | $0, fast enough (55 to 100 minutes per round) |
| Evaluation | Execution against real databases | Reading answers misleads; running them does not |

## How the work was organised
The implementation plan (`docs/implementation-plan.md`) defined phases with entry and exit gates:

```
0 scaffold  1 feasibility  2 fetch+clean  3 dedup  4 tokenizer  5 pack  6 pretrain
7 SFT data  8 SFT training  9 evaluation  10 release
```

Phases 1 to 6 ran as cloud stages, each approved before spending. Phases 7 to 10 ran locally and were repeated: the
evaluation of each fine-tuning round decided what the next round's data should contain (pages 8 and 9).
This guide adds an "agent loop" page (10) and a delivery page (11) for what was built on top.

## Coding standards applied
Immutable data structures, small focused modules, errors handled explicitly, input validated at boundaries, tests first
with at least 80% coverage (the build reached about 96%), no secrets in code (credentials live in `.env.local`).

## What this build did not do
* The planned **supply-chain** and **data-science-docs** pretraining slices were never sourced, so the base model has
  no supply-chain text. Supply-chain behaviour comes only from fine-tuning data.
* HumanEval, MBPP, DS-1000 and BIRD were planned as evaluations and were **not run**; Spider dev was.
* The model was not instruction-tuned for general knowledge, so chat answers are weak.
