# Page 9: Evaluation

**Runs on:** the local GPU. **Cost:** $0. **Code:** `viki_slm_125m/eval/` (`evaluator.py`, `eval_sft.py`,
`eval_python.py`, `eval_vote.py`, `eval_spider.py`). **Reports:** `reports/eval_*_report.json`.

## Principle
A model that writes code is judged by running the code. Reading answers rewards fluent mistakes, which this model makes
often. Every score below comes from executing the generated SQL or Python.

## How each test works
| Test | Method | Size |
|---|---|---|
| **SQL execution accuracy** | Run the generated query and the gold query on the same database; the result sets must match (order-insensitive, column names ignored, numbers rounded to 4 decimals) | per test below |
| **Rounding-tolerant accuracy** | Same, but numbers in a column are compared at the coarsest precision used in it, so `ROUND(x,2)` matches `ROUND(x,1)`; real differences still fail. Reported next to the strict score; the strict score is never changed | |
| **gretel unseen domains** | Questions from four domains never in training | 600 |
| **Domain templates** | Held-out question templates on our finance and supply-chain databases; seeds not seen in training | 420 (60 per template) |
| **Spider dev** | The standard benchmark: 1,034 questions on 20 databases with real data, none of which are in pretraining or fine-tuning (checked in page 3 and page 7). Prompts are the real database's `CREATE TABLE` statements | 1,034 |
| **Python (pandas)** | The generated code runs in the sandbox on unseen data; its printed table must equal the gold code's table | 58 |
| **Behaviour** | 100 prompts each for "refuse a write", "ask to clarify", "say data is missing", on schemas from domains never trained on. Pass = no SQL written, an insight block, and the expected wording | 300 |
| **Faithfulness** | Every number in the model's explanation must appear in the executed result | all answered items |
| **Voting** | Greedy query against 4 or 8 sampled queries chosen by majority result | gretel, domain, Spider |

## Results across the five fine-tuning rounds
| Test | v1 | v2 | v3 | v4 | v5 |
|---|---|---|---|---|---|
| gretel unseen domains, greedy | 54.7% | 57.0% | 56.2% | 55.7% | **58.0%** |
| gretel, 8-sample voting | | | | 59.3% | 60.2% |
| Original four held-out templates (240) | 21.7% | 19.6% | **91.2%** | 91.7% | 89.2% |
| All seven held-out templates (420), greedy | | | | 62.9% | 61.4% (66.7% tolerant) |
| Three newer held-out templates | | | | 24% | 24% |
| Python (pandas), unseen data | | 17.2% | 100% | 100% | 100% |
| Refuse / clarify / missing on unseen schemas | 0% | 100% | 100% | 100% | 100% |
| Insights match the result | 98.7% | 98.9% | 98.9% | 99.1% | 99.5% |
| **Spider dev, greedy** | | | | 4.6% | **22.1%** |
| **Spider dev, 8-sample voting** | | | | 12.3% | **33.0%** |

Notes: the v5 gretel figure is from the final run after the rotary-precision fix (57.3% before it; page 11). The voting
column for v5 was measured before that fix.

## How to read these numbers
1. **Spider is the honest outside measure.** 33.0% with voting means about two thirds of unseen multi-table questions are
   still wrong. Before v5 it was 12.3%.
2. **The template scores measure transfer, not open-ended skill.** After v3 the new templates taught the same patterns
   as the held-out ones, so 91.2% shows the model learned those patterns. The three newer templates, written to test
   something different (ascending order, lowest/shortest, column choice), score 24%.
3. **Python and behaviour at 100% are in-distribution.** They prove the format and the safety behaviour work; they do
   not prove generality.
4. **Voting helps when mistakes are random** (+4 points on gretel, +7.7 on Spider for v4) **and not when they are
   systematic** (for v5 it added nothing on the held-out templates because every sample repeats the same error).
5. **The faithfulness score is narrower than its name.** It checks that every number in an insight appears in the result.
   It does not check that the right name is attached to a number: in the README's playground example the insight says the
   smallest group is "Corporate (1)" when the table shows it is SME. A name-aware check would be a useful next metric.
6. **Noise.** Per-template scores rest on 60 questions; Spider validation loss on 10 databases. Differences of one or two
   points between versions are not meaningful.

## What was not evaluated
HumanEval, MBPP, DS-1000 and BIRD were planned and not run; there is no measured score for general Python coding or
for chat and knowledge questions. A set of eight data-science concept questions was tried by hand: answers began
correctly and then drifted into invented details (an invented formula for a p-value, a wrong definition of a left join),
which is why the delivered claim excludes concept explanation.

## Hand-off
Evaluation fed three things: the next round's data (page 8), the choice to use voting and repair in the product (page 10),
and the limits written into the model card (page 11).
