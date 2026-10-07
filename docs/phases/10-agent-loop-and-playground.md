# Page 10: The agent loop and the playground

**Code:** `viki_slm_125m/app/ui_server.py`, `ui_modes.py`, `static/index.html`; `viki_slm_125m/sft/sql_vote.py`,
`sql_verify.py`, `sandbox.py`. **Run:** `python -m viki_slm_125m.app.ui_server`, then http://127.0.0.1:8000.

## Why an agent loop
The model alone writes text. Its raw accuracy on hard questions is low, but its mistakes are often **detectable**: a query
that errors, or one whose result most other attempts disagree with. Running the model inside a loop that executes its
output turns detectable mistakes into retries and makes the system better than a single greedy answer.

## SQL mode
```
question + schema
      │
      ├─ greedy query  ─┐
      └─ 8 sampled queries (temperature 0.7, top-p 0.95) ─┤
                                                          ▼
                    run every candidate on the database (read-only, timeout, row cap)
                                                          │
            drop queries that error ─► group the rest by identical result ─► largest group wins (ties: greedy)
                                                          │
                        none ran?  ─► feed the database error back, retry (up to 2 times)
                                                          │
                        result table appended ─► model writes the insight ─► page shows SQL, table, insight, vote count
```
* The vote is on the **result**, not the SQL text: different queries that return the same rows count together.
* The repair step uses the format taught in fine-tuning (the failing query is shown, the error follows, the model rewrites it).
* Measured effect: voting raised Spider dev from 4.6% to 12.3% (v4) and 22.1% to 33.0% (v5); query run rate on gretel
  went from 93% to 98%. It does not fix a mistake that all samples repeat.
* Cost: about 2 to 3 seconds per question on the local GPU.

## Python mode
The user picks sample data (the generated supply-chain or finance tables exported as CSV) or pastes CSV files in a
`--- name.csv` format. The model writes pandas code; **an AST allow-list** rejects anything outside a short list of safe
imports and banned names, then the code runs in a fresh interpreter with a timeout. The printed output goes back to the
model, which writes the explanation. A model that forgets the opening `<|python|>` tag has its code recovered, and the
page shows a format warning.

## Chat mode
Free text, nothing is executed. Decoding uses a repetition penalty (1.15) and bans repeating any 4-word sequence,
because plain greedy decoding on a small model loops. The answers are not reliable for factual questions; the page labels
this mode "weak".

## The page
* Three tabs (SQL analyst, Python on data, Chat), sample chips for each (about 28 in all), a data-source picker, a
  collapsible schema box, and a results area showing the generated code, the real result table, the insight, the vote
  information, any repair attempts, and the raw model output.
* Sections for **The numbers** (parameters, tokens, the real cloud cost of $40.83 split into $11.37 data processing
  and $29.46 GPU, SFT size), **Evaluation** (the v5 scores) and **Architecture and data**, plus a plain disclaimer.
* Backend: a standard-library HTTP server. `GET /` serves the page, `GET /api/databases` the sample schemas,
  `POST /api/ask {mode, db, schema, question}` runs the loop. It listens on 127.0.0.1 only.

## Safety measures in the loop
| Risk | Measure |
|---|---|
| The model writes `DELETE` / `DROP` / multiple statements | `sql_verify.is_read_only` refuses them before execution; the model is also trained to decline writes |
| Very large or never-ending query | Row cap (200) and a time limit on every query |
| Model-written Python does something unsafe | AST allow-list, fresh process, temp directory, timeout (not a hard security boundary: run it on a machine you trust) |
| User text containing reserved tokens | Chat input has `<|` neutralised so users cannot forge role tokens |
| Wrong answer that looks right | The page always shows the SQL or code beside the result |

## Known limits
A query that runs but is wrong looks plausible, and voting cannot detect it when all samples agree on the same mistake.
The loop does not know which table a question refers to; wrong-table errors on large schemas remain the main failure.
