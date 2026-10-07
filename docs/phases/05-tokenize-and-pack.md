# Page 5: Tokenize and pack

**Runs on:** Modal, CPU. **Code:** `viki_slm_125m/data/packing.py`, `modal_app.py::tokenize`.
**Output:** `tokens/train/*.bin`, `tokens/val/*.bin`, `tokens/index.json` (copy in `reports/index.json`).

## What the phase does
Turns the cleaned text into the exact arrays the trainer reads.

1. Each document is tokenized and followed by one `<|eos|>` token, so the model learns where documents end.
2. The token stream of each file is cut into **windows of 2,048 tokens**.
3. Windows are stored as `uint16` (a token id fits in two bytes because the vocabulary is under 65,536).
4. **Every 100th window goes to validation**, file by file, so each slice has its own validation set and its loss can be
   tracked separately.
5. An index records every file, its window count and its slice, so the trainer can sample by slice.

SQL is marked to be seen **twice** (two epochs) when training samples from the index.

## Results (real counts from the tokenizer)
| Slice | Train tokens | Validation tokens | Passes | Share of what the model sees |
|---|---|---|---|---|
| Python | 2.671B | 27.0M | 1 | 35.6% |
| FineWeb-edu | 1.601B | 16.2M | 1 | 21.4% |
| SQL | 0.461B | 4.7M | **2** | 12.3% |
| Notebooks | 0.766B | 7.8M | 1 | 10.2% |
| SEC filings | 0.601B | 6.1M | 1 | 8.0% |
| Cosmopedia | 0.514B | 5.2M | 1 | 6.9% |
| Math | 0.420B | 4.3M | 1 | 5.6% |
| **Total** | **7.033B** | **71.2M** | | **7.49B effective** |

The real token total is about 6% above the earlier characters-divided-by-four estimate (code costs more tokens per
character, web text slightly fewer).

## Known trade-offs
* The train and validation split is by window, not by document. A document that crosses a boundary leaks a little from
  train into validation. Accepted: the validation numbers are used to watch training, not as a benchmark.
* Because sampling is by window, a 2,048-token window can start in the middle of a document. This is standard for
  pretraining.

## Hand-off
A fixed dataset of 7.03B train tokens (7.49B counting SQL twice) and 71.2M validation tokens, ready for pretraining.
The pretraining step count follows directly: 7.5B tokens divided by 524,288 tokens per step is 14,300 steps.
