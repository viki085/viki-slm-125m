# Page 6: Pretraining

**Runs on:** Modal, 1 GPU for the pilot and 8 x H100 for the real run. **Cost:** $29.46 billed for all GPU apps
(epoch-1 run $25.86). **Code:** `viki_slm_125m/pretrain/trainlib.py`, `train_ddp.py`, `viki_slm_125m/model/architecture.py`,
`modal_app.py::pretrain_pilot`, `::pretrain_8gpu`, `::smoke`.
**Reports:** `reports/phase6_pilot_report.json`, `phase6_trial8_report.json`, `phase6_base-e1_report.json`, `smoke_base-e1.json`.

## What pretraining is
The model starts with random weights and learns to predict the next token over 7.5B tokens of code, notebooks, filings
and web text. Everything it knows about Python syntax, SQL structure and English comes from this phase.

## The design
| Item | Choice |
|---|---|
| Architecture | The 124.3M Llama-style decoder (see the README) |
| Batch | 524,288 tokens per step (256 windows of 2,048) |
| Steps | 14,300, so 7.5B tokens |
| Optimiser | AdamW, betas 0.9 / 0.95, weight decay 0.1 (not on norms and biases), gradient clip 1.0 |
| Learning rate | Warmup-stable-decay: linear warmup over 200M tokens to 6e-4, constant, then cosine decay to 6e-5 over the last 10% of steps |
| Precision and speed | bf16 mixed precision, `torch.compile`, scaled dot-product attention |
| Parallelism | PyTorch DDP, 8 processes on one 8 x H100 node (`torchrun -m viki_slm_125m.pretrain.train_ddp`) |
| Data order | A deterministic sampler picks each step's windows from the slice weights using (seed, step), so a stopped run resumes with exactly the same data |
| Monitoring | Validation loss per slice every 1,000 steps; checkpoint every 500 steps; metrics appended to `metrics.jsonl` |
| Cost protection | A budget guard computes spend from elapsed time and GPU count and aborts with a saved checkpoint at a dollar cap; all ranks must agree before stopping |
| Continuing later | A snapshot (`ckpt.pt.predecay`) is saved at step 12,870, the end of the constant-rate phase, so more training can resume from there |

## The three runs, in order
Each needed explicit approval before spending ("6a + 6b only", then "8-GPU trial only", then "Yes, epoch 1: 7.5B tokens").

| Run | Setup | Result | Purpose |
|---|---|---|---|
| Pilot | 1 GPU, 572 steps (about 300M tokens) | Loss 10.55 to 2.67; 404K tokens per second; fluent but repetitive code samples | Prove the loop learns; measure speed; project the full cost (about $24 for 8B tokens) |
| 8-GPU trial | 200 steps | 3.2M tokens per second; loss 10.55 to 4.34 | Prove multi-GPU, resume and the budget guard on real hardware |
| **Epoch 1** | **14,300 steps, 8 x H100** | **About 45 minutes, 3.22M tokens per second, no loss spikes, no budget stop; training loss 10.55 to 1.69** | The base model, `base-e1` |

## Final validation loss (lower is better)
| Slice | Loss | Reading |
|---|---|---|
| SQL | 0.701 | Very low: SQL is repetitive and was seen twice |
| Notebooks | 1.055 | |
| Python | 1.291 | |
| SEC filings | 1.654 | |
| Math | 2.012 | |
| Cosmopedia | 2.080 | |
| FineWeb-edu | 3.124 | Open-ended web prose is the hardest to predict |

The pattern is what the data mix would predict: the model is strong where it saw a lot of regular code, and
weak at broad prose.

## Smoke test (generation after pretraining)
Prompted with Python and SQL openings, the base model completed them fluently and with sensible structure. Finance
text came out plausible. **Supply-chain text was wrong**: the corpus contained none. This predicted the later finding
that supply-chain behaviour could only come from fine-tuning data.

## Cost
| App | Billed |
|---|---|
| 1-GPU pilot | $1.62 |
| 8 x H100 trial | $1.96 |
| **Epoch-1 run** | **$25.86** (H100 $23.93, container CPU $1.15, memory $0.78) |
| Smoke tests | $0.03 |

The epoch-1 run was first reported as $23.69 from node time at an assumed $3.95 per GPU-hour; the bill also charges the
host CPU and memory. Adding the data-processing stages gives **$40.83** for the base model.

## Problems met and how they were fixed
| Problem | Fix |
|---|---|
| The GPU image lacked `tokenizer_lib`, so the pilot crashed on import | Added the module to the GPU image sources |
| `transformers` failed to import under `torch.device("meta")` | Import it at module top, before the device context |
| A smoke-test patch matched the wrong function name | Match on the exact definition line |
| Git Bash mangled Modal volume paths | `MSYS_NO_PATHCONV=1` |

## Can it train longer?
Yes. More tokens would almost certainly lower the validation loss, because the model saw each document about once.
Training could resume from `ckpt.pt.predecay` on the same or a new corpus (for example one that adds supply-chain and
data-science text) at roughly the epoch-1 cost, about $25 per extra pass. This was deliberately **not** done.

## Hand-off
`base-e1`: 14,300 steps, final loss 1.69, weights at `artifacts/base-e1/ckpt.pt`. Fine-tuning starts from it.
