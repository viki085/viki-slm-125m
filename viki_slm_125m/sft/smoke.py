"""Short GPU smoke test of the SFT loop: speed and peak memory.

Usage: python -m viki_slm_125m.sft.smoke [micro_tokens] [accum_steps]
"""

from __future__ import annotations

import shutil
import sys
import time

import torch
from tokenizers import Tokenizer

from viki_slm_125m import config
from viki_slm_125m.sft import sft_dataset as sds
from viki_slm_125m.sft import train as sft_train

TOKENIZER = "artifacts/tokenizer/tokenizer.json"
BASE = "artifacts/base-e1/ckpt.pt"
OUT = "artifacts/sft_smoke"
CAPS = {"gretel": ("data/sft/gretel_train.jsonl", 40_000), "repair": ("data/sft/repair_train.jsonl", 10_000),
        "domain": ("data/sft/domain_train.jsonl", 40_000), "general": ("data/sft/general_train.jsonl", 25_000)}


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    micro = int(args[0]) if len(args) > 0 else 8192
    accum = int(args[1]) if len(args) > 1 else 8
    tok = Tokenizer.from_file(TOKENIZER)
    pad = tok.token_to_id("<|pad|>")

    def enc(text):
        e = tok.encode(text)
        return e.ids, e.offsets

    t0 = time.time()
    mix = sds.load_mixture(CAPS, enc, 2048, 0)
    train = [e for v in mix.values() for e in v]
    toks = sum(len(e.ids) for e in train)
    trained = sum(sum(e.mask) for e in train)
    print({k: len(v) for k, v in mix.items()}, "examples", len(train), "tokens %.1fM" % (toks / 1e6),
          "trained tokens %.1fM (%.0f%%)" % (trained / 1e6, 100 * trained / toks), "load %.0fs" % (time.time() - t0))
    shutil.rmtree(OUT, ignore_errors=True)
    cfg = sft_train.SftConfig(epochs=1, micro_tokens=micro, accum_steps=accum, ckpt_dir=OUT, log_every=5,
                              eval_every=1000)
    torch.cuda.reset_peak_memory_stats()
    out = sft_train.train_sft(config.MODEL, cfg, train, {}, pad, base_path=BASE, stop_after=20)
    h = out["history"]
    steady = [x["tokens_per_sec"] for x in h[1:]] or [h[0]["tokens_per_sec"]]
    rate = sum(steady) / len(steady)
    print("steps", out["final_step"], "of", out["total_steps"], "| loss first %.3f last %.3f" % (h[0]["loss"], h[-1]["loss"]),
          "| tok/s %.0f" % rate, "| peak GPU mem %.1f GB" % (torch.cuda.max_memory_allocated() / 1e9))
    print("est. time for 3 epochs: %.0f min" % (3 * toks / rate / 60))
    return 0


if __name__ == "__main__":
    sys.exit(main())
