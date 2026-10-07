"""Supervised fine-tuning of the pretrained base model (single GPU, runs locally on Windows).

Loss is computed only on trained (assistant) tokens. No torch.compile (unreliable on Windows).
Usage: python -m viki_slm_125m.sft.train --epochs 3
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
import torch.nn.functional as F

from viki_slm_125m import config
from viki_slm_125m.sft import sft_dataset as sds
from viki_slm_125m.pretrain import trainlib


@dataclass(frozen=True)
class SftConfig:
    lr: float = 3e-5
    epochs: int = 3
    micro_tokens: int = 8_192        # padded tokens per micro-batch
    accum_steps: int = 8
    warmup_frac: float = 0.03
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    eval_every: int = 100
    ckpt_dir: str = "artifacts/sft"
    seed: int = 1337
    device: str = "cuda"
    decay_frac: float = 0.5
    log_every: int = 10
    max_val_examples: int = 300


# multi-table real schemas were the biggest measured gap (Spider dev), so they are seen twice per epoch
REPEAT = {"spider": 2}


def load_base_weights(path: str) -> dict:
    return torch.load(path, map_location="cpu", weights_only=True)["model"]


def masked_loss_sum(model, batch: Mapping[str, torch.Tensor]) -> tuple[torch.Tensor, int]:
    """Summed cross-entropy over trained tokens and their count."""
    logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
    labels = batch["labels"][:, 1:]
    keep = labels != -100
    n = int(keep.sum())
    if n == 0:
        return logits.sum() * 0.0, 0
    selected = logits[:, :-1][keep].float()
    return F.cross_entropy(selected, labels[keep], reduction="sum"), n


@torch.no_grad()
def evaluate(model, val_sets: Mapping[str, Sequence[sds.Encoded]], pad_id: int, cfg: SftConfig) -> dict[str, float]:
    model.eval()
    out: dict[str, float] = {}
    for name, exs in val_sets.items():
        exs = list(exs)[: cfg.max_val_examples]
        if not exs:
            continue
        total, count = 0.0, 0
        for b in sds.token_budget_batches([len(e.ids) for e in exs], cfg.micro_tokens, random.Random(0)):
            batch = {k: v.to(cfg.device) for k, v in sds.collate([exs[i] for i in b], pad_id).items()}
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cfg.device.startswith("cuda")):
                s, n = masked_loss_sum(model, batch)
            total += float(s)
            count += n
        out[name] = total / max(1, count)
    model.train()
    return out


def _save(path: str, model, opt, step: int) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    torch.save({"model": model.state_dict(), "opt": opt.state_dict() if opt else None, "step": step}, tmp)
    os.replace(tmp, path)


def train_sft(model_cfg: config.ModelConfig, cfg: SftConfig, train: Sequence[sds.Encoded],
              val_sets: Mapping[str, Sequence[sds.Encoded]], pad_id: int, base_path: str | None = None,
              stop_after: int | None = None, metrics_path: str | None = None) -> dict:
    torch.manual_seed(cfg.seed)
    model = trainlib.build_model(model_cfg)
    if base_path:
        model.load_state_dict(load_base_weights(base_path))
    model.to(cfg.device).train()
    opt = trainlib.make_optimizer(model, cfg.lr, cfg.weight_decay, (0.9, 0.95), fused=cfg.device.startswith("cuda"))

    lengths = [len(e.ids) for e in train]
    micro = [b for ep in range(cfg.epochs)
             for b in sds.token_budget_batches(lengths, cfg.micro_tokens, random.Random(cfg.seed + ep))]
    total_steps = math.ceil(len(micro) / cfg.accum_steps)
    warmup = max(1, int(total_steps * cfg.warmup_frac))
    ckpt_path = os.path.join(cfg.ckpt_dir, "ckpt.pt")
    step = 0
    if os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        model.load_state_dict(ck["model"])
        if ck.get("opt"):
            opt.load_state_dict(ck["opt"])
        step = ck["step"]
    history: list[dict] = []
    best = float("inf")
    start, t_start = step, time.time()
    use_amp = cfg.device.startswith("cuda")
    while step < total_steps:
        t0 = time.time()
        lr = trainlib.lr_at(step, total_steps, warmup, cfg.lr, cfg.lr * 0.1, cfg.decay_frac)
        for g in opt.param_groups:
            g["lr"] = lr
        group = micro[step * cfg.accum_steps:(step + 1) * cfg.accum_steps]
        batches = [sds.collate([train[i] for i in b], pad_id) for b in group]
        total_n = sum(int((b["labels"][:, 1:] != -100).sum()) for b in batches) or 1
        opt.zero_grad(set_to_none=True)
        loss_total, tokens = 0.0, 0
        for batch in batches:
            batch = {k: v.to(cfg.device) for k, v in batch.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                loss_sum, _ = masked_loss_sum(model, batch)
            (loss_sum / total_n).backward()
            loss_total += float(loss_sum) / total_n
            tokens += int(batch["attention_mask"].sum())
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip))
        opt.step()
        step += 1
        want_eval = bool(val_sets) and (step % cfg.eval_every == 0 or step == total_steps)
        if step % cfg.log_every == 0 or step == start + 1 or want_eval or step == total_steps:
            entry = {"step": step, "total_steps": total_steps, "loss": loss_total, "lr": lr,
                     "grad_norm": grad_norm, "tokens_per_sec": tokens / max(1e-9, time.time() - t0),
                     "elapsed_s": time.time() - t_start}
            if want_eval:
                entry["val_loss"] = evaluate(model, val_sets, pad_id, cfg)
                mean_val = sum(entry["val_loss"].values()) / max(1, len(entry["val_loss"]))
                if mean_val < best:
                    best = mean_val
                    _save(os.path.join(cfg.ckpt_dir, "best.pt"), model, None, step)
            history.append(entry)
            if metrics_path:
                with open(metrics_path, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(entry) + "\n")
        done = step >= total_steps or (stop_after is not None and step >= stop_after)
        if want_eval or done:
            _save(ckpt_path, model, opt, step)
        if done:
            break
    if not val_sets:
        _save(os.path.join(cfg.ckpt_dir, "best.pt"), model, None, step)
    return {"history": history, "final_step": step, "total_steps": total_steps}


def main(argv: list[str] | None = None) -> None:
    from tokenizers import Tokenizer

    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data/sft")
    p.add_argument("--base", default="artifacts/base-e1/ckpt.pt")
    p.add_argument("--tokenizer", default="artifacts/tokenizer/tokenizer.json")
    p.add_argument("--out", default="artifacts/sft")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--micro-tokens", type=int, default=8192)
    p.add_argument("--accum", type=int, default=8)
    args = p.parse_args(argv)

    tok = Tokenizer.from_file(args.tokenizer)
    pad_id = tok.token_to_id(config.SPECIAL_TOKENS["pad_token"])

    def enc(text):
        e = tok.encode(text)
        return e.ids, e.offsets

    d = args.data_dir
    caps = {"gretel": (f"{d}/gretel_train.jsonl", 40_000), "repair": (f"{d}/repair_train.jsonl", 12_000),
            "domain": (f"{d}/domain_train.jsonl", 40_000), "general": (f"{d}/general_train.jsonl", 25_000),
            "refusal": (f"{d}/refusal_train.jsonl", 12_000),
            "spider": (f"{d}/spider_train.jsonl", 9_000)}
    vcaps = {"gretel": (f"{d}/gretel_val.jsonl", 300), "repair": (f"{d}/repair_val.jsonl", 150),
             "domain": (f"{d}/domain_val.jsonl", 300), "general": (f"{d}/general_val.jsonl", 300),
             "refusal": (f"{d}/refusal_val.jsonl", 200),
             "spider": (f"{d}/spider_val.jsonl", 300)}
    mix = sds.load_mixture(caps, enc, config.SEQ_LEN, seed=0)
    val = sds.load_mixture(vcaps, enc, config.SEQ_LEN, seed=0)
    train = [e for k, v in mix.items() for e in v * REPEAT.get(k, 1)]
    print({k: len(v) for k, v in mix.items()}, "train examples", len(train))
    cfg = SftConfig(lr=args.lr, epochs=args.epochs, micro_tokens=args.micro_tokens, accum_steps=args.accum,
                    ckpt_dir=args.out, device="cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(cfg.ckpt_dir, exist_ok=True)
    out = train_sft(config.MODEL, cfg, train, val, pad_id, base_path=args.base,
                    metrics_path=os.path.join(cfg.ckpt_dir, "metrics.jsonl"))
    print("final step", out["final_step"], "of", out["total_steps"])


if __name__ == "__main__":
    main()
