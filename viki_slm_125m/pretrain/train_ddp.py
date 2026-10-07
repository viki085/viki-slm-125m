"""torchrun entry point for multi-GPU pretraining: `torchrun --nproc_per_node=8 -m viki_slm_125m.pretrain.train_ddp ...`.

Expects the packed tokens to be staged under --tokens-dir (local disk) beforehand.
"""

from __future__ import annotations

import argparse
import json
import os

import torch
import torch.distributed as dist

from viki_slm_125m import config
from viki_slm_125m.pretrain import trainlib as tl


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--tokens-dir", required=True)
    p.add_argument("--ckpt-dir", required=True)
    p.add_argument("--total-steps", type=int, required=True)
    p.add_argument("--warmup-steps", type=int, required=True)
    p.add_argument("--micro-batch", type=int, default=32)
    p.add_argument("--ckpt-every", type=int, default=500)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--cap-usd", type=float, required=True)
    p.add_argument("--no-compile", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    if world > 1:
        dist.init_process_group("nccl")
    with open(f"{args.tokens_dir}/index.json", encoding="utf-8") as fh:
        index = json.load(fh)
    refs = tl.refs_from_index(index, f"{args.tokens_dir}/train")
    reader = tl.WindowReader([r.path for r in refs], config.SEQ_LEN)
    sampler = tl.BatchSampler(refs, config.TRAIN.seed)
    val_sets = tl.load_val_sets(index, f"{args.tokens_dir}/val", config.SEQ_LEN, per_source=32)

    seqs_per_step = config.TRAIN.global_batch_tokens // config.SEQ_LEN
    per_rank = seqs_per_step // world
    if per_rank % args.micro_batch:
        raise SystemExit(f"per-rank batch {per_rank} not divisible by micro-batch {args.micro_batch}")
    run = tl.RunConfig(
        total_steps=args.total_steps, micro_batch=args.micro_batch,
        accum_steps=per_rank // args.micro_batch, lr=config.TRAIN.lr, min_lr=config.TRAIN.min_lr,
        warmup_steps=args.warmup_steps, weight_decay=config.TRAIN.weight_decay,
        grad_clip=config.TRAIN.grad_clip, betas=(config.TRAIN.beta1, config.TRAIN.beta2),
        ckpt_every=args.ckpt_every, eval_every=args.eval_every, log_every=10,
        seed=config.TRAIN.seed, device=f"cuda:{local_rank}", compile=not args.no_compile,
        rank=rank, world_size=world)
    os.makedirs(args.ckpt_dir, exist_ok=True)
    budget = tl.BudgetGuard(config.PRETRAIN_GPU_USD_PER_HOUR, world, args.cap_usd)
    out = tl.train(config.MODEL, run, sampler, reader, ckpt_path=f"{args.ckpt_dir}/ckpt.pt",
                   metrics_path=f"{args.ckpt_dir}/metrics.jsonl", val_sets=val_sets, budget=budget)
    if rank == 0:
        hist = out["history"]
        steady = [h["tokens_per_sec"] for h in hist if h["step"] > args.warmup_steps + 20] or \
                 [h["tokens_per_sec"] for h in hist[1:]] or [hist[-1]["tokens_per_sec"]]
        evals = [h for h in hist if "val_loss" in h]
        summary = {"world_size": world, "steps": out["final_step"], "total_steps": args.total_steps,
                   "stopped_by_budget": out["stopped_by_budget"],
                   "first_loss": hist[0]["loss"], "final_loss": hist[-1]["loss"],
                   "val_loss_last": evals[-1]["val_loss"] if evals else None,
                   "tokens_per_sec_total": sum(steady) / len(steady),
                   "elapsed_s": hist[-1]["elapsed_s"]}
        with open(f"{args.ckpt_dir}/summary.json", "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)
        print("SUMMARY", json.dumps(summary))
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
