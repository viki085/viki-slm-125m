"""Pretraining library: WSD schedule, deterministic mixture sampler, training loop.

Pure PyTorch (no framework). Resumable: the batch for a step depends only on (seed, step),
so a restarted run continues exactly where it stopped. Multi-GPU uses DDP when
`RunConfig.world_size > 1` and the caller has initialised torch.distributed.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import torch

from viki_slm_125m import config
from viki_slm_125m.model.architecture import build_model  # noqa: F401  (re-exported for training code)


# ------------------------------------------------------------------ schedule

def lr_at(step: int, total_steps: int, warmup_steps: int, lr: float, min_lr: float,
          decay_frac: float = 0.1) -> float:
    """Warmup-stable-decay: linear warmup, constant, cosine decay over the last `decay_frac`."""
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    decay_start = int(total_steps * (1.0 - decay_frac))
    if warmup_steps >= decay_start:
        raise ValueError("warmup must end before the decay phase starts")
    if step < warmup_steps:
        return lr * (step + 1) / warmup_steps
    if step < decay_start:
        return lr
    progress = min(1.0, (step - decay_start) / max(1, total_steps - 1 - decay_start))
    return min_lr + 0.5 * (lr - min_lr) * (1.0 + math.cos(math.pi * progress))


# ------------------------------------------------------------------- sampler

@dataclass(frozen=True)
class WindowRef:
    source: str
    path: str
    n_windows: int
    weight: float          # sampling weight = windows x epochs


class BatchSampler:
    """i.i.d. mixture sampling; the batch for a step is a pure function of (seed, step)."""

    def __init__(self, refs: Sequence[WindowRef], seed: int):
        if not refs:
            raise ValueError("no data files to sample from")
        self.refs = list(refs)
        self.seed = seed
        self._cum = np.cumsum([r.weight for r in self.refs])
        self._n = np.asarray([r.n_windows for r in self.refs])

    def sample(self, step: int, n: int) -> list[tuple[int, int]]:
        rng = np.random.default_rng([self.seed, step])
        files = np.searchsorted(self._cum, rng.random(n) * self._cum[-1], side="right")
        files = np.minimum(files, len(self.refs) - 1)
        wins = (rng.random(n) * self._n[files]).astype(np.int64)
        return list(zip(files.tolist(), wins.tolist()))


def refs_from_index(index: Mapping, split_dir: str) -> list[WindowRef]:
    """WindowRefs for the train files described by tokens/index.json."""
    refs = []
    for s in index["shards"]:
        n = s["train_windows"]
        if n <= 0:
            continue
        stem = s["file"][: -len(".jsonl")]
        refs.append(WindowRef(s["source"], f"{split_dir}/{s['source']}-{stem}.bin", n, n * s["epochs"]))
    return refs


class WindowReader:
    """Lazy memory-mapped access to packed uint16 windows."""

    def __init__(self, paths: Sequence[str], seq_len: int):
        self.paths = list(paths)
        self.seq_len = seq_len
        self._maps: dict[int, np.ndarray] = {}

    def _map(self, i: int) -> np.ndarray:
        if i not in self._maps:
            self._maps[i] = np.memmap(self.paths[i], dtype=np.uint16, mode="r").reshape(-1, self.seq_len)
        return self._maps[i]

    def get(self, file_idx: int, window: int) -> np.ndarray:
        return np.asarray(self._map(file_idx)[window])

    def batch(self, picks: Sequence[tuple[int, int]]) -> torch.Tensor:
        return torch.from_numpy(np.stack([self.get(f, w) for f, w in picks]).astype(np.int64))


def load_val_sets(index: Mapping, val_dir: str, seq_len: int, per_source: int = 32) -> dict[str, torch.Tensor]:
    """First `per_source` validation windows of every source (deterministic)."""
    out: dict[str, list[np.ndarray]] = {}
    for s in index["shards"]:
        have = out.setdefault(s["source"], [])
        if s["val_windows"] <= 0 or len(have) >= per_source:
            continue
        stem = s["file"][: -len(".jsonl")]
        data = np.memmap(f"{val_dir}/{s['source']}-{stem}.bin", dtype=np.uint16,
                         mode="r").reshape(-1, seq_len)
        have.extend(np.asarray(data[i]) for i in range(min(len(data), per_source - len(have))))
    return {k: torch.from_numpy(np.stack(v).astype(np.int64)) for k, v in out.items() if v}


# --------------------------------------------------------------------- model

def tiny_model_config() -> config.ModelConfig:
    return config.ModelConfig(vocab_size=64, hidden_size=32, intermediate_size=64,
                              num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
                              max_position_embeddings=64)


def make_optimizer(model, lr: float, weight_decay: float, betas: tuple[float, float], fused: bool):
    decay = [p for p in model.parameters() if p.requires_grad and p.ndim >= 2]
    no_decay = [p for p in model.parameters() if p.requires_grad and p.ndim < 2]
    groups = [{"params": decay, "weight_decay": weight_decay},
              {"params": no_decay, "weight_decay": 0.0}]
    return torch.optim.AdamW(groups, lr=lr, betas=betas, fused=fused or None)


# -------------------------------------------------------------------- budget

class BudgetGuard:
    def __init__(self, rate_per_gpu_hour: float, n_gpus: int, cap_usd: float):
        if n_gpus <= 0:
            raise ValueError("n_gpus must be positive")
        self.rate, self.n, self.cap = rate_per_gpu_hour, n_gpus, cap_usd

    def spent(self, elapsed_s: float) -> float:
        return self.rate * self.n * elapsed_s / 3600.0

    def exceeded(self, elapsed_s: float) -> bool:
        return self.spent(elapsed_s) >= self.cap


# ---------------------------------------------------------------- checkpoints

def save_checkpoint(path: str, model, opt, step: int) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step}, tmp)
    os.replace(tmp, path)


def load_checkpoint(path: str) -> dict | None:
    if not os.path.exists(path):
        return None
    return torch.load(path, map_location="cpu", weights_only=True)


# ------------------------------------------------------------------ training

@dataclass(frozen=True)
class RunConfig:
    total_steps: int
    micro_batch: int
    accum_steps: int
    lr: float
    min_lr: float
    warmup_steps: int
    weight_decay: float = 0.1
    betas: tuple[float, float] = (0.9, 0.95)
    grad_clip: float = 1.0
    decay_frac: float = 0.1
    ckpt_every: int = 500
    eval_every: int = 1_000
    log_every: int = 20
    seed: int = 1337
    device: str = "cuda"
    compile: bool = True
    rank: int = 0
    world_size: int = 1
    stop_after: int | None = None   # stop (and checkpoint) after this many completed steps


@torch.no_grad()
def evaluate(model, val_sets: Mapping[str, torch.Tensor], device: str, micro_batch: int) -> dict[str, float]:
    model.eval()
    out = {}
    for name, data in val_sets.items():
        losses = []
        for i in range(0, len(data), micro_batch):
            batch = data[i:i + micro_batch].to(device)
            losses.append(model(input_ids=batch, labels=batch).loss.item() * len(batch))
        out[name] = sum(losses) / len(data)
    model.train()
    return out


def agree_any(flag: bool, run: RunConfig) -> bool:
    """True on every rank if the flag is True on any rank (keeps ranks in lockstep)."""
    if run.world_size == 1:
        return flag
    import torch.distributed as dist

    t = torch.tensor([1.0 if flag else 0.0], device=run.device)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return t.item() > 0


def _mean_across_ranks(value: float, run: RunConfig) -> float:
    if run.world_size == 1:
        return value
    import torch.distributed as dist

    t = torch.tensor([value], device=run.device)
    dist.all_reduce(t)
    return t.item() / run.world_size


def train(model_cfg: config.ModelConfig, run: RunConfig, sampler: BatchSampler, reader: WindowReader,
          ckpt_path: str, metrics_path: str | None = None, val_sets: Mapping | None = None,
          budget: BudgetGuard | None = None) -> dict:
    torch.manual_seed(run.seed)
    device = run.device
    raw = build_model(model_cfg).to(device)
    opt = make_optimizer(raw, run.lr, run.weight_decay, run.betas, fused=device.startswith("cuda"))
    start = 0
    ck = load_checkpoint(ckpt_path)
    if ck is not None:
        raw.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        start = ck["step"]
    model = raw
    if run.world_size > 1:
        from torch.nn.parallel import DistributedDataParallel as DDP

        model = DDP(raw, device_ids=[torch.device(device).index])
    if run.compile:
        model = torch.compile(model)

    local = run.micro_batch * run.accum_steps
    seq = reader.seq_len
    history: list[dict] = []
    t_start = time.time()
    stopped_by_budget = False
    use_amp = device.startswith("cuda")
    model.train()
    step = start
    while step < run.total_steps:
        t0 = time.time()
        lr = lr_at(step, run.total_steps, run.warmup_steps, run.lr, run.min_lr, run.decay_frac)
        for g in opt.param_groups:
            g["lr"] = lr
        picks = sampler.sample(step, local * run.world_size)[run.rank * local:(run.rank + 1) * local]
        opt.zero_grad(set_to_none=True)
        loss_sum = 0.0
        for a in range(run.accum_steps):
            batch = reader.batch(picks[a * run.micro_batch:(a + 1) * run.micro_batch]).to(device)
            sync = run.world_size == 1 or a == run.accum_steps - 1
            ctx = model.no_sync() if (not sync and hasattr(model, "no_sync")) else _nullcontext()
            with ctx, torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                loss = model(input_ids=batch, labels=batch).loss / run.accum_steps
            loss.backward()
            loss_sum += loss.item()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(raw.parameters(), run.grad_clip))
        opt.step()
        step += 1
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        dt = time.time() - t0

        want_eval = bool(val_sets) and run.eval_every > 0 and (step % run.eval_every == 0 or step == run.total_steps)
        if step % run.log_every == 0 or want_eval or step == run.total_steps or step == start + 1:
            entry = {"step": step, "loss": _mean_across_ranks(loss_sum, run), "lr": lr,
                     "grad_norm": grad_norm,
                     "tokens_per_sec": local * run.world_size * seq / dt,
                     "elapsed_s": time.time() - t_start}
            if want_eval and run.rank == 0:
                entry["val_loss"] = evaluate(raw, val_sets, device, run.micro_batch)
            history.append(entry)
            if run.rank == 0 and metrics_path:
                os.makedirs(os.path.dirname(metrics_path) or ".", exist_ok=True)
                with open(metrics_path, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(entry) + "\n")
        over_budget = agree_any(budget is not None and budget.exceeded(time.time() - t_start), run)
        if run.rank == 0 and step == int(run.total_steps * (1.0 - run.decay_frac)):
            save_checkpoint(ckpt_path + ".predecay", raw, opt, step)   # end of the stable phase
        done = step == run.total_steps or (run.stop_after is not None and step >= run.stop_after)
        if run.rank == 0 and (step % run.ckpt_every == 0 or done or over_budget):
            save_checkpoint(ckpt_path, raw, opt, step)
        if over_budget:
            stopped_by_budget = True
            break
        if done:
            break
    return {"history": history, "stopped_by_budget": stopped_by_budget, "final_step": step,
            "model": raw}


class _nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False
