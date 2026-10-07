"""Tests for trainlib.py: schedule, sampler, model build, checkpoints, training loop (CPU, tiny)."""

import json
import math

import numpy as np
import pytest
import torch

from viki_slm_125m import config
from viki_slm_125m.pretrain import trainlib as tl


# ---------- learning-rate schedule (warmup-stable-decay) ----------

def test_lr_warms_up_holds_then_decays_to_min():
    kw = dict(total_steps=1000, warmup_steps=100, lr=1e-3, min_lr=1e-4, decay_frac=0.2)
    assert tl.lr_at(0, **kw) == pytest.approx(1e-3 / 100)
    assert tl.lr_at(99, **kw) == pytest.approx(1e-3)
    assert tl.lr_at(500, **kw) == pytest.approx(1e-3)
    assert tl.lr_at(799, **kw) == pytest.approx(1e-3)
    assert tl.lr_at(999, **kw) == pytest.approx(1e-4, rel=0.01)
    decay = [tl.lr_at(s, **kw) for s in range(800, 1000)]
    assert all(a >= b for a, b in zip(decay, decay[1:]))
    assert min(decay) >= 1e-4 - 1e-12


def test_lr_rejects_bad_arguments():
    with pytest.raises(ValueError):
        tl.lr_at(0, total_steps=0, warmup_steps=1, lr=1e-3, min_lr=1e-4)
    with pytest.raises(ValueError):
        tl.lr_at(0, total_steps=100, warmup_steps=100, lr=1e-3, min_lr=1e-4, decay_frac=0.5)


# ---------- sampler ----------

def _refs():
    return [tl.WindowRef("a", "/x/a.bin", 1000, 1000.0),
            tl.WindowRef("b", "/x/b.bin", 1000, 3000.0)]  # b is weighted 3x (epochs)


def test_sampler_is_deterministic_per_step_and_differs_across_steps():
    s = tl.BatchSampler(_refs(), seed=1)
    assert s.sample(5, 64) == s.sample(5, 64)
    assert s.sample(5, 64) != s.sample(6, 64)
    assert tl.BatchSampler(_refs(), seed=2).sample(5, 64) != s.sample(5, 64)


def test_sampler_respects_weights_and_bounds():
    s = tl.BatchSampler(_refs(), seed=0)
    picks = [p for step in range(200) for p in s.sample(step, 64)]
    share_b = sum(1 for f, _ in picks if f == 1) / len(picks)
    assert 0.72 < share_b < 0.78
    assert all(0 <= w < 1000 for _, w in picks)


def test_sampler_rejects_empty():
    with pytest.raises(ValueError):
        tl.BatchSampler([], seed=0)


# ---------- window reader ----------

def test_reader_returns_requested_windows(tmp_path):
    data = np.arange(3 * 8, dtype=np.uint16)
    p = tmp_path / "w.bin"
    data.tofile(p)
    r = tl.WindowReader([str(p)], seq_len=8)
    assert r.get(0, 1).tolist() == list(range(8, 16))
    batch = r.batch([(0, 0), (0, 2)])
    assert batch.shape == (2, 8) and batch.dtype == torch.long
    assert batch[1].tolist() == list(range(16, 24))


# ---------- model ----------

def test_full_model_param_count_matches_config():
    with torch.device("meta"):
        model = tl.build_model(config.MODEL)
    n = sum(p.numel() for p in model.parameters())
    assert n == pytest.approx(config.MODEL.approx_params(), rel=0.001)


def test_optimizer_excludes_norms_and_biases_from_decay():
    model = tl.build_model(tl.tiny_model_config())
    opt = tl.make_optimizer(model, lr=1e-3, weight_decay=0.1, betas=(0.9, 0.95), fused=False)
    decay, no_decay = opt.param_groups
    assert decay["weight_decay"] == 0.1 and no_decay["weight_decay"] == 0.0
    assert all(p.ndim >= 2 for p in decay["params"])
    assert all(p.ndim < 2 for p in no_decay["params"])
    assert len(decay["params"]) + len(no_decay["params"]) == len(list(model.parameters()))


# ---------- budget guard ----------

def test_budget_guard():
    g = tl.BudgetGuard(rate_per_gpu_hour=4.0, n_gpus=8, cap_usd=10.0)
    assert g.spent(3600) == pytest.approx(32.0)
    assert not g.exceeded(1000)      # $8.89
    assert g.exceeded(1200)          # $10.67
    with pytest.raises(ValueError):
        tl.BudgetGuard(4.0, 0, 10.0)


# ---------- training loop on a tiny model ----------

def _tiny_setup(tmp_path, steps=40):
    cfg = tl.tiny_model_config()
    rng = np.random.default_rng(0)
    # learnable pattern: every window repeats the same short cycle
    cycle = np.arange(16, dtype=np.uint16)
    wins = np.tile(cycle, (200, 4))            # 200 windows of 64 tokens
    p = tmp_path / "t.bin"
    wins.tofile(p)
    reader = tl.WindowReader([str(p)], seq_len=64)
    sampler = tl.BatchSampler([tl.WindowRef("t", str(p), 200, 200.0)], seed=0)
    run = tl.RunConfig(total_steps=steps, micro_batch=4, accum_steps=2, lr=3e-3, min_lr=3e-4,
                       warmup_steps=5, weight_decay=0.0, ckpt_every=10, eval_every=0,
                       log_every=1, seed=0, device="cpu", compile=False)
    return cfg, reader, sampler, run


def test_training_reduces_loss_and_logs_metrics(tmp_path):
    cfg, reader, sampler, run = _tiny_setup(tmp_path)
    metrics = tmp_path / "m.jsonl"
    out = tl.train(cfg, run, sampler, reader, ckpt_path=str(tmp_path / "ck.pt"),
                   metrics_path=str(metrics))
    losses = [m["loss"] for m in out["history"]]
    assert losses[-1] < losses[0] * 0.6
    rows = [json.loads(l) for l in metrics.read_text().splitlines()]
    assert rows and {"step", "loss", "lr", "tokens_per_sec"} <= set(rows[0])


def test_resume_matches_uninterrupted_run(tmp_path):
    cfg, reader, sampler, run = _tiny_setup(tmp_path, steps=20)
    full = tl.train(cfg, run, sampler, reader, ckpt_path=str(tmp_path / "full.pt"))
    # interrupted: run 10 steps, then resume to 20 from the checkpoint
    first = tl.RunConfig(**{**run.__dict__, "total_steps": 20, "stop_after": 10})
    tl.train(cfg, first, sampler, reader, ckpt_path=str(tmp_path / "part.pt"))
    resumed = tl.train(cfg, run, sampler, reader, ckpt_path=str(tmp_path / "part.pt"))
    assert resumed["history"][0]["step"] == 11
    assert resumed["history"][-1]["loss"] == pytest.approx(full["history"][-1]["loss"], rel=1e-4)


def test_budget_cap_stops_training_and_saves_checkpoint(tmp_path):
    cfg, reader, sampler, run = _tiny_setup(tmp_path, steps=40)
    guard = tl.BudgetGuard(rate_per_gpu_hour=3600.0 * 1e3, n_gpus=1, cap_usd=1.0)  # trips immediately
    ck = tmp_path / "cap.pt"
    out = tl.train(cfg, run, sampler, reader, ckpt_path=str(ck), budget=guard)
    assert out["stopped_by_budget"] and ck.exists()
    assert len(out["history"]) < 40


def test_eval_reports_per_source_loss(tmp_path):
    cfg, reader, sampler, run = _tiny_setup(tmp_path, steps=10)
    val = {"t": reader.batch([(0, i) for i in range(4)])}
    run_eval = tl.RunConfig(**{**run.__dict__, "eval_every": 5})
    out = tl.train(cfg, run_eval, sampler, reader, ckpt_path=str(tmp_path / "e.pt"), val_sets=val)
    evals = [m for m in out["history"] if "val_loss" in m]
    assert evals and "t" in evals[0]["val_loss"]


def test_load_checkpoint_missing_returns_none(tmp_path):
    assert tl.load_checkpoint(str(tmp_path / "nope.pt")) is None


def test_refs_from_index_weights_by_epochs_and_skips_empty():
    index = {"shards": [
        {"source": "sql", "file": "shard-000-00.jsonl", "train_windows": 10, "val_windows": 1, "epochs": 2.0},
        {"source": "sql", "file": "shard-000-01.jsonl", "train_windows": 0, "val_windows": 0, "epochs": 2.0},
        {"source": "python", "file": "shard-001-00.jsonl", "train_windows": 5, "val_windows": 1, "epochs": 1.0}]}
    refs = tl.refs_from_index(index, "/t/train")
    assert [(r.source, r.weight) for r in refs] == [("sql", 20.0), ("python", 5.0)]
    assert refs[0].path == "/t/train/sql-shard-000-00.bin"


def test_load_val_sets_takes_first_windows_per_source(tmp_path):
    for name in ("python-shard-000-00", "python-shard-000-01"):
        np.arange(3 * 8, dtype=np.uint16).tofile(tmp_path / f"{name}.bin")
    index = {"shards": [
        {"source": "python", "file": "shard-000-00.jsonl", "val_windows": 3},
        {"source": "python", "file": "shard-000-01.jsonl", "val_windows": 3}]}
    val = tl.load_val_sets(index, str(tmp_path), seq_len=8, per_source=4)
    assert val["python"].shape == (4, 8)


def test_predecay_snapshot_is_saved_at_end_of_stable_phase(tmp_path):
    cfg, reader, sampler, run = _tiny_setup(tmp_path, steps=20)
    run = tl.RunConfig(**{**run.__dict__, "decay_frac": 0.2, "ckpt_every": 1000})
    ck = tmp_path / "ck.pt"
    tl.train(cfg, run, sampler, reader, ckpt_path=str(ck))
    snap = tl.load_checkpoint(str(ck) + ".predecay")
    assert snap is not None and snap["step"] == 16   # int(20 * 0.8)


def test_resume_from_predecay_snapshot_with_more_steps_continues_stable_lr(tmp_path):
    cfg, reader, sampler, run = _tiny_setup(tmp_path, steps=20)
    run = tl.RunConfig(**{**run.__dict__, "decay_frac": 0.2, "ckpt_every": 1000})
    ck = tmp_path / "ck.pt"
    tl.train(cfg, run, sampler, reader, ckpt_path=str(ck))
    (tmp_path / "ck2.pt").write_bytes((tmp_path / "ck.pt.predecay").read_bytes())
    longer = tl.RunConfig(**{**run.__dict__, "total_steps": 40})
    out = tl.train(cfg, longer, sampler, reader, ckpt_path=str(tmp_path / "ck2.pt"))
    first = out["history"][0]
    assert first["step"] == 17 and first["lr"] == pytest.approx(run.lr)   # no re-warmup, no decay yet


def test_agree_is_identity_for_single_process():
    run = tl.RunConfig(total_steps=1, micro_batch=1, accum_steps=1, lr=1e-3, min_lr=1e-4,
                       warmup_steps=0, device="cpu", compile=False)
    assert tl.agree_any(True, run) is True
    assert tl.agree_any(False, run) is False
