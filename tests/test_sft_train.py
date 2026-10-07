"""Tests for sft_dataset.py and sft.py: batching, collation, masked loss, training on a tiny model."""

import json
import random

import pytest
import torch

from viki_slm_125m import config
from viki_slm_125m.sft import sft_dataset as sds
from viki_slm_125m.sft import train as sft_mod
from viki_slm_125m.data import tokenizer_lib as tl
from viki_slm_125m.pretrain import trainlib


def make_examples(n=40, vocab=60):
    rng = random.Random(0)
    out = []
    for _ in range(n):
        L = rng.randint(8, 24)
        ids = [rng.randrange(25, vocab) for _ in range(L)]
        # learnable pattern: trained tokens repeat the previous token
        ids = [ids[0]] + [ids[0]] * (L - 1)
        mask = [0] * (L // 2) + [1] * (L - L // 2)
        out.append(sds.Encoded(ids, mask))
    return out


# ---------- batching ----------

def test_token_budget_batches_respect_budget_and_cover_everything():
    lengths = [random.Random(i).randint(5, 100) for i in range(200)]
    batches = sds.token_budget_batches(lengths, max_tokens=400, rng=random.Random(0))
    seen = sorted(i for b in batches for i in b)
    assert seen == list(range(200))
    for b in batches:
        assert max(lengths[i] for i in b) * len(b) <= 400 or len(b) == 1


def test_token_budget_batches_group_similar_lengths():
    lengths = list(range(10, 210))
    batches = sds.token_budget_batches(lengths, max_tokens=600, rng=random.Random(1), bucket=200)
    pad_waste = sum(max(lengths[i] for i in b) * len(b) - sum(lengths[i] for i in b) for b in batches)
    assert pad_waste / sum(lengths) < 0.15


def test_token_budget_batches_is_deterministic():
    lengths = [random.Random(i).randint(5, 60) for i in range(100)]
    a = sds.token_budget_batches(lengths, 300, random.Random(2))
    b = sds.token_budget_batches(lengths, 300, random.Random(2))
    assert a == b


def test_collate_pads_and_masks_labels():
    exs = [sds.Encoded([5, 6, 7], [0, 1, 1]), sds.Encoded([8, 9], [1, 1])]
    b = sds.collate(exs, pad_id=0)
    assert b["input_ids"].tolist() == [[5, 6, 7], [8, 9, 0]]
    assert b["attention_mask"].tolist() == [[1, 1, 1], [1, 1, 0]]
    assert b["labels"].tolist() == [[-100, 6, 7], [8, 9, -100]]


# ---------- loading ----------

def test_example_from_record_honours_trained_flag():
    rec = {"turns": [{"role": "user", "text": "q"}, {"role": "assistant", "text": "a", "trained": False},
                     {"role": "tool", "text": "t"}, {"role": "assistant", "text": "final"}]}
    turns = sds.turns_from_record(rec)
    assert [t.trained for t in turns] == [True, False, True, True]


def test_load_mixture_caps_per_source_and_encodes(tmp_path):
    tok = tl.build_tokenizer(iter(["hello world data"] * 50), vocab_size=300, specials=config.all_special_tokens())
    enc = lambda t: (tok.encode(t).ids, tok.encode(t).offsets)  # noqa: E731
    path = tmp_path / "a.jsonl"
    with open(path, "w") as fh:
        for i in range(10):
            fh.write(json.dumps({"id": str(i), "turns": [{"role": "user", "text": "hello"},
                                                         {"role": "assistant", "text": "world data"}]}) + "\n")
    exs = sds.load_mixture({"a": (str(path), 4)}, enc, max_len=64, seed=0)
    assert len(exs["a"]) == 4 and all(sum(e.mask) > 0 for e in exs["a"])


# ---------- masked loss ----------

def test_masked_loss_ignores_unlabelled_tokens():
    torch.manual_seed(0)
    model = trainlib.build_model(trainlib.tiny_model_config())
    a = sds.collate([sds.Encoded([5, 6, 7, 8], [0, 0, 1, 1])], 0)
    b = sds.collate([sds.Encoded([9, 9, 7, 8], [0, 0, 1, 1])], 0)  # different masked prefix tokens
    la, na = sft_mod.masked_loss_sum(model, a)
    assert na == 2 and la.item() > 0
    # changing a masked label position's *label* does not change the summed loss
    c = dict(a)
    c["labels"] = a["labels"].clone()
    c["labels"][0, 1] = -100
    lc, nc = sft_mod.masked_loss_sum(model, c)
    assert nc == 2 and lc.item() == pytest.approx(la.item())
    assert b["labels"][0, 2].item() == 7


# ---------- training ----------

def _cfg(tmp_path, **kw):
    base = dict(lr=3e-3, epochs=4, micro_tokens=256, accum_steps=2, warmup_frac=0.1, weight_decay=0.0,
                grad_clip=1.0, eval_every=5, ckpt_dir=str(tmp_path), seed=0, device="cpu", decay_frac=0.3)
    base.update(kw)
    return sft_mod.SftConfig(**base)


def test_sft_training_reduces_loss_and_saves_best(tmp_path):
    train = make_examples(60)
    val = {"v": make_examples(10)}
    out = sft_mod.train_sft(trainlib.tiny_model_config(), _cfg(tmp_path, epochs=16), train, val, pad_id=0)
    losses = [h["loss"] for h in out["history"]]
    assert losses[-1] < losses[0] * 0.8
    assert (tmp_path / "best.pt").exists()
    evals = [h for h in out["history"] if "val_loss" in h]
    assert evals and "v" in evals[0]["val_loss"]


def test_sft_loads_base_weights_when_given(tmp_path):
    cfgm = trainlib.tiny_model_config()
    base = trainlib.build_model(cfgm)
    path = tmp_path / "base.pt"
    torch.save({"model": base.state_dict()}, path)
    state = sft_mod.load_base_weights(str(path))
    assert set(state) == set(base.state_dict())


def test_sft_resume_continues_from_saved_step(tmp_path):
    train = make_examples(60)
    cfg = _cfg(tmp_path, epochs=2)
    first = sft_mod.train_sft(trainlib.tiny_model_config(), cfg, train, {}, pad_id=0, stop_after=3)
    assert first["final_step"] == 3
    second = sft_mod.train_sft(trainlib.tiny_model_config(), cfg, train, {}, pad_id=0)
    assert second["history"][0]["step"] == 4
