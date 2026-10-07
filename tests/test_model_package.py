"""Tests for viki_slm_125m.model: architecture, checkpoint loading, generation and the VikiSLM facade."""

import pytest
import torch

from viki_slm_125m import config
from viki_slm_125m.data import tokenizer_lib as tl
from viki_slm_125m.model import architecture, checkpoint, generation
from viki_slm_125m.model.assistant import VikiSLM
from viki_slm_125m.pretrain import trainlib


@pytest.fixture(scope="module")
def tok():
    return tl.build_tokenizer(iter(["select count from table where name"] * 80), vocab_size=300,
                              specials=config.all_special_tokens())


@pytest.fixture()
def tiny_cfg(tok):
    return config.ModelConfig(vocab_size=tok.get_vocab_size(), hidden_size=32, intermediate_size=64,
                              num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
                              max_position_embeddings=128)


def test_full_model_has_the_documented_parameter_count():
    model = architecture.build_model()
    assert architecture.count_parameters(model) == 124_275_456
    assert architecture.count_parameters(model) == config.MODEL.approx_params()


def test_tiny_model_forward_shape(tiny_cfg):
    model = architecture.build_model(tiny_cfg)
    out = model(input_ids=torch.randint(0, tiny_cfg.vocab_size, (2, 7)))
    assert out.logits.shape == (2, 7, tiny_cfg.vocab_size)


def test_trainlib_build_model_is_the_same_function():
    assert trainlib.build_model is architecture.build_model


def test_load_model_roundtrip_matches_the_saved_weights(tmp_path, tiny_cfg):
    torch.manual_seed(0)
    original = architecture.build_model(tiny_cfg).eval()
    path = tmp_path / "ckpt.pt"
    torch.save({"model": original.state_dict(), "step": 5}, path)
    loaded = checkpoint.load_model(str(path), "cpu", model_cfg=tiny_cfg, dtype=torch.float32)
    ids = torch.randint(0, tiny_cfg.vocab_size, (1, 6))
    assert torch.allclose(original(input_ids=ids).logits, loaded(input_ids=ids).logits)
    assert not loaded.training


def test_load_model_reports_a_missing_checkpoint(tmp_path, tiny_cfg):
    with pytest.raises(FileNotFoundError, match="checkpoint not found"):
        checkpoint.load_model(str(tmp_path / "nope.pt"), "cpu", model_cfg=tiny_cfg)


def test_generate_is_deterministic_and_keeps_prompt_order(tok, tiny_cfg):
    torch.manual_seed(0)
    model = architecture.build_model(tiny_cfg).eval()
    prompts = ["select count from table where name", "select", "name"]
    a = generation.generate(model, tok, prompts, ["<|eos|>"], 6, "cpu", batch=2)
    b = generation.generate(model, tok, prompts, ["<|eos|>"], 6, "cpu", batch=3)
    assert len(a) == 3 and a == b
    assert generation.generate(model, tok, [], ["<|eos|>"], 6, "cpu") == []


def test_generate_accepts_sampling_arguments(tok, tiny_cfg):
    model = architecture.build_model(tiny_cfg).eval()
    out = generation.generate(model, tok, ["select"] * 3, ["<|eos|>"], 5, "cpu", do_sample=True, temperature=0.8)
    assert len(out) == 3 and all(isinstance(t, str) for t in out)


def test_facade_chat_and_sql_return_text(tok, tiny_cfg):
    model = architecture.build_model(tiny_cfg).eval()
    slm = VikiSLM(model, tok, "cpu")
    assert isinstance(slm.chat("select count"), str)
    sql = slm.sql_for("CREATE TABLE t(a INT);", "how many rows?")
    assert sql is None or isinstance(sql, str)


def test_load_model_keeps_rotary_frequencies_in_float32(tmp_path, tiny_cfg):
    model = architecture.build_model(tiny_cfg)
    path = tmp_path / "ckpt.pt"
    torch.save({"model": model.state_dict()}, path)
    loaded = checkpoint.load_model(str(path), "cpu", model_cfg=tiny_cfg, dtype=torch.bfloat16)
    freqs = [b for n, b in loaded.named_buffers() if n.endswith("inv_freq")]
    assert freqs and all(b.dtype == torch.float32 for b in freqs)
    assert all(p.dtype == torch.bfloat16 for p in loaded.parameters())
    reference = [b for n, b in model.named_buffers() if n.endswith("inv_freq")]
    assert all(torch.equal(a, b) for a, b in zip(freqs, reference))
