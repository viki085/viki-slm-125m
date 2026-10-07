"""Tests for model/export.py: write a Hugging Face model folder and load it back."""

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from viki_slm_125m import config
from viki_slm_125m.data import tokenizer_lib as tl
from viki_slm_125m.model import architecture, export


@pytest.fixture(scope="module")
def tok():
    return tl.build_tokenizer(iter(["select count from table where name"] * 80), vocab_size=300,
                              specials=config.all_special_tokens())


@pytest.fixture()
def exported(tmp_path, tok):
    cfg = config.ModelConfig(vocab_size=tok.get_vocab_size(), hidden_size=32, intermediate_size=64,
                             num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
                             max_position_embeddings=128)
    torch.manual_seed(0)
    model = architecture.build_model(cfg).eval()
    ckpt, tok_path, out = tmp_path / "ckpt.pt", tmp_path / "tokenizer.json", tmp_path / "model"
    torch.save({"model": model.state_dict(), "step": 9}, ckpt)
    tok.save(str(tok_path))
    export.export_hf(str(ckpt), str(tok_path), str(out), model_cfg=cfg, dtype=torch.float32)
    return model, out, cfg


def test_export_writes_the_expected_files(exported):
    _, out, _ = exported
    names = {p.name for p in out.iterdir()}
    assert {"config.json", "generation_config.json", "model.safetensors", "tokenizer.json",
            "tokenizer_config.json", "special_tokens_map.json", "README.md"} <= names


def test_exported_model_loads_with_from_pretrained_and_matches(exported):
    model, out, cfg = exported
    loaded = AutoModelForCausalLM.from_pretrained(str(out), torch_dtype=torch.float32).eval()
    ids = torch.randint(0, cfg.vocab_size, (1, 8))
    assert torch.allclose(model(input_ids=ids).logits, loaded(input_ids=ids).logits, atol=1e-6)


def test_exported_tokenizer_matches_the_original_including_special_tokens(exported, tok):
    _, out, _ = exported
    fast = AutoTokenizer.from_pretrained(str(out))
    text = "<|bos|><|system|>hi<|user|>select count<|assistant|>"
    assert fast(text, add_special_tokens=False)["input_ids"] == tok.encode(text).ids
    assert fast.pad_token == "<|pad|>" and fast.eos_token == "<|eos|>" and fast.bos_token == "<|bos|>"


def test_generation_config_has_stop_and_pad_ids(exported, tok):
    _, out, _ = exported
    loaded = AutoModelForCausalLM.from_pretrained(str(out))
    assert loaded.generation_config.eos_token_id == tok.token_to_id("<|eos|>")
    assert loaded.generation_config.pad_token_id == tok.token_to_id("<|pad|>")


def test_model_card_documents_prompt_format_and_limits(exported):
    _, out, _ = exported
    card = (out / "README.md").read_text(encoding="utf-8")
    assert "<|schema|>" in card and "<|/sql|>" in card and "Limitations" in card


def test_export_rejects_a_missing_checkpoint(tmp_path, tok):
    with pytest.raises(FileNotFoundError):
        export.export_hf(str(tmp_path / "none.pt"), str(tmp_path / "t.json"), str(tmp_path / "o"))


def test_exported_tokenizer_returns_only_what_generate_accepts(exported):
    _, out, _ = exported
    enc = AutoTokenizer.from_pretrained(str(out))("<|bos|>select", return_tensors="pt", add_special_tokens=False)
    assert set(enc.keys()) == {"input_ids", "attention_mask"}
