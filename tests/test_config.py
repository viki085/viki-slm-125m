"""Tests for config.py (written first; see guide section 7.1)."""

import dataclasses

import pytest

from viki_slm_125m import config


def test_param_count_is_about_124_3m():
    params = config.MODEL.approx_params()
    assert 124_000_000 <= params <= 125_000_000


def test_model_dimensions_are_consistent():
    m = config.MODEL
    assert m.hidden_size % m.num_attention_heads == 0
    assert m.num_attention_heads % m.num_key_value_heads == 0
    assert m.max_position_embeddings == config.SEQ_LEN == 2_048
    assert m.vocab_size == 32_768
    assert m.vocab_size <= 65_536  # token ids must fit uint16


def test_llama_kwargs_cover_every_model_field():
    kwargs = config.MODEL.to_llama_kwargs()
    assert set(kwargs) == {f.name for f in dataclasses.fields(config.MODEL)}


def test_configs_are_frozen():
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.MODEL.hidden_size = 1  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.TRAIN.lr = 1.0  # type: ignore[misc]


def test_special_tokens_are_unique_and_complete():
    all_tokens = config.all_special_tokens()
    assert len(all_tokens) == len(set(all_tokens))
    for tok in ("<|bos|>", "<|eos|>", "<|pad|>", "<|unk|>",
                "<|user|>", "<|assistant|>", "<|system|>",
                "<|sql|>", "<|/sql|>", "<|python|>", "<|/python|>",
                "<|result|>", "<|output|>", "<|insight|>", "<|think|>"):
        assert tok in all_tokens
    assert len(all_tokens) < 64


def test_agent_tokens_come_in_open_close_pairs():
    opens = [t for t in config.AGENT_TOKENS if not t.startswith("<|/")]
    closes = {t for t in config.AGENT_TOKENS if t.startswith("<|/")}
    assert len(opens) == len(closes)
    for t in opens:
        assert t.replace("<|", "<|/", 1) in closes


def test_paths_live_under_data_root():
    for path in (config.CLEAN_DIR, config.CORPUS_DIR, config.TOKENIZER_DIR,
                 config.TOKENS_DIR, config.SFT_DIR, config.CKPT_DIR,
                 config.EVAL_DIR, config.RAW_META_DIR, config.RAW_TEXT_DIR):
        assert path.startswith(config.DATA_ROOT + "/")
    assert config.TRAIN_TOKENS_DIR.startswith(config.TOKENS_DIR)
    assert config.VAL_TOKENS_DIR.startswith(config.TOKENS_DIR)


def test_data_mix_names_are_unique_and_valid():
    names = [s.name for s in config.DATA_MIX]
    assert len(names) == len(set(names))
    for s in config.DATA_MIX:
        assert s.kind in {"text", "light", "code", "sql", "notebook"}
        assert s.token_budget > 0
        assert s.epochs >= 1.0
        assert s.hf_id and "/" in s.hf_id


def test_data_mix_total_is_within_target():
    total = config.planned_tokens()
    assert total <= config.TARGET_TOKENS
    # active + pending slices together should reach the planned budget
    assert config.planned_tokens(include_pending=True) == pytest.approx(
        config.TARGET_TOKENS, rel=0.02)


def test_pending_sources_are_named_and_not_in_active_mix():
    active = {s.name for s in config.DATA_MIX}
    for name, budget in config.PENDING_SOURCES.items():
        assert name not in active
        assert budget > 0


def test_stack_edu_slices_require_permissive_license_and_fetch():
    code = [s for s in config.DATA_MIX if s.hf_id == "HuggingFaceTB/stack-edu"]
    assert {s.config_name for s in code} == {"Python", "SQL"}
    for s in code:
        assert s.needs_fetch is True
        assert s.license_filter == "permissive"


def test_budget_cap_is_positive_and_ordered():
    assert 0 < config.TRAIN_BUDGET_CAP_USD <= config.BUDGET_CAP_USD


def test_clean_thresholds_sane():
    c = config.CLEAN
    assert 0 < c.max_nonalnum_ratio < 1
    assert 0 < c.max_repetition_ratio < 1
    assert c.code_min_chars < c.code_max_chars
