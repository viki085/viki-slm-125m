"""Tests for train_ddp.py argument handling (the GPU path is exercised by the Modal trial)."""

import pytest

from viki_slm_125m.pretrain import train_ddp


def test_parse_args_requires_core_arguments():
    with pytest.raises(SystemExit):
        train_ddp.parse_args([])


def test_parse_args_defaults_and_flags():
    a = train_ddp.parse_args(["--tokens-dir", "/t", "--ckpt-dir", "/c", "--total-steps", "200",
                              "--warmup-steps", "20", "--cap-usd", "6", "--no-compile"])
    assert a.micro_batch == 32 and a.no_compile and a.cap_usd == 6.0
    assert a.total_steps == 200 and a.ckpt_every == 500
