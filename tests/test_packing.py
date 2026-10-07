"""Tests for packing.py: window packing, train/val routing and the token index (written first)."""

import numpy as np
import pytest

from viki_slm_125m.data import packing


def test_pack_stream_concatenates_with_eos_and_drops_tail():
    docs = [[1, 2, 3], [4, 5], [6, 7, 8, 9]]
    wins = list(packing.pack_stream(docs, eos_id=0, seq_len=4))
    # stream: 1 2 3 0 | 4 5 0 6 | 7 8 9 0 -> three full windows, nothing left over
    assert [w.tolist() for w in wins] == [[1, 2, 3, 0], [4, 5, 0, 6], [7, 8, 9, 0]]


def test_pack_stream_drops_incomplete_final_window():
    wins = list(packing.pack_stream([[1, 2, 3, 4, 5]], eos_id=0, seq_len=4))
    assert [w.tolist() for w in wins] == [[1, 2, 3, 4]]


def test_pack_stream_windows_are_uint16_and_fixed_length():
    wins = list(packing.pack_stream([list(range(1, 50))] * 3, eos_id=0, seq_len=16))
    assert wins and all(w.dtype == np.uint16 and w.shape == (16,) for w in wins)


def test_pack_stream_rejects_ids_that_do_not_fit_uint16():
    with pytest.raises(ValueError):
        list(packing.pack_stream([[70_000, 1, 2]], eos_id=0, seq_len=2))


def test_pack_stream_accepts_lazy_input_and_empty_docs():
    gen = (d for d in [[], [1, 2], []])
    assert [w.tolist() for w in packing.pack_stream(gen, eos_id=9, seq_len=3)] == [[9, 1, 2]]


def test_route_window_every_hundredth_goes_to_val():
    routes = [packing.route_window(i, 100) for i in range(201)]
    assert [i for i, r in enumerate(routes) if r == "val"] == [0, 100, 200]
    with pytest.raises(ValueError):
        packing.route_window(0, 0)


def test_split_windows_counts_match_routing():
    wins = [np.full(4, i, dtype=np.uint16) for i in range(250)]
    train, val = packing.split_windows(wins, val_every=100)
    assert len(val) == 3 and len(train) == 247
    assert {int(w[0]) for w in val} == {0, 100, 200}


def test_build_index_aggregates_sources_and_realized_mix():
    shards = [
        {"source": "python", "file": "a", "train_windows": 100, "val_windows": 1, "epochs": 1.0},
        {"source": "python", "file": "b", "train_windows": 100, "val_windows": 1, "epochs": 1.0},
        {"source": "sql", "file": "a", "train_windows": 100, "val_windows": 1, "epochs": 2.0},
    ]
    idx = packing.build_index(shards, seq_len=10, dtype="uint16")
    assert idx["train_windows"] == 300 and idx["val_windows"] == 3
    assert idx["train_tokens"] == 3000
    assert idx["sources"]["python"]["train_tokens"] == 2000
    assert idx["sources"]["sql"]["effective_train_tokens"] == 2000  # epochs applied
    mix = idx["realized_mix"]
    assert mix["python"] == pytest.approx(0.5) and mix["sql"] == pytest.approx(0.5)
    assert sum(mix.values()) == pytest.approx(1.0)


def test_build_index_empty():
    idx = packing.build_index([], seq_len=10, dtype="uint16")
    assert idx["train_windows"] == 0 and idx["realized_mix"] == {}
