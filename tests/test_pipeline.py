"""Tests for pipeline.py: pure helpers used by the Modal clean workers."""

import pytest

from viki_slm_125m.data import pipeline


def test_worker_budget_splits_evenly():
    assert pipeline.worker_budget(1_000_000, 4) == 250_000
    assert pipeline.worker_budget(10, 3) == 4  # rounds up so the total is never short
    with pytest.raises(ValueError):
        pipeline.worker_budget(100, 0)


def test_owns_row_partitions_exactly():
    owned = [[i for i in range(20) if pipeline.owns_row(i, k, 4)] for k in range(4)]
    assert sorted(x for part in owned for x in part) == list(range(20))
    assert all(len(p) == 5 for p in owned)


def test_owns_row_validates():
    with pytest.raises(ValueError):
        pipeline.owns_row(0, 4, 4)


def _row(**kw):
    base = {"license_type": "permissive", "int_score": 4, "length_bytes": 2_000,
            "blob_id": "x", "src_encoding": "UTF-8"}
    return {**base, **kw}


def test_accept_code_row_filters():
    assert pipeline.accept_code_row(_row())
    assert not pipeline.accept_code_row(_row(license_type="no_license"))
    assert not pipeline.accept_code_row(_row(length_bytes=50))
    assert not pipeline.accept_code_row(_row(length_bytes=500_000))
    assert not pipeline.accept_code_row(_row(int_score=2), min_score=3)
    assert pipeline.accept_code_row(_row(int_score=3), min_score=3)


def test_est_tokens_uses_chars_per_token():
    assert pipeline.est_tokens(4_000, 4.0) == 1_000
    assert pipeline.est_tokens(3_300, 3.3) == 1_000


def test_merge_reports_sums_counts_and_reasons():
    a = {"streamed": 10, "kept": 6, "est_tokens": 100, "reasons": {"kept": 6, "too_short": 4}}
    b = {"streamed": 5, "kept": 2, "est_tokens": 50, "reasons": {"kept": 2, "syntax_error": 3}}
    m = pipeline.merge_reports([a, b])
    assert m == {"streamed": 15, "kept": 8, "est_tokens": 150,
                 "reasons": {"kept": 8, "too_short": 4, "syntax_error": 3}}
    assert pipeline.merge_reports([]) == {"streamed": 0, "kept": 0, "est_tokens": 0, "reasons": {}}


def test_decode_text_handles_encodings_and_garbage():
    assert pipeline.decode_text("héllo".encode("latin-1"), "ISO-8859-1") == "héllo"
    assert pipeline.decode_text(b"abc", "not-a-codec") == "abc"
    assert pipeline.decode_text(b"abc", None) == "abc"


def test_copy_tree_parallel_copies_nested_files(tmp_path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    (src / "train").mkdir(parents=True)
    (src / "val").mkdir()
    (src / "train" / "a.bin").write_bytes(b"aaaa")
    (src / "val" / "b.bin").write_bytes(b"bb")
    (src / "index.json").write_text("{}")
    n = pipeline.copy_tree_parallel(str(src), str(dst), workers=4)
    assert n == 3
    assert (dst / "train" / "a.bin").read_bytes() == b"aaaa"
    assert (dst / "val" / "b.bin").read_bytes() == b"bb"
    assert (dst / "index.json").read_text() == "{}"


def test_copy_tree_parallel_missing_source_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        pipeline.copy_tree_parallel(str(tmp_path / "nope"), str(tmp_path / "d"), workers=2)
