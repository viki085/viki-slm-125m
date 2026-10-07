"""Tests for measure.py (pure helpers for Phase 1 feasibility)."""

import pytest

from viki_slm_125m.data import measure


def test_estimate_clean_tokens():
    assert measure.estimate_clean_tokens(1_000, 4_000.0, 4.0) == 1_000_000


def test_estimate_clean_tokens_rejects_bad_input():
    with pytest.raises(ValueError):
        measure.estimate_clean_tokens(10, 100.0, 0)
    with pytest.raises(ValueError):
        measure.estimate_clean_tokens(-1, 100.0, 4.0)


def test_license_from_card_prefers_card_data():
    info = {"cardData": {"license": "odc-by"}, "tags": ["license:mit"]}
    assert measure.license_from_card(info) == "odc-by"


def test_license_from_card_falls_back_to_tags_and_lists():
    assert measure.license_from_card({"tags": ["x", "license:apache-2.0"]}) == "apache-2.0"
    assert measure.license_from_card({"cardData": {"license": ["mit", "cc0-1.0"]}}) == "mit,cc0-1.0"
    assert measure.license_from_card({}) is None


def test_summarize_stack_edu_permissive_share():
    rows = [
        ("permissive", 1_000, 4),
        ("permissive", 3_000, 3),
        ("no_license", 6_000, 4),
    ]
    s = measure.summarize_stack_edu(rows)
    assert s["files"] == 3
    assert s["permissive_files"] == 2
    assert s["permissive_file_share"] == pytest.approx(2 / 3)
    assert s["permissive_byte_share"] == pytest.approx(0.4)
    assert s["permissive_int_score_4plus_files"] == 1


def test_summarize_stack_edu_empty():
    s = measure.summarize_stack_edu([])
    assert s["files"] == 0
    assert s["permissive_file_share"] == 0.0


def test_summarize_fetch_rates():
    s = measure.summarize_fetch(n_ok=90, n_missing=8, n_error=2, total_bytes=900_000, elapsed_s=10.0)
    assert s["files_per_sec"] == pytest.approx(10.0)
    assert s["failure_rate"] == pytest.approx(0.10)
    assert s["avg_bytes_ok"] == pytest.approx(10_000)


def test_summarize_fetch_zero_division_safe():
    s = measure.summarize_fetch(0, 0, 0, 0, 0.0)
    assert s["files_per_sec"] == 0.0
    assert s["failure_rate"] == 0.0


def test_project_fetch_seconds():
    # 1,000,000 files at 50 files/s/worker with 100 workers -> 200 s
    assert measure.project_fetch_seconds(1_000_000, 50.0, 100) == pytest.approx(200.0)
    with pytest.raises(ValueError):
        measure.project_fetch_seconds(10, 0.0, 4)
