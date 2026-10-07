"""Tests for evaluator.py: prompt building, block extraction, execution matching, summaries."""

import sqlite3

import pytest

from viki_slm_125m.eval import evaluator as ev


@pytest.fixture()
def conn():
    c = sqlite3.connect(":memory:")
    c.executescript("CREATE TABLE t(a INT, b TEXT); INSERT INTO t VALUES (1,'x'),(2,'y'),(3,'x');")
    yield c
    c.close()


def test_build_prompt_ends_with_assistant_token():
    p = ev.build_prompt("SYS", "USER")
    assert p == "<|bos|><|system|>SYS<|user|>USER<|assistant|>"


def test_extract_block_handles_think_prefix_and_missing_tags():
    text = "<|think|>\nPlan.\n<|/think|>\n<|sql|>\nSELECT 1\n<|/sql|>"
    assert ev.extract_block(text, "sql") == "SELECT 1"
    assert ev.extract_block("<|sql|>\nSELECT 1", "sql") is None          # not closed
    assert ev.extract_block("no tags", "sql") is None


def test_judge_sql_statuses(conn):
    gold = "SELECT b, COUNT(*) FROM t GROUP BY b"
    assert ev.judge_sql(None, gold, conn)[0] == "no_sql"
    assert ev.judge_sql("SELECT nope FROM t", gold, conn)[0] == "error"
    assert ev.judge_sql("SELECT b, COUNT(*) FROM t WHERE a > 1 GROUP BY b", gold, conn)[0] == "mismatch"
    assert ev.judge_sql("SELECT b, COUNT(*) AS n FROM t GROUP BY b ORDER BY n DESC", gold, conn)[0] == "match"
    assert ev.judge_sql("DELETE FROM t", gold, conn)[0] == "error"


def test_summarize_rates():
    recs = [{"status": "match", "format_ok": True, "insight_ok": True, "faithful": True},
            {"status": "mismatch", "format_ok": True, "insight_ok": True, "faithful": False},
            {"status": "no_sql", "format_ok": False, "insight_ok": False, "faithful": None},
            {"status": "error", "format_ok": True, "insight_ok": None, "faithful": None}]
    s = ev.summarize(recs)
    assert s["n"] == 4 and s["execution_accuracy"] == pytest.approx(0.25)
    assert s["format_rate"] == pytest.approx(0.75) and s["executes_rate"] == pytest.approx(0.5)
    assert s["insight_format_rate"] == pytest.approx(2 / 3)       # among items that reached the insight step
    assert s["faithful_rate"] == pytest.approx(0.5)


def test_summarize_empty():
    assert ev.summarize([])["n"] == 0


def test_judge_sql_reports_rounding_only_differences(conn):
    gold = "SELECT ROUND(AVG(a), 1) FROM t"
    assert ev.judge_sql("SELECT ROUND(AVG(a), 3) FROM t", gold, conn)[0] == "match"      # same after 4-dp norm? 2.0 == 2.0
    gold2 = "SELECT ROUND(1.0 * SUM(a) / 7, 1) FROM t"                                    # 0.9
    assert ev.judge_sql("SELECT ROUND(1.0 * SUM(a) / 7, 2) FROM t", gold2, conn)[0] == "match_rounding"
    assert ev.judge_sql("SELECT ROUND(1.0 * SUM(a) / 3, 2) FROM t", gold2, conn)[0] == "mismatch"


def test_summarize_counts_rounding_matches_only_in_the_tolerant_rate():
    recs = [{"status": "match", "format_ok": True, "insight_ok": None, "faithful": None},
            {"status": "match_rounding", "format_ok": True, "insight_ok": None, "faithful": None},
            {"status": "mismatch", "format_ok": True, "insight_ok": None, "faithful": None},
            {"status": "error", "format_ok": True, "insight_ok": None, "faithful": None}]
    s = ev.summarize(recs)
    assert s["execution_accuracy"] == pytest.approx(0.25)
    assert s["execution_accuracy_tolerant"] == pytest.approx(0.5)
    assert s["executes_rate"] == pytest.approx(0.75)
