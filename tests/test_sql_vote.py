"""Tests for sql_vote.py: choose a query by majority vote over execution results."""

import sqlite3

import pytest

from viki_slm_125m.sft import sql_vote as sv
from viki_slm_125m.sft.sql_verify import run_query


@pytest.fixture()
def conn():
    c = sqlite3.connect(":memory:")
    c.executescript("CREATE TABLE t(a INT, b TEXT); INSERT INTO t VALUES (1,'x'),(2,'y'),(3,'x');")
    yield c
    c.close()


def test_majority_result_wins_over_greedy(conn):
    cands = ["SELECT COUNT(*) FROM t WHERE a > 1",       # greedy: 2
             "SELECT COUNT(*) FROM t",                     # 3
             "SELECT COUNT(a) FROM t",                     # 3
             "SELECT COUNT(*) FROM t WHERE a >= 1"]        # 3
    choice = sv.vote(conn, cands)
    assert choice.sql in cands[1:] and choice.votes == 3 and choice.executed == 4


def test_failing_candidates_are_ignored(conn):
    choice = sv.vote(conn, ["SELECT nope FROM t", "SELECT COUNT(*) FROM t", "DELETE FROM t"])
    assert choice.sql == "SELECT COUNT(*) FROM t" and choice.votes == 1 and choice.executed == 1


def test_tie_prefers_the_earliest_candidate(conn):
    choice = sv.vote(conn, ["SELECT MAX(a) FROM t", "SELECT MIN(a) FROM t"])
    assert choice.sql == "SELECT MAX(a) FROM t"


def test_same_rows_in_any_order_or_column_name_count_as_one_vote(conn):
    choice = sv.vote(conn, ["SELECT b FROM t ORDER BY a", "SELECT b AS letter FROM t ORDER BY a DESC",
                            "SELECT COUNT(*) FROM t"])
    assert choice.sql.startswith("SELECT b") and choice.votes == 2


def test_no_executable_candidate_returns_none(conn):
    assert sv.vote(conn, ["SELECT nope FROM t", "not sql"]) is None
    assert sv.vote(conn, []) is None
    assert sv.vote(conn, [None, ""]) is None
