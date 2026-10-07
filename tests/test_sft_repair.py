"""Tests for sft_repair.py: build SQL self-repair traces from verified queries."""

import random
import sqlite3

import pytest

from viki_slm_125m.sft import sft_repair as sr

GOOD = "SELECT region, SUM(amount) AS total FROM orders WHERE amount > 1 GROUP BY region"


@pytest.fixture()
def conn():
    c = sqlite3.connect(":memory:")
    c.executescript("CREATE TABLE orders(id INT, region TEXT, amount REAL);"
                    "INSERT INTO orders VALUES (1,'N',10.5),(2,'S',20),(3,'N',5.25);")
    yield c
    c.close()


def test_candidate_mutations_cover_three_kinds():
    cands = sr.candidate_mutations(GOOD)
    assert set(cands) == {"column", "table", "syntax"}
    assert all(sql != GOOD for sql in cands.values())
    assert "FORM" in cands["syntax"]


def test_column_and_table_mutations_produce_real_sqlite_errors(conn):
    from viki_slm_125m.sft.sql_verify import run_query
    cands = sr.candidate_mutations(GOOD)
    assert "no such column" in run_query(conn, cands["column"]).error
    assert "no such table" in run_query(conn, cands["table"]).error
    assert "syntax error" in run_query(conn, cands["syntax"]).error


def test_repair_trace_structure_and_masking(conn):
    turns = sr.make_repair_trace("CREATE TABLE orders(id INT, region TEXT, amount REAL);",
                                 "Total amount per region?", GOOD, conn, random.Random(0))
    assert [t.role for t in turns] == ["system", "user", "assistant", "tool", "assistant", "tool", "assistant"]
    assert turns[2].trained is False and turns[4].trained and turns[6].trained
    assert "Error:" in turns[3].text
    assert "<|think|>" in turns[4].text and "failed" in turns[4].text.lower()
    assert GOOD in turns[4].text
    assert "<|insight|>" in turns[6].text


def test_repair_trace_returns_none_when_good_query_fails(conn):
    assert sr.make_repair_trace("CREATE TABLE orders(id INT);", "q", "SELECT nope FROM orders",
                                conn, random.Random(0)) is None


def test_candidate_mutations_handle_queries_without_columns():
    cands = sr.candidate_mutations("SELECT 1")
    assert "syntax" not in cands or "FORM" not in cands.get("syntax", "")


# ---------- rewrite repairs (v4): the whole select list is wrong, not a one-character typo

def test_rewrite_mutation_uses_a_column_that_is_not_in_the_query_tables():
    from viki_slm_125m.sft.sql_verify import run_query
    c = sqlite3.connect(":memory:")
    c.executescript("CREATE TABLE suppliers(id INT, name TEXT, rating REAL);"
                    "CREATE TABLE inventory(sku TEXT, qty INT);"
                    "INSERT INTO suppliers VALUES (1,'a',4.5);")
    broken = sr.rewrite_mutation("SELECT name, rating FROM suppliers ORDER BY rating DESC LIMIT 5", c, random.Random(0))
    assert broken is not None and "FROM suppliers" in broken
    assert ("sku" in broken) or ("qty" in broken)
    assert "no such column" in run_query(c, broken).error


def test_rewrite_mutation_falls_back_when_the_database_has_one_table(conn):
    from viki_slm_125m.sft.sql_verify import run_query
    broken = sr.rewrite_mutation(GOOD, conn, random.Random(0))
    assert broken is not None and broken != GOOD
    assert "no such column" in run_query(conn, broken).error


def test_rewrite_mutation_returns_none_for_unparseable_sql(conn):
    assert sr.rewrite_mutation("this is not sql", conn, random.Random(0)) is None


def test_repair_trace_can_use_the_rewrite_kind(conn):
    kinds_seen = set()
    for seed in range(40):
        turns = sr.make_repair_trace("CREATE TABLE orders(id INT, region TEXT, amount REAL);", "totals?", GOOD, conn,
                                     random.Random(seed))
        assert turns is not None
        if "using only columns from the schema" in turns[4].text:
            kinds_seen.add("rewrite")
            assert turns[2].trained is False and "Error: no such column" in turns[3].text
    assert "rewrite" in kinds_seen
