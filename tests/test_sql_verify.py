"""Tests for sql_verify.py: read-only enforcement, safe execution, result formatting."""

import sqlite3

import pytest

from viki_slm_125m.sft import sql_verify as sv


@pytest.fixture()
def conn():
    c = sqlite3.connect(":memory:")
    c.executescript("""
        CREATE TABLE orders (id INTEGER PRIMARY KEY, region TEXT, amount REAL);
        INSERT INTO orders VALUES (1,'N',10.5),(2,'S',20.0),(3,'N',5.25),(4,'E',NULL);
    """)
    yield c
    c.close()


# ---------- read-only gate ----------

@pytest.mark.parametrize("sql", [
    "SELECT * FROM orders", "  select 1;", "WITH t AS (SELECT 1 a) SELECT a FROM t",
    "-- note\nSELECT region FROM orders"])
def test_read_only_accepts_selects(sql):
    assert sv.is_read_only(sql)


@pytest.mark.parametrize("sql", [
    "DROP TABLE orders", "DELETE FROM orders", "UPDATE orders SET amount=1",
    "INSERT INTO orders VALUES (9,'x',1)", "SELECT 1; DROP TABLE orders",
    "PRAGMA writable_schema=1", "ATTACH DATABASE 'x.db' AS x", "CREATE TABLE t(a)", ""])
def test_read_only_rejects_writes_and_multi_statements(sql):
    assert not sv.is_read_only(sql)


# ---------- execution ----------

def test_run_query_returns_columns_and_rows(conn):
    r = sv.run_query(conn, "SELECT region, SUM(amount) AS total FROM orders GROUP BY region ORDER BY region")
    assert r.ok and r.columns == ["region", "total"]
    assert r.rows == [("E", None), ("N", 15.75), ("S", 20.0)]


def test_run_query_reports_sql_errors_as_text(conn):
    r = sv.run_query(conn, "SELECT nope FROM orders")
    assert not r.ok and "no such column" in r.error


def test_run_query_refuses_writes_without_executing(conn):
    r = sv.run_query(conn, "DELETE FROM orders")
    assert not r.ok and "read-only" in r.error
    assert conn.execute("SELECT COUNT(*) FROM orders").fetchone()[0] == 4


def test_run_query_caps_rows(conn):
    r = sv.run_query(conn, "SELECT * FROM orders", max_rows=2)
    assert r.ok and len(r.rows) == 2 and r.truncated


def test_run_query_times_out_on_runaway_queries(conn):
    runaway = ("WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM c) "
               "SELECT COUNT(*) FROM c")
    r = sv.run_query(conn, runaway, timeout_s=0.3)
    assert not r.ok and "timeout" in r.error.lower()


# ---------- formatting ----------

def test_format_result_pipe_table_and_nulls():
    text = sv.format_result(["region", "total"], [("N", 15.75), ("E", None)])
    assert text.splitlines()[0] == "region | total"
    assert "N | 15.75" in text and "E | NULL" in text


def test_format_result_truncation_and_empty():
    rows = [(i,) for i in range(30)]
    text = sv.format_result(["n"], rows, max_rows=5, total_rows=30)
    assert text.count("\n") <= 7 and "(25 more rows)" in text
    assert "no rows" in sv.format_result(["n"], []).lower()


def test_results_equal_ignores_row_order_and_float_noise():
    a = sv.QueryResult(True, ["x"], [(1.0000001,), (2,)], "", False)
    b = sv.QueryResult(True, ["y"], [(2,), (1.0,)], "", False)
    assert sv.results_equal(a, b)
    c = sv.QueryResult(True, ["x"], [(3,), (1,)], "", False)
    assert not sv.results_equal(a, c)


def test_build_database_from_script_runs_ddl_and_inserts():
    c = sv.build_database("CREATE TABLE t(a INT); INSERT INTO t VALUES (1),(2);")
    assert c.execute("SELECT COUNT(*) FROM t").fetchone()[0] == 2


def test_build_database_rejects_dangerous_scripts():
    with pytest.raises(ValueError):
        sv.build_database("ATTACH DATABASE 'x.db' AS x;")


# ---------- rounding-tolerant comparison (differences in ROUND precision only)

def _qr(rows):
    return sv.QueryResult(True, ["x"], rows, "", False)


def test_tolerant_equal_accepts_coarser_rounding():
    assert sv.results_equal_tolerant(_qr([("a", 12613.58)]), _qr([("a", 12613.6)]))
    assert sv.results_equal_tolerant(_qr([("a", 12613.6)]), _qr([("a", 12613.58)]))


def test_tolerant_equal_still_rejects_real_differences():
    assert not sv.results_equal_tolerant(_qr([("a", 12613.58)]), _qr([("a", 12614.6)]))
    assert not sv.results_equal_tolerant(_qr([("a", 1)]), _qr([("b", 1)]))
    assert not sv.results_equal_tolerant(_qr([("a", 1)]), _qr([("a", 1), ("b", 2)]))


def test_tolerant_equal_is_order_insensitive_and_requires_ok_results():
    assert sv.results_equal_tolerant(_qr([("a", 1.24), ("b", 2.5)]), _qr([("b", 2.5), ("a", 1.2)]))
    bad = sv.QueryResult(False, [], [], "boom", False)
    assert not sv.results_equal_tolerant(bad, _qr([("a", 1)]))
