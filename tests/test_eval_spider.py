"""Tests for eval_spider.py helpers."""

import sqlite3

from viki_slm_125m.eval import eval_spider as sp


def test_schema_from_sqlite_lists_create_statements_without_internal_tables():
    c = sqlite3.connect(":memory:")
    c.executescript("CREATE TABLE a(id INT PRIMARY KEY, name TEXT);CREATE TABLE b(a_id INT, v REAL);"
                    "INSERT INTO a VALUES (1,'x');")
    schema = sp.schema_from_sqlite(c)
    assert "CREATE TABLE a" in schema and "CREATE TABLE b" in schema
    assert "sqlite_" not in schema and "INSERT" not in schema


def test_open_readonly_refuses_writes(tmp_path):
    path = tmp_path / "d.sqlite"
    c = sqlite3.connect(path)
    c.execute("CREATE TABLE t(x INT)")
    c.commit()
    c.close()
    ro = sp.open_readonly(str(path))
    try:
        ro.execute("INSERT INTO t VALUES (1)")
        raised = False
    except sqlite3.OperationalError:
        raised = True
    assert raised


def test_open_readonly_tolerates_non_utf8_text(tmp_path):
    path = tmp_path / "d.sqlite"
    c = sqlite3.connect(path)
    c.execute("CREATE TABLE t(x TEXT)")
    c.execute("INSERT INTO t VALUES (CAST(X'FF41' AS TEXT))")
    c.commit()
    c.close()
    assert sp.open_readonly(str(path)).execute("SELECT x FROM t").fetchone()[0].endswith("A")
