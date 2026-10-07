"""Tests for spider_source.py: turn Spider train records into verified SFT examples."""

import random
import sqlite3

import pytest

from viki_slm_125m.sft import spider_source as ss


@pytest.fixture()
def conn():
    c = sqlite3.connect(":memory:")
    c.executescript("CREATE TABLE singer(id INT PRIMARY KEY, name TEXT, country TEXT, age INT);"
                    "INSERT INTO singer VALUES (1,'a','FR',30),(2,'b','US',25),(3,'c','FR',41);")
    yield c
    c.close()


def test_normalize_sql_collapses_spider_spacing():
    assert ss.normalize_sql("select name ,  country ,  age from singer  order by age desc") == \
        "select name, country, age from singer order by age desc"
    assert ss.normalize_sql("SELECT 'a  b' FROM t ;") == "SELECT 'a  b' FROM t"


def test_exclude_dbs_removes_every_record_of_those_databases():
    recs = [{"db_id": "x"}, {"db_id": "y"}, {"db_id": "x"}]
    assert ss.exclude_dbs(recs, {"x"}) == [{"db_id": "y"}]


def test_is_val_db_is_deterministic_and_roughly_five_percent():
    ids = [f"db_{i}" for i in range(2000)]
    flags = [ss.is_val_db(i) for i in ids]
    assert flags == [ss.is_val_db(i) for i in ids]
    assert 0.02 < sum(flags) / len(flags) < 0.09


def test_record_to_example_builds_a_verified_agent_trace(conn):
    schema = ss.schema_from_sqlite(conn)
    rec = {"db_id": "music", "question": "How many singers are from France?",
           "query": "select count(*)  from singer where country  =  'FR'"}
    ex = ss.record_to_example(rec, conn, schema, random.Random(0), index=7)
    assert ex["source"] == "spider" and ex["domain"] == "music" and ex["id"] == "spider-7"
    roles = [t["role"] for t in ex["turns"]]
    assert roles == ["system", "user", "assistant", "tool", "assistant"]
    assert "<|schema|>" in ex["turns"][1]["text"] and "CREATE TABLE singer" in ex["turns"][1]["text"]
    assert "count(*) from singer where country = 'FR'" in ex["turns"][2]["text"]
    assert "2" in ex["turns"][3]["text"]


def test_record_to_example_rejects_queries_that_fail_or_return_nothing(conn):
    schema = ss.schema_from_sqlite(conn)
    bad = {"db_id": "m", "question": "q", "query": "select nope from singer"}
    empty = {"db_id": "m", "question": "q", "query": "select name from singer where age > 100"}
    assert ss.record_to_example(bad, conn, schema, random.Random(0), 1) is None
    assert ss.record_to_example(empty, conn, schema, random.Random(0), 2) is None
