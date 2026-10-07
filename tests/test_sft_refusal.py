"""Tests for sft_refusal.py: refusals, clarifications and missing-data answers on arbitrary schemas."""

import random

import pytest

from viki_slm_125m.sft import sft_refusal as sr

SCHEMA = ("CREATE TABLE donations (id INT, donor TEXT, amount FLOAT, year INT, campaign_score DECIMAL(4,1));\n"
          "CREATE TABLE donors (donor_id INT, name VARCHAR(50), city TEXT);")


def test_write_verb_detection():
    assert sr.write_verb("INSERT INTO t VALUES (1)") == "insert"
    assert sr.write_verb("  update t set a=1") == "update"
    assert sr.write_verb("DELETE FROM t") == "delete"
    assert sr.write_verb("CREATE TABLE x (a INT)") == "create"
    assert sr.write_verb("DROP TABLE x") == "drop"
    assert sr.write_verb("SELECT 1") is None


def test_target_table_extraction():
    assert sr.target_table("INSERT INTO donations VALUES (1)") == "donations"
    assert sr.target_table("UPDATE donors SET name='x'") == "donors"
    assert sr.target_table("DELETE FROM donations WHERE a=1") == "donations"
    assert sr.target_table("DROP TABLE IF EXISTS old_t") == "old_t"
    assert sr.target_table("garbage") is None


def test_refusal_from_gretel_row_on_write_queries():
    rec = {"id": "1", "domain": "nonprofit", "sql_task_type": "data manipulation",
           "sql_prompt": "Delete all donations from 2019.", "sql": "DELETE FROM donations WHERE year = 2019;",
           "sql_context": SCHEMA + " INSERT INTO donations VALUES (1,'a',5.0,2019,1.0);"}
    ex = sr.refusal_from_gretel(rec, random.Random(0))
    assert [t["role"] for t in ex["turns"]] == ["system", "user", "assistant"]
    reply = ex["turns"][2]["text"]
    assert "<|insight|>" in reply and "<|sql|>" not in reply
    assert "read-only" in reply.lower() and "delete" in reply.lower()
    assert "Delete all donations from 2019." in ex["turns"][1]["text"]


@pytest.mark.parametrize("override", [
    {"sql_task_type": "analytics and reporting"},
    {"sql": "SELECT * FROM donations"},
    {"sql_context": "INSERT INTO t VALUES (1);"},          # no schema to show
    {"sql_prompt": ""},
])
def test_refusal_from_gretel_skips_non_write_rows(override):
    rec = {"id": "1", "domain": "d", "sql_task_type": "data manipulation", "sql_prompt": "Remove old rows.",
           "sql": "DELETE FROM donations WHERE year = 1;", "sql_context": SCHEMA}
    rec.update(override)
    assert sr.refusal_from_gretel(rec, random.Random(0)) is None


def test_numeric_columns_skips_ids_and_years():
    cols = sr.numeric_columns(SCHEMA)
    assert cols["donations"] == ["amount", "campaign_score"]
    assert "donors" not in cols or cols["donors"] == []


def test_clarification_lists_real_numeric_columns():
    ex = sr.clarify_from_schema(SCHEMA, "x1", "nonprofit", random.Random(0))
    reply = ex["turns"][2]["text"]
    assert "clarify" in reply.lower() and "amount" in reply and "campaign score" in reply
    assert "<|sql|>" not in reply


def test_clarification_needs_two_numeric_columns():
    assert sr.clarify_from_schema("CREATE TABLE t (id INT, name TEXT, amount FLOAT);", "x", "d", random.Random(0)) is None


def test_missing_info_picks_a_concept_absent_from_the_schema():
    ex = sr.missing_from_schema(SCHEMA, "x2", "nonprofit", random.Random(1))
    reply = ex["turns"][2]["text"]
    assert "no data about" in reply and "donations" in reply and "donors" in reply
    concept = reply.split("no data about ")[1].split(".")[0]
    assert concept.replace(" ", "_") not in SCHEMA.lower()
