"""Tests for sft_sources.py: turning public text-to-SQL records into verified agent traces."""

import random

import pytest

from viki_slm_125m.sft import sft_sources as ss

CONTEXT = ("CREATE TABLE sales (id INT, region TEXT, amount FLOAT); "
           "INSERT INTO sales (id, region, amount) VALUES (1, 'N; north', 10.5), (2, 'S', 20.0), (3, 'N', 5.25);")


def _rec(**kw):
    base = {"id": "7", "domain": "finance", "sql_task_type": "analytics and reporting",
            "sql_prompt": "What is the total amount per region?", "sql_context": CONTEXT,
            "sql": "SELECT region, SUM(amount) FROM sales GROUP BY region;"}
    return {**base, **kw}


def test_split_statements_respects_quotes():
    parts = ss.split_sql_statements(CONTEXT)
    assert len(parts) == 2 and parts[0].startswith("CREATE TABLE sales")
    assert "'N; north'" in parts[1]


def test_schema_from_context_keeps_only_create_statements():
    schema = ss.schema_from_context(CONTEXT)
    assert schema.startswith("CREATE TABLE sales") and "INSERT" not in schema


def test_gretel_record_becomes_agent_trace():
    out = ss.gretel_record_to_example(_rec(), random.Random(0))
    assert out is not None
    assert out["source"] == "gretel" and out["domain"] == "finance"
    roles = [t["role"] for t in out["turns"]]
    assert roles == ["system", "user", "assistant", "tool", "assistant"]
    assert "N | 15.75" in out["turns"][3]["text"] or "S | 20" in out["turns"][3]["text"]


@pytest.mark.parametrize("override", [
    {"sql_task_type": "data manipulation"},
    {"sql": "SELECT nope FROM sales"},
    {"sql": "SELECT * FROM sales WHERE id = 99"},
    {"sql": "DROP TABLE sales"},
    {"sql_context": "CREATE TABLE broken (("},
    {"sql_context": "ATTACH DATABASE 'x' AS y;"},
    {"sql_prompt": ""},
])
def test_gretel_record_skips_unusable_rows(override):
    assert ss.gretel_record_to_example(_rec(**override), random.Random(0)) is None


def test_domain_weight_prefers_target_domains():
    assert ss.domain_weight("finance") > ss.domain_weight("sports")
    assert ss.domain_weight("supply chain logistics") > ss.domain_weight("beauty industry")
    assert ss.domain_weight("financial services") == ss.domain_weight("banking")


def test_select_examples_caps_per_domain_and_keeps_targets():
    exs = ([{"domain": "sports", "id": i} for i in range(50)]
           + [{"domain": "finance", "id": i} for i in range(50)])
    picked = ss.select_examples(exs, cap_general=10, cap_target=40, rng=random.Random(0))
    assert sum(1 for e in picked if e["domain"] == "sports") == 10
    assert sum(1 for e in picked if e["domain"] == "finance") == 40
