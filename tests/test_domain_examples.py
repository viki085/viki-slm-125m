"""Tests for the domain databases, templates and example builders."""

import random

import pytest

from viki_slm_125m.sft.domain import domain_data as dd
from viki_slm_125m.sft.domain import domain_examples as de
from viki_slm_125m.sft.domain.domain_templates import PY_TEMPLATES, SQL_TEMPLATES, instantiate
from viki_slm_125m.sft.sql_verify import run_query

SEEDS = range(8)


@pytest.mark.parametrize("domain", ["supply_chain", "finance"])
def test_databases_build_with_data_in_every_table(domain):
    conn = de.build_db(domain, 0)
    names = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")]
    assert len(names) >= 6
    for n in names:
        assert conn.execute(f"SELECT COUNT(*) FROM {n}").fetchone()[0] > 0, n


def test_database_is_deterministic_per_seed():
    a = dd.table_csv(de.build_db("finance", 3), "accounts")
    b = dd.table_csv(de.build_db("finance", 3), "accounts")
    c = dd.table_csv(de.build_db("finance", 4), "accounts")
    assert a == b and a != c


@pytest.mark.parametrize("t", SQL_TEMPLATES, ids=lambda t: t.name)
def test_every_sql_template_runs_and_is_usually_non_empty(t):
    nonempty = 0
    for seed in SEEDS:
        conn = de.build_db(t.domain, seed)
        _, sql = instantiate(t, conn, random.Random(seed))
        res = run_query(conn, sql)
        assert res.ok, f"{t.name} seed {seed}: {res.error}"
        nonempty += bool(res.rows)
    assert nonempty >= 6, f"{t.name} empty on too many seeds ({nonempty}/8)"


def test_templates_have_holdouts_in_both_domains():
    for d in ("supply_chain", "finance"):
        assert de.sql_templates(d, holdout=True) and de.sql_templates(d)


def test_parse_pipe_output_types_values():
    cols, rows = de.parse_pipe_output("name|total\nAcme|12.5\nBeta|3\n")
    assert cols == ["name", "total"] and rows == [("Acme", 12.5), ("Beta", 3)]
    assert de.parse_pipe_output("only header") == ([], [])


def test_build_sql_example_is_complete_and_reproducible():
    t = de.sql_templates("supply_chain")[0]
    ex = de.build_sql_example(t, 5, random.Random(0))
    assert ex["gold_sql"].startswith("SELECT") and ex["template"] == t.name and ex["seed"] == 5
    assert [x["role"] for x in ex["turns"]] == ["system", "user", "assistant", "tool", "assistant"]
    # the gold SQL re-runs on the regenerated database (needed for the unseen-schema eval)
    conn = de.build_db(t.domain, ex["seed"])
    assert run_query(conn, ex["gold_sql"]).ok


@pytest.mark.parametrize("name", ["py_spend_supplier", "py_balance_type", "py_qty_outliers"])
def test_python_examples_execute_and_have_faithful_insights(name):
    t = next(t for t in PY_TEMPLATES if t.name == name)
    ex = de.build_python_example(t, 2, random.Random(0))
    assert ex is not None
    roles = [x["role"] for x in ex["turns"]]
    assert roles == ["system", "user", "assistant", "tool", "assistant"]
    assert "<|python|>" in ex["turns"][2]["text"] and "<|output|>" in ex["turns"][3]["text"]
    assert ".csv:" in ex["turns"][1]["text"]


@pytest.mark.parametrize("builder,needle", [
    (de.build_refusal, "read-only"), (de.build_missing_info, "no data about"),
    (de.build_clarification, "clarify")])
def test_refusal_style_examples_have_no_tool_call(builder, needle):
    for domain in ("supply_chain", "finance"):
        ex = builder(domain, 1, random.Random(0))
        assert [x["role"] for x in ex["turns"]] == ["system", "user", "assistant"]
        assert needle in ex["turns"][2]["text"] and "<|sql|>" not in ex["turns"][2]["text"]


def test_paraphrase_is_deterministic_and_keeps_the_question():
    q = "What is the total balance by account type?"
    outs = {de.paraphrase(q, random.Random(i)) for i in range(40)}
    assert len(outs) >= 4                                # real variety
    assert all(q.rstrip("?").lower()[:20] in o.lower() for o in outs)
    assert de.paraphrase(q, random.Random(3)) == de.paraphrase(q, random.Random(3))
    assert q in outs                                      # the plain form stays possible


def test_sql_example_uses_the_paraphrased_question_everywhere():
    t = de.sql_templates("finance")[0]
    ex = de.build_sql_example(t, 11, random.Random(5))
    assert ex["question"] in ex["turns"][1]["text"]
