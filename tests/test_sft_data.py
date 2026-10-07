"""Tests for sft_data.py: conversation rendering, loss masks, insights, SQL traces (written first)."""

import random
import sqlite3

import pytest

from viki_slm_125m import config
from viki_slm_125m.sft import sft_data as sd
from viki_slm_125m.data import tokenizer_lib as tl


@pytest.fixture(scope="module")
def tok():
    texts = ["SELECT region, SUM(amount) AS total FROM orders GROUP BY region ORDER BY total DESC LIMIT 5",
             "The query returned 3 rows. total is highest for north (15.75)",
             "You are a data analyst assistant. Use only the schema provided.",
             "CREATE TABLE orders (id INTEGER, region TEXT, amount REAL)"] * 30
    return tl.build_tokenizer(iter(texts), vocab_size=500, specials=config.all_special_tokens())


def enc_fn(tok):
    def enc(text):
        e = tok.encode(text)
        return e.ids, e.offsets
    return enc


TURNS = [
    sd.Turn("system", "You are a data analyst."),
    sd.Turn("user", "<|schema|>\nCREATE TABLE t(a INT)\n<|/schema|>\nHow many rows?"),
    sd.Turn("assistant", "<|sql|>\nSELECT COUNT(*) FROM t\n<|/sql|>"),
    sd.Turn("tool", "<|result|>\n3\n<|/result|>"),
    sd.Turn("assistant", "<|insight|>\nThe table has 3 rows.\n<|/insight|>"),
]


# ---------- rendering ----------

def test_render_layout_and_trained_spans():
    text, spans = sd.render(TURNS)
    assert text.startswith("<|bos|><|system|>You are a data analyst.<|user|>")
    assert text.count("<|assistant|>") == 2
    trained = [text[s:e] for s, e in spans]
    assert trained == ["<|sql|>\nSELECT COUNT(*) FROM t\n<|/sql|>",
                       "<|insight|>\nThe table has 3 rows.\n<|/insight|><|eos|>"]


def test_render_adds_eos_only_after_final_assistant_turn():
    text, _ = sd.render(TURNS)
    assert text.count("<|eos|>") == 1 and text.endswith("<|eos|>")


def test_render_rejects_bad_conversations():
    with pytest.raises(ValueError):
        sd.render([sd.Turn("user", "hi")])                      # no assistant turn
    with pytest.raises(ValueError):
        sd.render([sd.Turn("alien", "x"), sd.Turn("assistant", "y")])
    with pytest.raises(ValueError):
        sd.render([sd.Turn("assistant", "y"), sd.Turn("user", "ends on user")])


# ---------- encoding and masks ----------

def test_encode_example_masks_only_assistant_tokens(tok):
    ex = sd.encode_example(TURNS, enc_fn(tok), max_len=512)
    assert ex is not None and len(ex.ids) == len(ex.mask)
    trained = tok.decode([i for i, m in zip(ex.ids, ex.mask) if m], skip_special_tokens=False)
    assert "SELECT COUNT" in trained and "table has 3 rows" in trained
    assert "data analyst" not in trained and "How many rows" not in trained
    assert "3\n<|/result|>" not in trained.replace("The table has 3 rows.", "")
    # the eos id is trained exactly once, as the very last token
    eos = tok.token_to_id("<|eos|>")
    assert ex.ids[-1] == eos and ex.mask[-1] == 1
    assert sum(1 for i, m in zip(ex.ids, ex.mask) if i == eos and m) == 1


def test_role_tokens_and_tool_results_are_not_trained(tok):
    ex = sd.encode_example(TURNS, enc_fn(tok), max_len=512)
    for name in ("<|assistant|>", "<|user|>", "<|system|>", "<|result|>", "<|/result|>"):
        tid = tok.token_to_id(name)
        assert all(m == 0 for i, m in zip(ex.ids, ex.mask) if i == tid)


def test_encode_example_drops_overlong_examples(tok):
    assert sd.encode_example(TURNS, enc_fn(tok), max_len=10) is None


# ---------- insights ----------

def test_make_insight_shapes():
    assert "no rows" in sd.make_insight(["a"], []).lower()
    assert sd.make_insight(["n"], [(42,)]) == "The result is 42."
    one = sd.make_insight(["name", "age"], [("Ann", 31)])
    assert "name = Ann" in one and "age = 31" in one
    many = sd.make_insight(["region", "total"], [("N", 15.75), ("S", 20.0), ("E", 3.0)])
    assert "3 rows" in many and "S (20)" in many and "E (3)" in many


def test_make_insight_mentions_truncation():
    rows = [(f"r{i}", i) for i in range(10)]
    text = sd.make_insight(["k", "v"], rows, total_rows=50)
    assert "50" in text and "first 10" in text


def test_insights_are_faithful_by_construction():
    cases = [(["a"], []), (["n"], [(42,)]), (["r", "t"], [("N", 15.75), ("S", 20.0)]),
             (["q", "v"], [("Q3", 1234.5), ("Q4", 99.0), ("Q1", 7.0)])]
    for cols, rows in cases:
        assert sd.numbers_supported(sd.make_insight(cols, rows), cols, rows)


def test_numbers_supported_flags_invented_numbers():
    assert not sd.numbers_supported("Revenue was 999 in the north.", ["r", "t"], [("N", 15.75)])


# ---------- plan and trace ----------

def test_plan_from_sql_describes_structure():
    plan = sd.plan_from_sql("SELECT region, SUM(amount) FROM orders WHERE amount > 1 "
                            "GROUP BY region ORDER BY 2 DESC LIMIT 3")
    assert "orders" in plan and "SUM" in plan and "group" in plan.lower() and "limit" in plan.lower()
    assert sd.plan_from_sql("not sql at all ((") == ""


@pytest.fixture()
def conn():
    c = sqlite3.connect(":memory:")
    c.executescript("CREATE TABLE orders(id INT, region TEXT, amount REAL);"
                    "INSERT INTO orders VALUES (1,'N',10.5),(2,'S',20),(3,'N',5.25);")
    yield c
    c.close()


def test_build_sql_trace_makes_full_agent_loop(conn):
    ddl = "CREATE TABLE orders(id INT, region TEXT, amount REAL)"
    turns = sd.build_sql_trace(ddl, "Total revenue per region?",
                               "SELECT region, SUM(amount) AS total FROM orders GROUP BY region",
                               conn, rng=random.Random(0))
    roles = [t.role for t in turns]
    assert roles == ["system", "user", "assistant", "tool", "assistant"]
    assert "<|schema|>" in turns[1].text and "Total revenue" in turns[1].text
    assert "<|sql|>" in turns[2].text and "<|/sql|>" in turns[2].text
    assert "<|result|>" in turns[3].text and "N | 15.75" in turns[3].text
    assert "<|insight|>" in turns[4].text


def test_build_sql_trace_skips_failing_and_empty_queries(conn):
    ddl = "CREATE TABLE orders(id INT)"
    rng = random.Random(0)
    assert sd.build_sql_trace(ddl, "q", "SELECT nope FROM orders", conn, rng=rng) is None
    assert sd.build_sql_trace(ddl, "q", "SELECT * FROM orders WHERE id = 99", conn, rng=rng) is None
    assert sd.build_sql_trace(ddl, "q", "DELETE FROM orders", conn, rng=rng) is None


def test_untrained_assistant_turn_is_masked_but_present():
    turns = [sd.Turn("system", "s"), sd.Turn("user", "q"),
             sd.Turn("assistant", "<|sql|>\nSELECT oops\n<|/sql|>", trained=False),
             sd.Turn("tool", "<|result|>\nError: no such column\n<|/result|>"),
             sd.Turn("assistant", "<|sql|>\nSELECT 1\n<|/sql|>", trained=True),
             sd.Turn("tool", "<|result|>\n1\n<|/result|>"),
             sd.Turn("assistant", "<|insight|>\nThe result is 1.\n<|/insight|>")]
    text, spans = sd.render(turns)
    assert "SELECT oops" in text
    trained = [text[s:e] for s, e in spans]
    assert all("oops" not in t for t in trained) and len(trained) == 2


def test_final_assistant_turn_must_be_trained():
    with pytest.raises(ValueError):
        sd.render([sd.Turn("user", "q"), sd.Turn("assistant", "a", trained=False)])
