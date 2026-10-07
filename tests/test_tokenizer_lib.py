"""Tests for tokenizer_lib.py (written first). Uses a tiny BPE so tests run in seconds."""

import pytest

from viki_slm_125m import config
from viki_slm_125m.data import tokenizer_lib as tl

SAMPLE_TEXTS = [
    "def add(a, b):\n    return a + b\n",
    "SELECT name, SUM(amount) AS total FROM orders WHERE year = 2024 GROUP BY name;",
    "The Company reported revenue of 12345 dollars and 67890 in net income.",
    "    for i in range(10):\n        print(i * 3)\n",
    "Inventory turnover improved while supplier lead time fell by 15 percent.",
] * 40


@pytest.fixture(scope="module")
def tok():
    return tl.build_tokenizer(iter(SAMPLE_TEXTS), vocab_size=400, specials=config.all_special_tokens())


def test_every_special_token_is_a_single_id_and_ids_are_first(tok):
    specials = config.all_special_tokens()
    ids = tl.special_token_ids(tok, specials)
    assert list(ids.values()) == list(range(len(specials)))
    for text, i in ids.items():
        assert tok.encode(text).ids == [i]


def test_digits_are_split_individually(tok):
    assert len(tok.encode("12345").ids) == 5


def test_round_trip_is_exact_for_code_sql_and_unicode(tok):
    samples = ["def f(x):\n    return x\n\n\nclass A:\n        pass\n",
               "SELECT * FROM t WHERE name = 'café' AND n > 10;",
               "Prix: 5 € — résumé naïve 数据 🙂",
               "\tTabbed\tline  with   spaces\n"]
    assert tl.roundtrip_failures(tok, samples) == []


def test_indentation_runs_are_cheap(tok):
    assert len(tok.encode("\n" + " " * 12 + "x = 1").ids) <= 9


def test_chars_per_token_is_positive_and_consistent(tok):
    cpt = tl.chars_per_token(tok, SAMPLE_TEXTS[:5])
    assert cpt > 1.0
    assert tl.chars_per_token(tok, []) == 0.0


def test_quotas_split_budget_by_share():
    q = tl.source_quotas(1_000, {"a": 0.5, "b": 0.3, "c": 0.2})
    assert q == {"a": 500, "b": 300, "c": 200}
    with pytest.raises(ValueError):
        tl.source_quotas(1_000, {"a": 0.5, "b": 0.2})


def test_default_shares_cover_every_source_and_cap_web():
    shares = tl.TOKENIZER_SHARES
    assert set(shares) == {s.name for s in config.DATA_MIX}
    assert sum(shares.values()) == pytest.approx(1.0)
    web = sum(shares[n] for n in ("fineweb-edu", "cosmopedia", "math"))
    assert web <= 0.30 + 1e-9


def test_sample_docs_is_deterministic_and_disjoint_from_eval():
    docs = [f"doc number {i}" for i in range(100)]
    train = list(tl.sample_docs(docs, role="train", per_file_chars=10_000))
    ev = list(tl.sample_docs(docs, role="eval", per_file_chars=10_000))
    assert train == list(tl.sample_docs(docs, role="train", per_file_chars=10_000))
    assert train and ev and not set(train) & set(ev)


def test_sample_docs_respects_char_budget_and_truncates():
    docs = ["x" * 50_000] * 100
    out = list(tl.sample_docs(docs, role="train", per_file_chars=45_000))
    assert sum(len(d) for d in out) <= 45_000 + tl.MAX_DOC_CHARS
    assert all(len(d) <= tl.MAX_DOC_CHARS for d in out)
    with pytest.raises(ValueError):
        list(tl.sample_docs(docs, role="bogus", per_file_chars=10))
