"""Tests for dedup.py: exact/near duplicates and benchmark decontamination (written first)."""

import numpy as np
import pytest

from viki_slm_125m.data import dedup


# ---------- normalization and exact hashing ----------

def test_normalize_collapses_case_and_whitespace():
    assert dedup.normalize("  Hello \n\t World ") == "hello world"


def test_exact_hash_ignores_case_and_whitespace_only():
    assert dedup.exact_hash("Select  *\nfrom T") == dedup.exact_hash("select * from t")
    assert dedup.exact_hash("select a") != dedup.exact_hash("select b")
    assert isinstance(dedup.exact_hash("x"), int) and 0 <= dedup.exact_hash("x") < 2 ** 64


def test_canonical_for_hash_strips_comments_for_code_and_sql():
    py_a = "# header comment\nx = 1\n\ny = 2\n"
    py_b = "# another header\nx = 1\ny = 2\n"
    assert dedup.canonical_for_hash(py_a, "code") == dedup.canonical_for_hash(py_b, "code")
    sql_a = "-- note one\nSELECT 1;\n"
    sql_b = "-- note two\nSELECT 1;\n"
    assert dedup.canonical_for_hash(sql_a, "sql") == dedup.canonical_for_hash(sql_b, "sql")


def test_canonical_for_hash_keeps_prose_unchanged_apart_from_normalization():
    assert dedup.canonical_for_hash("# Title\nBody", "text") == dedup.normalize("# Title\nBody")


# ---------- global duplicate marking ----------

def test_duplicate_indices_keeps_first_occurrence_across_files():
    a = np.array([1, 2, 3, 2], dtype=np.uint64)   # file 0: idx 3 repeats idx 1
    b = np.array([3, 4, 1], dtype=np.uint64)      # file 1: idx 0 repeats a[2]; idx 2 repeats a[0]
    dups = dedup.duplicate_indices([a, b])
    assert dups[0].tolist() == [3]
    assert dups[1].tolist() == [0, 2]


def test_duplicate_indices_handles_empty_files():
    dups = dedup.duplicate_indices([np.array([], dtype=np.uint64), np.array([5], dtype=np.uint64)])
    assert [d.tolist() for d in dups] == [[], []]


# ---------- n-grams and contamination ----------

def test_words_tokenizes_lowercase_alnum():
    assert dedup.words("Def Add(a, b): return a+b") == ["def", "add", "a", "b", "return", "a", "b"]


def test_word_ngrams_counts_and_short_input():
    toks = list("abcdefghij")
    assert len(dedup.word_ngrams(toks, 3)) == 8
    assert dedup.word_ngrams(toks[:2], 3) == set()


BENCH_TEXT = ("write a function that returns the number of singers whose country is france "
              "and sort them by age in descending order using a single query")


def test_record_text_joins_strings_and_lists():
    rec = {"q": "How many?", "tests": ["assert a", "assert b"], "n": 3, "skip": "ignored"}
    assert dedup.record_text(rec, ("q", "tests", "n")) == "How many?\nassert a\nassert b"


def test_find_contamination_detects_planted_benchmark_text():
    grams = dedup.build_contamination_grams({"spider": [BENCH_TEXT]}, n=13)
    doc = "Some intro text that is fine. " + BENCH_TEXT + " More trailing text about nothing."
    assert dedup.find_contamination(doc, grams, n=13) == "spider"


def test_find_contamination_ignores_unrelated_and_short_overlap():
    grams = dedup.build_contamination_grams({"spider": [BENCH_TEXT]}, n=13)
    assert dedup.find_contamination("A completely different document about supply chain forecasting " * 5, grams, 13) is None
    # 12 shared words only: below the 13-gram window
    assert dedup.find_contamination(" ".join(BENCH_TEXT.split()[:12]), grams, 13) is None


def test_find_contamination_is_whitespace_and_case_insensitive():
    grams = dedup.build_contamination_grams({"mbpp": [BENCH_TEXT]}, n=13)
    assert dedup.find_contamination(BENCH_TEXT.upper().replace(" ", "\n  "), grams, 13) == "mbpp"


# ---------- near duplicates (MinHash) ----------

def _doc(seed: int, n: int = 400) -> str:
    rng = np.random.default_rng(seed)
    vocab = [f"w{i}" for i in range(5000)]
    return " ".join(vocab[i] for i in rng.integers(0, 5000, n))


def test_near_duplicate_ids_flags_later_copies_only():
    base = _doc(1)
    near = base.replace("w", "w", 1) + " extra tail words appended here"
    other = _doc(2)
    texts = {"a": base, "b": other, "c": near}
    assert dedup.near_duplicate_ids(texts) == ["c"]


def test_near_duplicate_ids_keeps_distinct_docs():
    assert dedup.near_duplicate_ids({"a": _doc(10), "b": _doc(11), "c": _doc(12)}) == []


def test_shingles_short_text():
    assert dedup.shingles(["a", "b"], 5) == {b"a b"}
    assert dedup.shingles([], 5) == set()


def test_near_duplicate_ids_accepts_streamed_pairs():
    base = _doc(1)
    pairs = iter([("a", base), ("b", _doc(2)), ("c", base + " tail")])
    assert dedup.near_duplicate_ids(pairs) == ["c"]


def test_numeric_and_repetitive_windows_are_not_contamination():
    bench = "1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 and 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0"
    grams = dedup.build_contamination_grams({"humaneval": [bench]}, n=13)
    assert dedup.find_contamination("table 3 4 5 6 7 8 9 10 11 12 13 14 15", grams, 13) is None
    assert dedup.find_contamination(" ".join(["0"] * 30), grams, 13) is None


def test_informative_ngram_requires_distinct_words():
    assert dedup.informative_ngram("select name from singer where age > 30 order by age desc limit 5 now".split()[:13] + [])
    assert not dedup.informative_ngram(["1"] * 13)
    assert not dedup.informative_ngram([str(i) for i in range(13)])


def test_short_boilerplate_overlap_is_not_contamination():
    boiler = "import numpy as np import pandas as pd import matplotlib pyplot as plt import seaborn as sns"
    bench = boiler + " df pd read csv data csv df head " + BENCH_TEXT
    grams = dedup.build_contamination_grams({"ds1000": [bench]}, n=13)
    # shares only the 18-word import boilerplate (6 windows), below the required run
    doc = boiler + " then we train a completely different model on unrelated customer churn data today"
    assert dedup.find_contamination(doc, grams, 13) is None


def test_long_consecutive_overlap_is_contamination():
    grams = dedup.build_contamination_grams({"humaneval": [BENCH_TEXT]}, n=13)
    assert dedup.find_contamination("intro " + BENCH_TEXT, grams, 13) == "humaneval"
