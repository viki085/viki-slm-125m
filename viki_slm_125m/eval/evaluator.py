"""Pure helpers for evaluating the SFT model on execution accuracy and the agent loop."""

from __future__ import annotations

import re
from typing import Mapping, Sequence

from viki_slm_125m.sft.sql_verify import results_equal, results_equal_tolerant, run_query


def build_prompt(system: str, user: str) -> str:
    return f"<|bos|><|system|>{system}<|user|>{user}<|assistant|>"


def extract_block(text: str, tag: str) -> str | None:
    """Text between <|tag|> and <|/tag|>, or None if the block is missing or not closed."""
    m = re.search(rf"<\|{tag}\|>(.*?)<\|/{tag}\|>", text, re.DOTALL)
    return m.group(1).strip() if m else None


def judge_sql(pred_sql: str | None, gold_sql: str, conn) -> tuple[str, str]:
    """(status, detail): no_sql | error | mismatch | match_rounding | match, by comparing executed results.

    match_rounding: equal except for ROUND precision; it counts only in the tolerant accuracy."""
    if not pred_sql:
        return "no_sql", ""
    pred = run_query(conn, pred_sql)
    if not pred.ok:
        return "error", pred.error
    gold = run_query(conn, gold_sql)
    if results_equal(pred, gold):
        return "match", ""
    return ("match_rounding", "") if results_equal_tolerant(pred, gold) else ("mismatch", "")


def _rate(values: Sequence, pred) -> float:
    return sum(1 for v in values if pred(v)) / len(values) if values else 0.0


def summarize(records: Sequence[Mapping]) -> dict:
    n = len(records)
    if n == 0:
        return {"n": 0}
    reached = [r for r in records if r.get("insight_ok") is not None]
    judged = [r for r in records if r.get("faithful") is not None]
    return {
        "n": n,
        "execution_accuracy": sum(r["status"] == "match" for r in records) / n,
        "execution_accuracy_tolerant": sum(r["status"] in ("match", "match_rounding") for r in records) / n,
        "executes_rate": sum(r["status"] in ("match", "match_rounding", "mismatch") for r in records) / n,
        "format_rate": sum(bool(r["format_ok"]) for r in records) / n,
        "insight_format_rate": _rate(reached, lambda r: r["insight_ok"]),
        "faithful_rate": _rate(judged, lambda r: r["faithful"]),
    }
