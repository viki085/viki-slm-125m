"""SFT data: conversation rendering with loss masks, faithful templated insights, SQL traces.

Layout (guide section 4):  <|bos|><|system|>..<|user|>..<|assistant|>[<|think|>..<|/think|>]
<|sql|>..<|/sql|>  then the app's <|result|>..<|/result|>  then <|assistant|><|insight|>..<|/insight|><|eos|>
Only assistant text (and the final <|eos|>) is trained; role tokens and tool results are masked.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Callable, Sequence

from viki_slm_125m.sft.sql_verify import QueryResult, format_cell, format_result, run_query

ROLES = ("system", "user", "assistant", "tool")
SYSTEM_PROMPTS = (
    "You are a data analyst assistant. Use only the tables and columns in the schema. "
    "Write one read-only SQL query, then explain the result.",
    "You are a careful SQL and data analysis assistant. Answer using the provided schema, "
    "never invent columns, and never modify data.",
    "You help users analyze data. Given a schema and a question, write a single SELECT query "
    "for SQLite, wait for the result, then summarize the findings clearly.",
)


@dataclass(frozen=True)
class Turn:
    role: str
    text: str
    trained: bool = True     # only meaningful for assistant turns (e.g. a deliberately broken query)


@dataclass(frozen=True)
class Example:
    ids: list[int]
    mask: list[int]          # 1 where the token is trained


def render(turns: Sequence[Turn]) -> tuple[str, list[tuple[int, int]]]:
    """Full text plus the (start, end) character spans that are trained."""
    if not any(t.role == "assistant" for t in turns):
        raise ValueError("conversation needs at least one assistant turn")
    if turns[-1].role != "assistant" or not turns[-1].trained:
        raise ValueError("conversation must end with a trained assistant turn")
    parts: list[str] = ["<|bos|>"]
    spans: list[tuple[int, int]] = []
    pos = len(parts[0])
    for i, t in enumerate(turns):
        if t.role not in ROLES:
            raise ValueError(f"unknown role: {t.role}")
        if t.role == "assistant":
            parts.append("<|assistant|>")
            pos += len("<|assistant|>")
            body = t.text + ("<|eos|>" if i == len(turns) - 1 else "")
            if t.trained:
                spans.append((pos, pos + len(body)))
            parts.append(body)
            pos += len(body)
        else:
            piece = {"system": "<|system|>", "user": "<|user|>", "tool": ""}[t.role] + t.text
            parts.append(piece)
            pos += len(piece)
    return "".join(parts), spans


def encode_example(turns: Sequence[Turn], encode: Callable[[str], tuple[list[int], list[tuple[int, int]]]],
                   max_len: int) -> Example | None:
    """Token ids and loss mask; None if longer than max_len. `encode` returns (ids, offsets)."""
    text, spans = render(turns)
    ids, offsets = encode(text)
    if len(ids) > max_len:
        return None
    mask = [1 if any(s <= start < e for s, e in spans) else 0 for start, _ in offsets]
    return Example(list(ids), mask) if any(mask) else None


# ------------------------------------------------------------------- insights

_NUMBER = re.compile(r"\d+(?:\.\d+)?")


def _is_number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _numeric_column(rows: Sequence[tuple]) -> int | None:
    for c in range(len(rows[0]) - 1, -1, -1):
        vals = [r[c] for r in rows if r[c] is not None]
        if vals and all(_is_number(v) for v in vals):
            return c
    return None


def make_insight(columns: Sequence[str], rows: Sequence[tuple], total_rows: int | None = None) -> str:
    """Plain-language summary built only from numbers present in the result (faithful by design)."""
    if not rows:
        return "The query returned no rows, so there is nothing to report for this question."
    if len(rows) == 1 and len(columns) == 1:
        return f"The result is {format_cell(rows[0][0])}."
    if len(rows) == 1:
        pairs = ", ".join(f"{c} = {format_cell(v)}" for c, v in zip(columns, rows[0]))
        return f"The result is: {pairs}."
    total = total_rows if total_rows is not None else len(rows)
    head = (f"The query returned {total} rows; only the first {len(rows)} are shown."
            if total > len(rows) else f"The query returned {len(rows)} rows.")
    num = _numeric_column(rows)
    if num is None:
        return f"{head} The first row is {format_cell(rows[0][0])}."
    candidates = [(i, r) for i, r in enumerate(rows) if r[num] is not None]
    hi = max(candidates, key=lambda x: x[1][num])
    lo = min(candidates, key=lambda x: x[1][num])

    def label(item):
        i, r = item
        return format_cell(r[0]) if num != 0 else f"row {i + 1}"

    return (f"{head} {columns[num]} is highest for {label(hi)} ({format_cell(hi[1][num])}) "
            f"and lowest for {label(lo)} ({format_cell(lo[1][num])}).")


def numbers_supported(insight: str, columns: Sequence[str], rows: Sequence[tuple],
                      total_rows: int | None = None) -> bool:
    """Every number in the insight appears in the result (or is a row count)."""
    allowed = {str(len(rows))}
    if total_rows is not None:
        allowed.add(str(total_rows))
    for row in rows:
        for v in row:
            allowed.update(_NUMBER.findall(format_cell(v)))
    return all(n in allowed for n in _NUMBER.findall(insight))


# ----------------------------------------------------------------------- plan

def plan_from_sql(sql: str) -> str:
    """One-sentence structural plan derived from the SQL; empty string if it cannot be parsed."""
    import sqlglot
    from sqlglot import exp

    try:
        tree = sqlglot.parse_one(sql, read="sqlite")
    except Exception:  # sqlglot raises several error types on odd input
        return ""
    if tree is None:
        return ""
    tables = sorted({t.name for t in tree.find_all(exp.Table) if t.name})
    steps = [f"read from {', '.join(tables)}" if tables else "compute the value"]
    if tree.find(exp.Join):
        steps.append("join the tables")
    if tree.find(exp.Where):
        steps.append("filter rows with WHERE")
    aggs = sorted({a.key.upper() for a in tree.find_all(exp.AggFunc)})
    if aggs:
        steps.append("aggregate with " + ", ".join(aggs))
    if tree.find(exp.Group):
        steps.append("group the results")
    if tree.find(exp.Order):
        steps.append("sort the results")
    limit = tree.find(exp.Limit)
    if limit is not None:
        steps.append(f"limit to {limit.expression.name} rows")
    return "Plan: " + "; ".join(steps) + "."


# ---------------------------------------------------------------------- traces

def build_sql_trace(schema_ddl: str, question: str, sql: str, conn, rng: random.Random,
                    think_prob: float = 0.7, max_rows: int = 10) -> list[Turn] | None:
    """Full agent loop for one (schema, question, sql): None unless the query runs and returns rows."""
    result: QueryResult = run_query(conn, sql, max_rows=200)
    if not result.ok or not result.rows or result.truncated:
        return None
    shown = result.rows[:max_rows]
    table = format_result(result.columns, shown, max_rows=max_rows, total_rows=len(result.rows))
    insight = make_insight(result.columns, shown, total_rows=len(result.rows))
    sql_block = f"<|sql|>\n{sql.strip().rstrip(';')}\n<|/sql|>"
    plan = plan_from_sql(sql) if rng.random() < think_prob else ""
    call = f"<|think|>\n{plan}\n<|/think|>\n{sql_block}" if plan else sql_block
    return [
        Turn("system", rng.choice(SYSTEM_PROMPTS)),
        Turn("user", f"<|schema|>\n{schema_ddl.strip()}\n<|/schema|>\n{question.strip()}"),
        Turn("assistant", call),
        Turn("tool", f"<|result|>\n{table}\n<|/result|>"),
        Turn("assistant", f"<|insight|>\n{insight}\n<|/insight|>"),
    ]
