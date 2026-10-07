"""Safe SQL execution for SFT data: read-only gate, timeouts, result formatting and comparison."""

from __future__ import annotations

import re
import sqlite3
import time
from dataclasses import dataclass
from typing import Sequence

_FORBIDDEN_READ = frozenset({"insert", "update", "delete", "drop", "attach", "detach", "pragma",
                             "create", "alter", "vacuum", "reindex"})
_FORBIDDEN_SCRIPT = frozenset({"attach", "detach", "pragma", "vacuum"})
_WORD = re.compile(r"[A-Za-z_]+")
_BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)
_LINE_COMMENT = re.compile(r"--[^\n]*")


@dataclass(frozen=True)
class QueryResult:
    ok: bool
    columns: list[str]
    rows: list[tuple]
    error: str
    truncated: bool


def _strip_quoted(sql: str) -> str:
    """Remove string literals and quoted identifiers so keyword scans ignore them."""
    out: list[str] = []
    quote = ""
    for ch in sql:
        if quote:
            if ch == quote:
                quote = ""
            continue
        if ch in ("'", '"', "`"):
            quote = ch
            continue
        out.append(ch)
    return "".join(out)


def _clean(sql: str) -> str:
    return _LINE_COMMENT.sub(" ", _BLOCK_COMMENT.sub(" ", sql)).strip()


def is_read_only(sql: str) -> bool:
    """True only for a single SELECT/WITH statement with no write or admin keywords."""
    body = _strip_quoted(_clean(sql)).strip()
    if not body:
        return False
    body = body.rstrip(";").strip()
    if ";" in body:
        return False
    words = [w.lower() for w in _WORD.findall(body)]
    if not words or words[0] not in ("select", "with"):
        return False
    return not any(w in _FORBIDDEN_READ for w in words)


def run_query(conn: sqlite3.Connection, sql: str, max_rows: int = 200,
              timeout_s: float = 2.0) -> QueryResult:
    """Execute a read-only query with a wall-clock timeout; errors come back as text."""
    if not is_read_only(sql):
        return QueryResult(False, [], [], "rejected: only a single read-only SELECT is allowed", False)
    deadline = time.monotonic() + timeout_s
    conn.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 1000)
    conn.execute("PRAGMA query_only = ON")
    try:
        cur = conn.execute(sql)
        cols = [d[0] for d in cur.description or []]
        rows = cur.fetchmany(max_rows + 1)
        truncated = len(rows) > max_rows
        return QueryResult(True, cols, rows[:max_rows], "", truncated)
    except sqlite3.OperationalError as exc:
        msg = "timeout: query exceeded %.1fs" % timeout_s if "interrupted" in str(exc) else str(exc)
        return QueryResult(False, [], [], msg, False)
    except sqlite3.Error as exc:
        return QueryResult(False, [], [], str(exc), False)
    finally:
        conn.set_progress_handler(None, 0)
        conn.execute("PRAGMA query_only = OFF")


def _cell(value) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def format_result(columns: Sequence[str], rows: Sequence[tuple], max_rows: int = 20,
                  total_rows: int | None = None) -> str:
    """Compact pipe table used inside <|result|> blocks."""
    if not rows:
        return "(no rows)"
    lines = [" | ".join(columns)]
    lines += [" | ".join(_cell(v) for v in row) for row in rows[:max_rows]]
    total = total_rows if total_rows is not None else len(rows)
    if total > max_rows:
        lines.append(f"({total - max_rows} more rows)")
    return "\n".join(lines)


def _norm_value(v):
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return round(float(v), 4)
    return v


def results_equal(a: QueryResult, b: QueryResult) -> bool:
    """Order-insensitive result comparison (column names ignored, floats rounded)."""
    if not (a.ok and b.ok) or len(a.rows) != len(b.rows):
        return False
    if a.rows and len(a.rows[0]) != len(b.rows[0]):
        return False
    key = lambda row: repr(tuple(_norm_value(v) for v in row))  # noqa: E731
    return sorted(map(key, a.rows)) == sorted(map(key, b.rows))


def _decimals(v: float) -> int:
    text = repr(round(float(v), 6)).rstrip("0")
    return len(text.split(".")[1]) if "." in text else 0


def _column_precision(rows_a: list, rows_b: list, j: int) -> int:
    """Fewest decimals among the non-integer numbers of column j in either result (4 if there are none)."""
    vals = [r[j] for r in rows_a + rows_b if isinstance(r[j], (int, float)) and not isinstance(r[j], bool)]
    fractional = [_decimals(v) for v in vals if float(v) != int(float(v))]
    return min(fractional) if fractional else 4


def results_equal_tolerant(a: QueryResult, b: QueryResult) -> bool:
    """Like results_equal, but numbers in a column are compared at the coarsest precision used in it,
    so ROUND(x, 2) matches ROUND(x, 1). Real value differences still fail."""
    if not (a.ok and b.ok) or len(a.rows) != len(b.rows):
        return False
    if not a.rows:
        return True
    width = len(a.rows[0])
    if width != len(b.rows[0]):
        return False
    prec = [_column_precision(list(a.rows), list(b.rows), j) for j in range(width)]

    def norm(row):
        return repr(tuple(round(float(v), prec[j]) if isinstance(v, (int, float)) and not isinstance(v, bool) else v
                          for j, v in enumerate(row)))

    return sorted(map(norm, a.rows)) == sorted(map(norm, b.rows))


def build_database(script: str) -> sqlite3.Connection:
    """In-memory database from a DDL/INSERT script. Raises ValueError on admin statements,
    sqlite3.Error on invalid SQL."""
    words = {w.lower() for w in _WORD.findall(_strip_quoted(_clean(script)))}
    bad = words & _FORBIDDEN_SCRIPT
    if bad:
        raise ValueError(f"script contains forbidden statements: {sorted(bad)}")
    conn = sqlite3.connect(":memory:")
    try:
        conn.executescript(script)
    except sqlite3.Error:
        conn.close()
        raise
    return conn


format_cell = _cell  # public name for other modules
