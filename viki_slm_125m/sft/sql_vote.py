"""Pick one SQL query from several candidates by executing them and voting on the results."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional, Sequence

from viki_slm_125m.sft.sql_verify import _norm_value, run_query


@dataclass(frozen=True)
class Choice:
    sql: str
    votes: int       # candidates that returned the same rows as the chosen one
    executed: int    # candidates that ran without error


def _signature(rows: Sequence[tuple]) -> tuple:
    """Rows as an order-insensitive key (column names are ignored, floats rounded)."""
    return tuple(sorted(repr(tuple(_norm_value(v) for v in row)) for row in rows))


def vote(conn, candidates: Sequence[Optional[str]], max_rows: int = 200) -> Optional[Choice]:
    """The query whose result most candidates agree on; ties go to the earliest candidate
    (put the greedy decode first). None if no candidate executes."""
    groups: "OrderedDict[tuple, list[str]]" = OrderedDict()
    for sql in candidates:
        if not sql:
            continue
        res = run_query(conn, sql, max_rows=max_rows)
        if res.ok:
            groups.setdefault(_signature(res.rows), []).append(sql)
    if not groups:
        return None
    best = max(groups.values(), key=len)       # max keeps the first of equal-sized groups
    return Choice(best[0], len(best), sum(len(g) for g in groups.values()))
