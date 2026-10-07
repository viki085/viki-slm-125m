"""SQL self-repair traces: a broken query, the real database error, then the fix.

The broken query is shown to the model but excluded from the loss, so it learns to repair
errors without learning to make them.
"""

from __future__ import annotations

import random
import re

from viki_slm_125m.sft import sft_data
from viki_slm_125m.sft.sql_verify import run_query


def _replace_word(sql: str, old: str, new: str) -> str:
    return re.sub(rf"\b{re.escape(old)}\b", new, sql, count=1)


def candidate_mutations(sql: str) -> dict[str, str]:
    """Plausible breakages of a good query, keyed by kind (not yet verified against a database)."""
    import sqlglot
    from sqlglot import exp

    out: dict[str, str] = {}
    try:
        tree = sqlglot.parse_one(sql, read="sqlite")
    except Exception:  # unparseable input: only the textual syntax mutation is possible
        tree = None
    if tree is not None:
        cols = sorted({c.name for c in tree.find_all(exp.Column) if c.name and len(c.name) > 2})
        tables = sorted({t.name for t in tree.find_all(exp.Table) if t.name})
        if cols:
            name = cols[0]
            out["column"] = _replace_word(sql, name, name[:-1] if len(name) > 3 else name + "x")
        if tables:
            out["table"] = _replace_word(sql, tables[0], tables[0] + "s")
    if re.search(r"\bFROM\b", sql):
        out["syntax"] = re.sub(r"\bFROM\b", "FORM", sql, count=1)
    return {k: v for k, v in out.items() if v != sql}


_WHAT = {"column": "column name", "table": "table name", "syntax": "syntax"}
_REWRITE_NOTE = "That column is not in the tables I used, so I will rewrite the query using only columns from the schema."


def _columns_by_table(conn) -> dict[str, list[str]]:
    tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")]
    return {t: [r[1] for r in conn.execute(f"PRAGMA table_info({t})")] for t in tables}


def rewrite_mutation(sql: str, conn, rng: random.Random) -> str | None:
    """A query whose select list starts with a column that does not belong to its tables.

    Teaches repairs that restructure the query rather than fix a one-character typo.
    """
    import sqlglot
    from sqlglot import exp

    try:
        tree = sqlglot.parse_one(sql, read="sqlite")
    except Exception:
        return None
    select = tree.find(exp.Select) if tree is not None else None
    if select is None or not select.expressions:
        return None
    used = {t.name for t in tree.find_all(exp.Table) if t.name}
    by_table = _columns_by_table(conn)
    own = {c.lower() for t in used for c in by_table.get(t, [])}
    foreign = sorted({c for t, cols in by_table.items() if t not in used for c in cols if c.lower() not in own})
    first = select.expressions[0]
    name = rng.choice(foreign) if foreign else (first.alias_or_name or "value") + "_total"
    select.set("expressions", [exp.column(name)] + list(select.expressions[1:]))
    broken = tree.sql(dialect="sqlite")
    return broken if broken != sql else None


def make_repair_trace(schema: str, question: str, good_sql: str, conn, rng: random.Random) -> list[sft_data.Turn] | None:
    """system, user, broken call (masked), error, fixed call, result, insight; None if not buildable."""
    good = sft_data.build_sql_trace(schema, question, good_sql, conn, rng, think_prob=0.0)
    if good is None:
        return None
    candidates = candidate_mutations(good_sql)
    rewrite = rewrite_mutation(good_sql, conn, rng)
    if rewrite is not None:
        candidates["rewrite"] = rewrite
    kinds = sorted(candidates)
    rng.shuffle(kinds)
    for kind in kinds:
        broken = candidates[kind]
        res = run_query(conn, broken)
        if res.ok:
            continue
        error = res.error.strip()
        clean_sql = good_sql.strip().rstrip(";")
        action = _REWRITE_NOTE if kind == "rewrite" else f"I will correct the {_WHAT[kind]} and run it again."
        fix = (f"<|think|>\nThe previous query failed: {error}. {action}\n<|/think|>\n"
               f"<|sql|>\n{clean_sql}\n<|/sql|>")
        return [good[0], good[1],
                sft_data.Turn("assistant", f"<|sql|>\n{broken.strip().rstrip(';')}\n<|/sql|>", trained=False),
                sft_data.Turn("tool", f"<|result|>\nError: {error}\n<|/result|>"),
                sft_data.Turn("assistant", fix),
                good[3], good[4]]
    return None
