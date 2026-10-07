"""Refusals, clarifications and missing-data answers built from arbitrary schemas (no templates per domain)."""

from __future__ import annotations

import random
import re
from typing import Mapping

from viki_slm_125m.sft import sft_data
from viki_slm_125m.sft.sft_sources import schema_from_context
from viki_slm_125m.sft.sql_verify import is_read_only

WRITE_TASK_TYPES = frozenset({"data manipulation", "data definition"})
_VERBS = {"insert": "add rows to", "update": "change data in", "delete": "delete data from",
          "create": "create or modify", "alter": "change the structure of", "drop": "drop"}
_TARGET = re.compile(r"(?:insert\s+into|update|delete\s+from|drop\s+table(?:\s+if\s+exists)?|alter\s+table|"
                     r"create\s+table(?:\s+if\s+not\s+exists)?)\s+([A-Za-z_]\w*)", re.IGNORECASE)
_NUMERIC_TYPES = ("int", "integer", "bigint", "smallint", "float", "double", "real", "decimal", "numeric", "number")
_SKIP_NAMES = ("id", "year", "month", "day", "zip", "phone")
_CONCEPTS = ("customer satisfaction score", "employee salaries", "profit margin", "weather conditions",
             "carbon emissions", "customer age", "website traffic", "shipping delays", "interest rate",
             "market share", "number of complaints", "inventory turnover", "employee headcount", "tax rate")
_CLARIFY_Q = (("best", "Show me the best {t}."), ("top", "Show the top {t}."),
              ("performing well", "Which {t} are performing well?"),
              ("most important", "What are the most important {t}?"), ("most popular", "Which {t} are the most popular?"))
_REFUSAL = ("I can only run read-only SELECT queries, so I can't {phrase} {table}. I can help you analyze the "
            "existing data instead, for example by counting or summarizing the rows in {table}.",
            "This assistant is read-only: it can't {phrase} {table}. If you want, I can write a SELECT query "
            "that inspects or summarizes the data in {table}.",
            "I'm not able to {phrase} {table} because I only run read-only SELECT queries. Tell me what you want "
            "to learn from the data and I will query it.")


def write_verb(sql: str) -> str | None:
    first = sql.strip().split(None, 1)[0].lower() if sql.strip() else ""
    return first if first in _VERBS else None


def target_table(sql: str) -> str | None:
    m = _TARGET.search(sql)
    return m.group(1) if m else None


def _wrap(source: str, rid: str, domain: str, schema: str, question: str, reply: str, rng: random.Random) -> dict:
    turns = [sft_data.Turn("system", rng.choice(sft_data.SYSTEM_PROMPTS)),
             sft_data.Turn("user", f"<|schema|>\n{schema.strip()}\n<|/schema|>\n{question.strip()}"),
             sft_data.Turn("assistant", f"<|insight|>\n{reply}\n<|/insight|>")]
    return {"source": source, "id": rid, "domain": domain,
            "turns": [{"role": t.role, "text": t.text} for t in turns]}


def refusal_from_gretel(rec: Mapping, rng: random.Random) -> dict | None:
    """A write/DDL request from gretel becomes a refusal on that row's own schema."""
    sql, prompt = rec.get("sql") or "", (rec.get("sql_prompt") or "").strip()
    verb = write_verb(sql)
    if rec.get("sql_task_type") not in WRITE_TASK_TYPES or verb is None or is_read_only(sql) or not prompt:
        return None
    schema = schema_from_context(rec.get("sql_context") or "")
    if not schema:
        return None
    table = target_table(sql) or "the database"
    reply = rng.choice(_REFUSAL).format(phrase=_VERBS[verb], table=table)
    return _wrap("gretel-refusal", f"refusal-{rec.get('id')}", rec.get("domain", ""), schema, prompt, reply, rng)


def _parse_tables(schema: str) -> dict[str, list[tuple[str, str]]]:
    tables: dict[str, list[tuple[str, str]]] = {}
    for m in re.finditer(r"create\s+table\s+(?:if\s+not\s+exists\s+)?([A-Za-z_]\w*)\s*\(", schema, re.IGNORECASE):
        depth, i, parts, cur = 1, m.end(), [], []
        while i < len(schema) and depth:
            ch = schema[i]
            depth += (ch == "(") - (ch == ")")
            if depth == 0:
                break
            if ch == "," and depth == 1:
                parts.append("".join(cur))
                cur = []
            else:
                cur.append(ch)
            i += 1
        parts.append("".join(cur))
        cols = []
        for p in parts:
            tokens = p.strip().split(None, 1)
            if len(tokens) == 2 and tokens[0].upper() not in ("PRIMARY", "FOREIGN", "UNIQUE", "CONSTRAINT"):
                cols.append((tokens[0].strip('`"'), tokens[1].lower()))
        tables[m.group(1)] = cols
    return tables


def numeric_columns(schema: str) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for table, cols in _parse_tables(schema).items():
        keep = [c for c, t in cols
                if t.split("(")[0].split()[0] in _NUMERIC_TYPES
                and not (c.lower() in _SKIP_NAMES or c.lower().endswith("_id"))]
        if keep:
            out[table] = keep
    return out


def _words(name: str) -> str:
    return name.replace("_", " ")


def clarify_from_schema(schema: str, rid: str, domain: str, rng: random.Random) -> dict | None:
    candidates = {t: c for t, c in numeric_columns(schema).items() if len(c) >= 2}
    if not candidates:
        return None
    table = rng.choice(sorted(candidates))
    cols = [_words(c) for c in candidates[table][:3]]
    term, template = rng.choice(_CLARIFY_Q)
    listing = cols[0] + " or " + cols[1] if len(cols) == 2 else ", ".join(cols[:-1]) + ", or " + cols[-1]
    reply = (f'Could you clarify what you mean by "{term}"? I can rank {_words(table)} by {listing}. '
             "Tell me which one you want and I will run the query.")
    return _wrap("schema-clarify", f"clarify-{rid}", domain, schema, template.format(t=_words(table)), reply, rng)


def missing_from_schema(schema: str, rid: str, domain: str, rng: random.Random) -> dict | None:
    tables = list(_parse_tables(schema))
    low = schema.lower()
    options = [c for c in _CONCEPTS if not any(w in low for w in c.split() if len(w) > 3)]
    if not tables or not options:
        return None
    concept = rng.choice(options)
    question = rng.choice((f"What is the {concept}?", f"Show the {concept} for each record.", f"Report the {concept}."))
    reply = (f"I can't answer that from this schema because it has no data about {concept}. "
             f"The available tables are {', '.join(tables)}. "
             "If you can point me to the right table or share its schema, I can write the query.")
    return _wrap("schema-missing", f"missing-{rid}", domain, schema, question, reply, rng)
