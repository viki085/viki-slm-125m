"""Turn domain templates into verified SFT examples (SQL loops, pandas loops, refusals)."""

from __future__ import annotations

import random
import sqlite3
from typing import Sequence

from viki_slm_125m.sft import sandbox
from viki_slm_125m.sft import sft_data
from viki_slm_125m.sft.domain.domain_data import DOMAINS, table_csv
from viki_slm_125m.sft.domain.domain_templates import PY_TEMPLATES, SQL_TEMPLATES, PyTemplate, SqlTemplate, instantiate
from viki_slm_125m.sft.domain.domain_templates_v3 import V3_PY_TEMPLATES, V3_SQL_TEMPLATES
from viki_slm_125m.sft.domain.domain_templates_v4 import V4_SQL_TEMPLATES
from viki_slm_125m.sft.sft_sources import schema_from_context
from viki_slm_125m.sft.sql_verify import run_query

PY_SYSTEM_PROMPTS = (
    "You are a data analyst assistant. Write Python with pandas to answer the question using the "
    "provided CSV files, print the result as a small table, then explain it.",
    "You help users analyze data in notebooks. Use only the files and columns listed, run the code, "
    "and summarize the printed output accurately.",
)


def _tables(domain: str) -> list[str]:
    ddl = DOMAINS[domain][0]
    return [s.split("(")[0].replace("CREATE TABLE", "").strip() for s in ddl.split(";") if s.strip()]


_PREFIXES = ("", "", "Can you tell me ", "I need to know ", "Please show me ", "Quick question: ", "Could you work out ")


def paraphrase(question: str, rng: random.Random) -> str:
    """Light, meaning-preserving rewording (prefix + lowercased first letter)."""
    prefix = rng.choice(_PREFIXES)
    if not prefix:
        return question
    body = question[0].lower() + question[1:]
    if prefix.endswith("show me ") and body.endswith("?"):
        body = body[:-1] + "."
    return prefix + body


def build_db(domain: str, seed: int) -> sqlite3.Connection:
    return DOMAINS[domain][1](random.Random(seed))


def parse_pipe_output(stdout: str) -> tuple[list[str], list[tuple]]:
    """Parse `df.to_csv(sep='|')` text back into columns and typed rows."""
    lines = [ln for ln in stdout.strip().splitlines() if ln.strip()]
    if len(lines) < 2:
        return [], []
    cols = [c.strip() for c in lines[0].split("|")]

    def typed(v: str):
        v = v.strip()
        for cast in (int, float):
            try:
                return cast(v)
            except ValueError:
                continue
        return v

    rows = [tuple(typed(v) for v in ln.split("|")) for ln in lines[1:]]
    return cols, [r for r in rows if len(r) == len(cols)]


def build_sql_example(t: SqlTemplate, seed: int, rng: random.Random) -> dict | None:
    ddl = DOMAINS[t.domain][0]
    conn = build_db(t.domain, seed)
    try:
        question, sql = instantiate(t, conn, random.Random(seed + 1))
        question = paraphrase(question, random.Random(seed + 2))
        turns = sft_data.build_sql_trace(schema_from_context(ddl), question, sql, conn, rng)
    finally:
        conn.close()
    if turns is None:
        return None
    return {"source": f"domain-sql-{t.domain}", "id": f"{t.name}-{seed}", "domain": t.domain,
            "template": t.name, "seed": seed, "question": question, "gold_sql": sql,
            "turns": [{"role": x.role, "text": x.text} for x in turns]}


def build_python_example(t: PyTemplate, seed: int, rng: random.Random) -> dict | None:
    conn = build_db(t.domain, seed)
    try:
        question, code = instantiate(t, conn, random.Random(seed + 1))
        question = paraphrase(question, random.Random(seed + 2))
        # every table of the domain is offered (as in the playground) so the model learns to pick the right ones
        names = _tables(t.domain)
        files = {f"{name}.csv": table_csv(conn, name) for name in names}
        header = "\n".join(f"{name}.csv: " + ", ".join(r[1] for r in conn.execute(f"PRAGMA table_info({name})"))
                           for name in names)
    finally:
        conn.close()
    res = sandbox.run_python(code, files=files, timeout_s=20)
    if not res.ok or not res.stdout.strip():
        return None
    cols, rows = parse_pipe_output(res.stdout)
    if not rows:
        return None
    insight = sft_data.make_insight(cols, rows)
    if not sft_data.numbers_supported(insight, cols, rows):
        return None
    out = res.stdout.strip()
    turns = [
        sft_data.Turn("system", rng.choice(PY_SYSTEM_PROMPTS)),
        sft_data.Turn("user", f"<|schema|>\n{header}\n<|/schema|>\n{question}"),
        sft_data.Turn("assistant", f"<|python|>\n{code}\n<|/python|>"),
        sft_data.Turn("tool", f"<|output|>\n{out}\n<|/output|>"),
        sft_data.Turn("assistant", f"<|insight|>\n{insight}\n<|/insight|>"),
    ]
    return {"source": f"domain-python-{t.domain}", "id": f"{t.name}-{seed}", "domain": t.domain,
            "template": t.name, "seed": seed, "question": question,
            "turns": [{"role": x.role, "text": x.text} for x in turns]}


# ---------------------------------------------------------- refusals and clarifications
_DESTRUCTIVE = ("Delete all records from the {table} table.", "Drop the {table} table.",
                "Update every row in {table} and set the numbers to zero.",
                "Remove all rows from {table} that are older than last year.")
_MISSING = {
    "supply_chain": ("customer satisfaction score", "carbon emissions per shipment", "employee headcount per warehouse",
                     "profit margin per product", "supplier payment terms"),
    "finance": ("customer credit score", "employee salaries", "branch locations",
                "customer age", "monthly interest earned"),
}
_AMBIGUOUS = {
    "supply_chain": (("Show me the best suppliers.", "best", "highest rating, shortest lead time, or best on-time delivery"),
                     ("Which warehouse is the biggest?", "biggest", "largest capacity or most stock on hand"),
                     ("Which product sells the most?", "sells the most", "highest ordered quantity or highest total spend")),
    "finance": (("Who are our best customers?", "best", "largest total balance, most transactions, or lowest risk rating"),
                ("Show me risky loans.", "risky", "loans with status late or default, a high interest rate, or many late repayments"),
                ("Which accounts are active?", "active", "accounts with recent transactions or accounts with a non-zero balance")),
}


def _wrap(domain: str, user_question: str, reply: str, rng: random.Random, kind: str, idx: int) -> dict:
    ddl = DOMAINS[domain][0]
    turns = [sft_data.Turn("system", rng.choice(sft_data.SYSTEM_PROMPTS)),
             sft_data.Turn("user", f"<|schema|>\n{schema_from_context(ddl)}\n<|/schema|>\n{user_question}"),
             sft_data.Turn("assistant", f"<|insight|>\n{reply}\n<|/insight|>")]
    return {"source": f"domain-{kind}", "id": f"{kind}-{domain}-{idx}", "domain": domain,
            "turns": [{"role": t.role, "text": t.text} for t in turns]}


def build_refusal(domain: str, idx: int, rng: random.Random) -> dict:
    table = rng.choice(_tables(domain))
    q = rng.choice(_DESTRUCTIVE).format(table=table)
    reply = (f"I can only run read-only SELECT queries, so I can't change or delete data in {table}. "
             f"I can help you analyze it instead, for example by counting or summarizing the rows in {table}.")
    return _wrap(domain, q, reply, rng, "refusal", idx)


def build_missing_info(domain: str, idx: int, rng: random.Random) -> dict:
    concept = rng.choice(_MISSING[domain])
    q = rng.choice((f"What is the {concept}?", f"Show the {concept} for each record.", f"Report the {concept}."))
    reply = (f"I can't answer that from this schema because it has no data about {concept}. "
             f"The available tables are {', '.join(_tables(domain))}. "
             "If you can point me to the right table or share its schema, I can write the query.")
    return _wrap(domain, q, reply, rng, "missing", idx)


def build_clarification(domain: str, idx: int, rng: random.Random) -> dict:
    q, term, options = rng.choice(_AMBIGUOUS[domain])
    reply = (f"Could you clarify what you mean by \"{term}\"? I can use {options}. "
             "Tell me which one you want and I will run the query.")
    return _wrap(domain, q, reply, rng, "clarify", idx)


def sql_templates(domain: str | None = None, holdout: bool = False) -> Sequence[SqlTemplate]:
    return [t for t in SQL_TEMPLATES + V3_SQL_TEMPLATES + V4_SQL_TEMPLATES if t.holdout == holdout and (domain is None or t.domain == domain)]


def python_templates(domain: str | None = None) -> Sequence[PyTemplate]:
    return [t for t in PY_TEMPLATES + V3_PY_TEMPLATES if domain is None or t.domain == domain]


__all__ = ["paraphrase", "build_db", "parse_pipe_output", "build_sql_example", "build_python_example", "build_refusal",
           "build_missing_info", "build_clarification", "sql_templates", "python_templates", "run_query"]
