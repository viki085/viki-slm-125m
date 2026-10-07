"""Converters from public text-to-SQL datasets to verified SFT agent traces."""

from __future__ import annotations

import random
import sqlite3
from collections import defaultdict
from typing import Iterable, Mapping

from viki_slm_125m.sft import sft_data
from viki_slm_125m.sft.sql_verify import build_database, is_read_only

ALLOWED_TASK_TYPES = frozenset({"analytics and reporting", "data retrieval"})
TARGET_DOMAIN_WORDS = ("financ", "bank", "invest", "insur", "payment", "credit", "loan",
                       "supply", "logistic", "inventory", "warehouse", "procure", "manufactur",
                       "retail", "transport", "shipping", "freight", "trade")


def split_sql_statements(script: str) -> list[str]:
    """Split a script on semicolons that are outside string literals."""
    out: list[str] = []
    cur: list[str] = []
    quote = ""
    for ch in script:
        if quote:
            cur.append(ch)
            if ch == quote:
                quote = ""
            continue
        if ch in ("'", '"', "`"):
            quote = ch
            cur.append(ch)
        elif ch == ";":
            stmt = "".join(cur).strip()
            if stmt:
                out.append(stmt)
            cur = []
        else:
            cur.append(ch)
    tail = "".join(cur).strip()
    if tail:
        out.append(tail)
    return out


def schema_from_context(context: str) -> str:
    """CREATE statements only (no data), one per line, each terminated with a semicolon."""
    creates = [s for s in split_sql_statements(context) if s.lstrip().lower().startswith("create")]
    return "\n".join(s + ";" for s in creates)


def domain_weight(domain: str) -> int:
    """2 for finance / supply-chain-like domains, 1 otherwise."""
    low = domain.lower()
    return 2 if any(w in low for w in TARGET_DOMAIN_WORDS) else 1


def gretel_record_to_example(rec: Mapping, rng: random.Random) -> dict | None:
    """One gretelai/synthetic_text_to_sql row -> trace, or None if it fails any check."""
    if rec.get("sql_task_type") not in ALLOWED_TASK_TYPES:
        return None
    prompt, sql, context = (rec.get("sql_prompt") or "").strip(), rec.get("sql") or "", rec.get("sql_context") or ""
    if not prompt or not is_read_only(sql):
        return None
    try:
        conn = build_database(context)
    except (ValueError, sqlite3.Error):
        return None
    try:
        schema = schema_from_context(context)
        if not schema:
            return None
        turns = sft_data.build_sql_trace(schema, prompt, sql, conn, rng)
    finally:
        conn.close()
    if turns is None:
        return None
    return {"source": "gretel", "id": str(rec.get("id", "")), "domain": rec.get("domain", ""),
            "turns": [{"role": t.role, "text": t.text} for t in turns]}


def select_examples(examples: Iterable[dict], cap_general: int, cap_target: int,
                    rng: random.Random) -> list[dict]:
    """Cap examples per domain; target (finance/supply-chain-like) domains get a larger cap."""
    by_domain: dict[str, list[dict]] = defaultdict(list)
    for e in examples:
        by_domain[e["domain"]].append(e)
    picked: list[dict] = []
    for domain, items in sorted(by_domain.items()):
        cap = cap_target if domain_weight(domain) > 1 else cap_general
        rng.shuffle(items)
        picked.extend(items[:cap])
    return picked
