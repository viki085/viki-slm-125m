"""Spider train records -> verified SFT agent traces (multi-table schemas with real data).

Spider is cc-by-sa-4.0: attribute it and keep the share-alike terms in mind when releasing weights.
The 20 dev databases are always excluded so the Spider dev benchmark stays clean.
"""

from __future__ import annotations

import hashlib
import random
import re
import sqlite3
from typing import Iterable, Mapping, Optional

from viki_slm_125m.sft import sft_data

VAL_DB_FRACTION = 0.05


def open_readonly(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, check_same_thread=False)
    conn.text_factory = lambda b: b.decode("utf-8", errors="ignore")
    return conn


def schema_from_sqlite(conn: sqlite3.Connection) -> str:
    rows = conn.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' "
                        "AND sql IS NOT NULL ORDER BY name").fetchall()
    return "\n".join(r[0].strip().rstrip(";") + ";" for r in rows)


def normalize_sql(sql: str) -> str:
    """Collapse Spider's irregular spacing outside string literals."""
    parts = re.split(r"('(?:[^']|'')*')", sql.strip().rstrip(";").strip())
    out = []
    for i, part in enumerate(parts):
        if i % 2 == 0:
            part = re.sub(r"\s+", " ", part)
            part = re.sub(r"\s+,", ",", part)
        out.append(part)
    return "".join(out).strip()


def exclude_dbs(records: Iterable[Mapping], dbs: set[str]) -> list[Mapping]:
    return [r for r in records if r["db_id"] not in dbs]


def is_val_db(db_id: str) -> bool:
    """Whole databases are held out for validation, so validation measures unseen schemas."""
    h = int(hashlib.blake2b(db_id.encode(), digest_size=4).hexdigest(), 16)
    return (h % 10_000) < VAL_DB_FRACTION * 10_000


def record_to_example(rec: Mapping, conn: sqlite3.Connection, schema: str, rng: random.Random,
                      index: int) -> Optional[dict]:
    sql = normalize_sql(rec["query"])
    turns = sft_data.build_sql_trace(schema, rec["question"], sql, conn, rng)
    if turns is None:
        return None
    return {"source": "spider", "id": f"spider-{index}", "domain": rec["db_id"],
            "turns": [{"role": t.role, "text": t.text, "trained": t.trained} for t in turns]}
