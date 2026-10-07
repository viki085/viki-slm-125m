"""Build SQL self-repair traces from the gretel training rows (local, free).

Usage: python -m viki_slm_125m.sft.builders.build_repair_sft [out_dir]  -> repair_train.jsonl, repair_val.jsonl, repair_report.json
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import sqlite3
import sys
from collections import Counter

from viki_slm_125m.sft.builders import build_sql_sft as bs
from viki_slm_125m.sft import sft_repair
from viki_slm_125m.sft.domain import domain_examples as de
from viki_slm_125m.sft.domain.domain_data import DOMAINS
from viki_slm_125m.sft import sft_sources as ss
from viki_slm_125m.sft.sql_verify import build_database, is_read_only

log = logging.getLogger("build_repair_sft")
TARGET = 8_000
VAL_FRACTION = 0.02


DOMAIN_TARGET = 3_000


def _kind_of(fix_text: str) -> str:
    if "using only columns from the schema" in fix_text:
        return "rewrite"
    for kind, what in sft_repair._WHAT.items():
        if f"correct the {what}" in fix_text:
            return kind
    return "other"


def domain_repairs(rng: random.Random, kinds: Counter) -> list[dict]:
    """Repair traces on the finance / supply-chain databases, from the verified template SQL."""
    out: list[dict] = []
    with open("data/sft/domain_train.jsonl", encoding="utf-8") as fh:
        rows = [r for r in map(json.loads, fh) if "gold_sql" in r]
    rng.shuffle(rows)
    for r in rows:
        if len(out) >= DOMAIN_TARGET:
            break
        conn = de.build_db(r["domain"], r["seed"])
        try:
            turns = sft_repair.make_repair_trace(ss.schema_from_context(DOMAINS[r["domain"]][0]),
                                                 r["question"], r["gold_sql"], conn, rng)
        finally:
            conn.close()
        if turns is None:
            continue
        kinds[_kind_of(turns[4].text)] += 1
        out.append({"source": "repair-domain", "id": f"repair-{r['id']}", "domain": r["domain"],
                    "turns": [{"role": t.role, "text": t.text, "trained": t.trained} for t in turns]})
    return out


def main(out_dir: str = "data/sft") -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    with open(os.path.join(out_dir, "gretel_train.jsonl"), encoding="utf-8") as fh:
        train_ids = {json.loads(line)["id"] for line in fh}
    rows = [r for r in bs.load_rows() if str(r.get("id", "")) in train_ids]
    rng = random.Random(1)
    rng.shuffle(rows)
    out: list[dict] = []
    kinds: Counter = Counter()
    for rec in rows:
        if len(out) >= TARGET:
            break
        sql, context = rec.get("sql") or "", rec.get("sql_context") or ""
        if not is_read_only(sql):
            continue
        try:
            conn = build_database(context)
        except (ValueError, sqlite3.Error):
            continue
        try:
            turns = sft_repair.make_repair_trace(ss.schema_from_context(context), rec["sql_prompt"].strip(),
                                                 sql, conn, rng)
        finally:
            conn.close()
        if turns is None:
            continue
        kinds[_kind_of(turns[4].text)] += 1
        out.append({"source": "repair", "id": f"repair-{rec['id']}", "domain": rec.get("domain", ""),
                    "turns": [{"role": t.role, "text": t.text, "trained": t.trained} for t in turns]})
    out += domain_repairs(rng, kinds)

    def is_val(i: str) -> bool:
        return int(hashlib.blake2b(i.encode(), digest_size=4).hexdigest(), 16) % 10_000 < VAL_FRACTION * 10_000

    splits = {"repair_train": [e for e in out if not is_val(e["id"])],
              "repair_val": [e for e in out if is_val(e["id"])]}
    for name, items in splits.items():
        with open(os.path.join(out_dir, f"{name}.jsonl"), "w", encoding="utf-8") as fh:
            for e in items:
                fh.write(json.dumps(e, ensure_ascii=False) + "\n")
    report = {"built": len(out), "train": len(splits["repair_train"]), "val": len(splits["repair_val"]),
              "kinds": dict(kinds)}
    with open(os.path.join(out_dir, "repair_report.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    log.info(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(*sys.argv[1:2]))
