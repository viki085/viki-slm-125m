"""Build verified SQL agent traces from the Spider train split (local, free).

Needs the Spider data at data/external/spider_raw/spider_data (HF mirror HAL-9001/spider-databases).
Excludes the 20 dev databases. Validation uses whole held-out databases.

Usage: python -m viki_slm_125m.sft.builders.build_spider_sft [out_dir]   -> spider_train.jsonl, spider_val.jsonl, report
"""

from __future__ import annotations

import json
import logging
import os
import random
import sys
from collections import Counter
from pathlib import Path

from viki_slm_125m.sft import spider_source as sp

log = logging.getLogger("build_spider_sft")
ROOT = Path("data/external/spider_raw/spider_data")
MAX_SAME_QUERY = 3        # Spider repeats a query with several paraphrased questions


def load_records() -> tuple[list[dict], set[str]]:
    train = json.loads((ROOT / "train_spider.json").read_text(encoding="utf-8"))
    others = json.loads((ROOT / "train_others.json").read_text(encoding="utf-8"))
    dev = json.loads((ROOT / "dev.json").read_text(encoding="utf-8"))
    return train + others, {d["db_id"] for d in dev}


def main(out_dir: str = "data/sft") -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    records, dev_dbs = load_records()
    records = sp.exclude_dbs(records, dev_dbs)
    rng = random.Random(3)
    train: list[dict] = []
    val: list[dict] = []
    seen: Counter = Counter()
    drops: Counter = Counter()
    conns: dict = {}
    for i, rec in enumerate(records):
        db = rec["db_id"]
        if db not in conns:
            conns[db] = sp.open_readonly(str(ROOT / "database" / db / f"{db}.sqlite"))
        conn = conns[db]
        key = (db, sp.normalize_sql(rec["query"]))
        if seen[key] >= MAX_SAME_QUERY:
            drops["repeat"] += 1
            continue
        ex = sp.record_to_example(rec, conn, sp.schema_from_sqlite(conn), rng, i)
        if ex is None:
            drops["no rows or error"] += 1
            continue
        seen[key] += 1
        (val if sp.is_val_db(db) else train).append(ex)
    os.makedirs(out_dir, exist_ok=True)
    for name, items in (("spider_train", train), ("spider_val", val)):
        with open(os.path.join(out_dir, f"{name}.jsonl"), "w", encoding="utf-8") as fh:
            for e in items:
                fh.write(json.dumps(e, ensure_ascii=False) + "\n")
    report = {"records": len(records), "excluded_dev_databases": sorted(dev_dbs), "train": len(train),
              "val": len(val), "train_databases": len({e["domain"] for e in train}),
              "val_databases": len({e["domain"] for e in val}), "dropped": dict(drops)}
    with open(os.path.join(out_dir, "spider_report.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    log.info(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(*sys.argv[1:2]))
