"""Build refusal / clarification / missing-data examples on many different gretel schemas (local, free).

Usage: python -m viki_slm_125m.sft.builders.build_refusal_sft [out_dir]
Writes refusal_train.jsonl, refusal_val.jsonl, refusal_test_unseen.jsonl (held-out domains), report.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import sys
from collections import Counter

from viki_slm_125m.sft.builders import build_sql_sft as bs
from viki_slm_125m.sft import sft_refusal as sr
from viki_slm_125m.sft import sft_sources as ss

log = logging.getLogger("build_refusal_sft")
N_REFUSE, N_CLARIFY, N_MISSING = 6_000, 3_000, 2_500
N_TEST_EACH = 100
VAL_FRACTION = 0.02


def _is_val(i: str) -> bool:
    return int(hashlib.blake2b(i.encode(), digest_size=4).hexdigest(), 16) % 10_000 < VAL_FRACTION * 10_000


def build(rows: list[dict], n_refuse: int, n_clarify: int, n_missing: int, rng: random.Random) -> list[dict]:
    rng.shuffle(rows)
    out: list[dict] = []
    counts: Counter = Counter()
    seen_schemas: set[str] = set()
    for rec in rows:
        rid, dom = str(rec.get("id", "")), rec.get("domain", "")
        if counts["refusal"] < n_refuse:
            ex = sr.refusal_from_gretel(rec, rng)
            if ex:
                out.append(ex)
                counts["refusal"] += 1
                continue
        if rec.get("sql_task_type") in sr.WRITE_TASK_TYPES:
            continue
        schema = ss.schema_from_context(rec.get("sql_context") or "")
        if not schema or schema in seen_schemas:
            continue
        seen_schemas.add(schema)
        if counts["clarify"] < n_clarify:
            ex = sr.clarify_from_schema(schema, rid, dom, rng)
            if ex:
                out.append(ex)
                counts["clarify"] += 1
                continue
        if counts["missing"] < n_missing:
            ex = sr.missing_from_schema(schema, rid, dom, rng)
            if ex:
                out.append(ex)
                counts["missing"] += 1
    log.info("built %s", dict(counts))
    return out


def main(out_dir: str = "data/sft") -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    rows = bs.load_rows()
    rng = random.Random(3)
    train_rows = [r for r in rows if r.get("domain") not in bs.TEST_DOMAINS]
    test_rows = [r for r in rows if r.get("domain") in bs.TEST_DOMAINS]
    train = build(train_rows, N_REFUSE, N_CLARIFY, N_MISSING, rng)
    test = build(test_rows, N_TEST_EACH, N_TEST_EACH, N_TEST_EACH, random.Random(4))
    for e in test:
        e["kind"] = e["source"].split("-")[-1]
    splits = {"refusal_train": [e for e in train if not _is_val(e["id"])],
              "refusal_val": [e for e in train if _is_val(e["id"])], "refusal_test_unseen": test}
    for name, items in splits.items():
        with open(os.path.join(out_dir, f"{name}.jsonl"), "w", encoding="utf-8") as fh:
            for e in items:
                fh.write(json.dumps(e, ensure_ascii=False) + "\n")
    report = {k: len(v) for k, v in splits.items()}
    report["train_by_source"] = dict(Counter(e["source"] for e in splits["refusal_train"]))
    report["test_by_kind"] = dict(Counter(e["kind"] for e in test))
    with open(os.path.join(out_dir, "refusal_report.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    log.info(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(*sys.argv[1:2]))
