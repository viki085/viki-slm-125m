"""Build verified SQL agent traces from gretelai/synthetic_text_to_sql (runs locally, free).

Usage: python -m viki_slm_125m.sft.builders.build_sql_sft [out_dir]   (default: data/sft/)
Writes: gretel_train.jsonl, gretel_val.jsonl, gretel_test_unseen.jsonl, gretel_report.json
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import random
import sys
import urllib.request
from collections import Counter

import pandas as pd

from viki_slm_125m.sft import sft_sources as ss

log = logging.getLogger("build_sql_sft")
DATASET = "gretelai/synthetic_text_to_sql"
TEST_DOMAINS = ("oceanography", "ethical fashion", "rural development", "arts and culture")
VAL_FRACTION = 0.02
CAP_GENERAL = 600       # examples per ordinary domain
CAP_TARGET = 1_500      # examples per finance / supply-chain-like domain


def _get_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "viki-slm-125m"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.load(resp)


def train_urls() -> list[str]:
    info = _get_json("https://datasets-server.huggingface.co/parquet?dataset=" + DATASET)
    return [f["url"] for f in info["parquet_files"] if f["split"] == "train"]


def load_rows() -> list[dict]:
    frames = []
    for url in train_urls():
        req = urllib.request.Request(url, headers={"User-Agent": "viki-slm-125m"})
        with urllib.request.urlopen(req, timeout=300) as resp:
            frames.append(pd.read_parquet(io.BytesIO(resp.read())))
        log.info("loaded %s rows so far", sum(len(f) for f in frames))
    return pd.concat(frames).to_dict("records")


def _is_val(example_id: str) -> bool:
    h = int(hashlib.blake2b(example_id.encode(), digest_size=4).hexdigest(), 16)
    return (h % 10_000) < VAL_FRACTION * 10_000


def main(out_dir: str = "data/sft") -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    os.makedirs(out_dir, exist_ok=True)
    rng = random.Random(0)
    rows = load_rows()
    drops: Counter = Counter()
    kept: list[dict] = []
    for i, rec in enumerate(rows):
        ex = ss.gretel_record_to_example(rec, rng)
        if ex is None:
            drops["skipped"] += 1
        else:
            kept.append(ex)
        if i and i % 20_000 == 0:
            log.info("processed %d rows, kept %d", i, len(kept))
    test = [e for e in kept if e["domain"] in TEST_DOMAINS]
    pool = [e for e in kept if e["domain"] not in TEST_DOMAINS]
    selected = ss.select_examples(pool, CAP_GENERAL, CAP_TARGET, rng)
    train = [e for e in selected if not _is_val(e["id"])]
    val = [e for e in selected if _is_val(e["id"])]
    for name, items in (("gretel_train", train), ("gretel_val", val), ("gretel_test_unseen", test)):
        with open(os.path.join(out_dir, f"{name}.jsonl"), "w", encoding="utf-8") as fh:
            for e in items:
                fh.write(json.dumps(e, ensure_ascii=False) + "\n")
    report = {"rows_read": len(rows), "verified_examples": len(kept), "train": len(train),
              "val": len(val), "test_unseen": len(test),
              "target_domain_examples": sum(1 for e in train if ss.domain_weight(e["domain"]) > 1),
              "domains": len({e["domain"] for e in train})}
    with open(os.path.join(out_dir, "gretel_report.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    log.info(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(*sys.argv[1:2]))
