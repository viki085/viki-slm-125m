"""Build finance and supply-chain SFT data from templates (local, free, no LLM).

Usage: python -m viki_slm_125m.sft.builders.build_domain_sft [out_dir]
Writes domain_train.jsonl, domain_val.jsonl, domain_test_unseen.jsonl (held-out templates, gold SQL
and seeds for execution-accuracy evaluation) and domain_report.json.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from viki_slm_125m.sft.domain import domain_examples as de
from viki_slm_125m.sft.domain.domain_data import DOMAINS
from viki_slm_125m.sft.sft_sources import schema_from_context

log = logging.getLogger("build_domain_sft")
N_SQL, N_PY, N_REFUSE = 18_000, 5_000, 500   # N_REFUSE per kind
MAX_SAME_QUESTION = 12
TEST_SEEDS_PER_TEMPLATE = 60
VAL_FRACTION = 0.02
PY_WORKERS = 10


def _is_val(i: str) -> bool:
    return int(hashlib.blake2b(i.encode(), digest_size=4).hexdigest(), 16) % 10_000 < VAL_FRACTION * 10_000


def build_sql(rng: random.Random) -> list[dict]:
    templates = de.sql_templates()
    out: list[dict] = []
    seen: Counter = Counter()
    seed = 10_000
    while len(out) < N_SQL and seed < 10_000 + N_SQL * 3:
        t = templates[seed % len(templates)]
        seed += 1
        ex = de.build_sql_example(t, seed, rng)
        if ex is None:
            continue
        key = (ex["question"], ex["gold_sql"])
        if seen[key] >= MAX_SAME_QUESTION:
            continue
        seen[key] += 1
        out.append(ex)
    return out


PY_CACHE = "domain_python_cache.jsonl"


def build_python(rng: random.Random, out_dir: str = "data/sft") -> list[dict]:
    cache = os.path.join(out_dir, PY_CACHE)
    if os.path.exists(cache):
        with open(cache, encoding="utf-8") as fh:
            return [json.loads(line) for line in fh]
    templates = de.python_templates()
    jobs = [(templates[i % len(templates)], 50_000 + i) for i in range(int(N_PY * 1.15))]
    seeds = [random.Random(s) for _, s in jobs]

    def run(args):
        (t, seed), r = args
        return de.build_python_example(t, seed, r)

    with ThreadPoolExecutor(max_workers=PY_WORKERS) as pool:
        results = list(pool.map(run, zip(jobs, seeds)))
    kept = [r for r in results if r is not None][:N_PY]
    with open(cache, "w", encoding="utf-8") as fh:
        for r in kept:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    return kept


def build_refusals(rng: random.Random) -> list[dict]:
    out = []
    for builder in (de.build_refusal, de.build_missing_info, de.build_clarification):
        for i in range(N_REFUSE):
            out.append(builder(rng.choice(list(DOMAINS)), i, rng))
    return out


def build_test_set() -> list[dict]:
    items = []
    for t in de.sql_templates(holdout=True):
        for seed in range(900_000, 900_000 + TEST_SEEDS_PER_TEMPLATE):
            ex = de.build_sql_example(t, seed, random.Random(seed))
            if ex:
                items.append({"id": ex["id"], "domain": ex["domain"], "template": ex["template"], "seed": seed,
                              "schema": schema_from_context(DOMAINS[ex["domain"]][0]),
                              "question": ex["question"], "gold_sql": ex["gold_sql"]})
    return items


def main(out_dir: str = "data/sft") -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    os.makedirs(out_dir, exist_ok=True)
    rng = random.Random(7)
    sql, py, refuse = build_sql(rng), build_python(rng, out_dir), build_refusals(rng)
    log.info("sql=%d python=%d refusal-style=%d", len(sql), len(py), len(refuse))
    allx = sql + py + refuse
    splits = {"domain_train": [e for e in allx if not _is_val(e["id"])],
              "domain_val": [e for e in allx if _is_val(e["id"])],
              "domain_test_unseen": build_test_set()}
    for name, items in splits.items():
        with open(os.path.join(out_dir, f"{name}.jsonl"), "w", encoding="utf-8") as fh:
            for e in items:
                fh.write(json.dumps(e, ensure_ascii=False) + "\n")
    report = {"sql": len(sql), "python": len(py), "refusal_style": len(refuse),
              "train": len(splits["domain_train"]), "val": len(splits["domain_val"]),
              "test_unseen_templates": len(splits["domain_test_unseen"]),
              "python_templates_used": sorted({e["template"] for e in py}),
              "by_template": dict(Counter(e.get("template", e["source"]) for e in allx))}
    with open(os.path.join(out_dir, "domain_report.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    log.info(json.dumps({k: v for k, v in report.items() if k != "by_template"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(*sys.argv[1:2]))
