"""Spider dev execution accuracy (external benchmark), greedy and voted.

Needs the Spider data (cc-by-sa-4.0, evaluation only) unpacked at data/external/spider_raw/spider_data
(mirror: HF dataset HAL-9001/spider-databases). Questions whose schema does not fit the context window
count as failures. Usage: python -m viki_slm_125m.eval.eval_spider [checkpoint] [--limit N] [--samples 8]
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

from tokenizers import Tokenizer

from viki_slm_125m.eval import eval_sft as es
from viki_slm_125m.eval import eval_vote as evv
from viki_slm_125m.eval import evaluator as ev
from viki_slm_125m.sft.spider_source import open_readonly, schema_from_sqlite

ROOT = Path("data/external/spider_raw/spider_data")
MAX_PROMPT_TOKENS = 1_700        # leaves room for the generated query inside the 2,048 context


def load_items(tok: Tokenizer, limit: int | None) -> tuple[list[dict], int]:
    dev = json.loads((ROOT / "dev.json").read_text(encoding="utf-8"))
    dev = dev[:limit] if limit else dev
    conns: dict[str, sqlite3.Connection] = {}
    items, too_long = [], 0
    for d in dev:
        db = d["db_id"]
        if db not in conns:
            conns[db] = open_readonly(str(ROOT / "database" / db / f"{db}.sqlite"))
        schema = schema_from_sqlite(conns[db])
        prompt = ev.build_prompt(es.SYSTEMS[0], es._user(schema, d["question"]))
        if len(tok.encode(prompt).ids) > MAX_PROMPT_TOKENS:
            too_long += 1
            continue
        items.append({"id": f"{db}-{len(items)}", "schema": schema, "question": d["question"],
                      "gold_sql": d["query"], "conn": conns[db], "domain": db})
    return items, too_long


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint", nargs="?", default="artifacts/sft_v4/best.pt")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--samples", type=int, default=8)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--report", default="reports/eval_spider_report.json")
    args = p.parse_args(argv)
    tok = Tokenizer.from_file(es.TOKENIZER)
    items, too_long = load_items(tok, args.limit)
    total = len(items) + too_long
    model = es.load_model(args.checkpoint, "cuda")
    cands = evv.candidates_for(model, tok, items, "cuda", args.samples, args.temperature)
    report = {"checkpoint": args.checkpoint, "questions": total, "evaluated": len(items), "too_long": too_long,
              "scores_on_evaluated": {}, "scores_all_questions": {}}
    for k in sorted({0, 4, args.samples}):
        s = evv.score(items, cands, k)
        report["scores_on_evaluated"][f"k={k}"] = s
        report["scores_all_questions"][f"k={k}"] = {m: v * len(items) / total for m, v in s.items()}
    print(json.dumps(report, indent=1))
    with open(args.report, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
