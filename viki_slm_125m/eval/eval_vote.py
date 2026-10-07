"""Greedy decoding vs sampling several queries and voting on their execution results.

Usage: python -m viki_slm_125m.eval.eval_vote [checkpoint] [--samples 8] [--temperature 0.7] [--gretel-n 600]
Inference-only: no training. Scores with the strict and the rounding-tolerant judge.
"""

from __future__ import annotations

import argparse
import json
import sys

from tokenizers import Tokenizer

from viki_slm_125m.eval import eval_sft as es
from viki_slm_125m.eval import evaluator as ev
from viki_slm_125m.sft import sql_vote

SAMPLE_BATCH = 64


def candidates_for(model, tok, items: list[dict], device: str, n: int, temperature: float) -> list[list[str]]:
    """Per item: [greedy sql, sample 1, ..., sample n] (None where no SQL block was produced)."""
    prompts = [ev.build_prompt(es.SYSTEMS[i % len(es.SYSTEMS)], es._user(it["schema"], it["question"]))
               for i, it in enumerate(items)]
    stops = ["<|/sql|>", "<|eos|>"]
    greedy = es.generate(model, tok, prompts, stops, 220, device)
    sampled = es.generate(model, tok, [p for p in prompts for _ in range(n)], stops, 220, device, batch=SAMPLE_BATCH,
                          do_sample=True, temperature=temperature, top_p=0.95) if n else []
    out = []
    for i, g in enumerate(greedy):
        row = [ev.extract_block(g, "sql")] + [ev.extract_block(t, "sql") for t in sampled[i * n:(i + 1) * n]]
        out.append(row)
    return out


def score(items: list[dict], cands: list[list[str]], k: int) -> dict:
    """Accuracy when voting over the greedy query plus the first k samples (k=0 is plain greedy)."""
    recs = []
    for it, row in zip(items, cands):
        pool = row[:1 + k]
        choice = sql_vote.vote(it["conn"], pool) if k else None
        chosen = choice.sql if choice else row[0]
        status, _ = ev.judge_sql(chosen, it["gold_sql"], it["conn"])
        recs.append({"status": status, "format_ok": chosen is not None, "insight_ok": None, "faithful": None})
    s = ev.summarize(recs)
    return {"execution_accuracy": s["execution_accuracy"], "tolerant": s["execution_accuracy_tolerant"],
            "executes_rate": s["executes_rate"]}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint", nargs="?", default="artifacts/sft_v4/best.pt")
    p.add_argument("--samples", type=int, default=8)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--gretel-n", type=int, default=600)
    p.add_argument("--report", default="reports/eval_vote_report.json")
    args = p.parse_args(argv)
    device = "cuda"
    tok = Tokenizer.from_file(es.TOKENIZER)
    model = es.load_model(args.checkpoint, device)
    report: dict = {"checkpoint": args.checkpoint, "samples": args.samples, "temperature": args.temperature}
    for name, items in (("gretel_unseen_domains", es.gretel_items(args.gretel_n)),
                        ("domain_unseen_templates", es.domain_items())):
        cands = candidates_for(model, tok, items, device, args.samples, args.temperature)
        report[name] = {f"k={k}": score(items, cands, k) for k in sorted({0, 4, args.samples})}
        print(name, json.dumps({k: {m: round(v, 3) for m, v in d.items()} for k, d in report[name].items()}), flush=True)
    with open(args.report, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
