"""Evaluate the SFT model on unseen questions: SQL execution accuracy and the full agent loop.

Usage: python -m viki_slm_125m.eval.eval_sft [checkpoint] [--gretel-n 600]
Writes reports/eval_sft_report.json (metrics + sample transcripts).
"""

from __future__ import annotations

import argparse
import json
import random
import sys

import torch
from tokenizers import Tokenizer

from viki_slm_125m.sft.domain import domain_examples as de
from viki_slm_125m.eval import evaluator as ev
from viki_slm_125m.model.checkpoint import load_model
from viki_slm_125m.model.generation import generate
from viki_slm_125m.sft import sft_data
from viki_slm_125m.sft.sft_sources import schema_from_context
from viki_slm_125m.sft.sql_verify import build_database, format_result, run_query

TOKENIZER = "artifacts/tokenizer/tokenizer.json"
SYSTEMS = sft_data.SYSTEM_PROMPTS


def _user(schema: str, question: str) -> str:
    return f"<|schema|>\n{schema.strip()}\n<|/schema|>\n{question.strip()}"


def run_loop(model, tok, items: list[dict], device: str, label: str) -> tuple[dict, list[dict]]:
    """items: dicts with schema, question, gold_sql, conn. Returns (summary, records)."""
    prompts = [ev.build_prompt(SYSTEMS[i % len(SYSTEMS)], _user(it["schema"], it["question"]))
               for i, it in enumerate(items)]
    gen1 = generate(model, tok, prompts, ["<|/sql|>", "<|eos|>"], 220, device)
    recs: list[dict] = []
    follow: list[tuple[int, str, object]] = []
    for i, (it, g) in enumerate(zip(items, gen1)):
        sql = ev.extract_block(g, "sql")
        status, detail = ev.judge_sql(sql, it["gold_sql"], it["conn"])
        recs.append({"id": it.get("id"), "domain": it.get("domain"), "template": it.get("template"),
                     "question": it["question"], "pred_sql": sql, "gold_sql": it["gold_sql"],
                     "status": status, "detail": detail, "format_ok": sql is not None,
                     "insight_ok": None, "faithful": None, "insight": None})
        if status in ("match", "match_rounding", "mismatch"):
            res = run_query(it["conn"], sql, max_rows=200)
            if res.ok and not res.truncated:
                table = format_result(res.columns, res.rows[:10], 10, total_rows=len(res.rows))
                follow.append((i, prompts[i] + g + f"<|result|>\n{table}\n<|/result|><|assistant|>", res))
    if follow:
        gen2 = generate(model, tok, [f[1] for f in follow], ["<|eos|>"], 140, device)
        for (i, _, res), g in zip(follow, gen2):
            insight = ev.extract_block(g, "insight")
            recs[i]["insight_ok"] = insight is not None
            recs[i]["insight"] = insight
            if insight is not None:
                recs[i]["faithful"] = sft_data.numbers_supported(insight, res.columns, res.rows[:10], len(res.rows))
    summary = ev.summarize(recs)
    print(f"[{label}] " + json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in summary.items()}))
    return summary, recs


def domain_items() -> list[dict]:
    items = []
    with open("data/sft/domain_test_unseen.jsonl", encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            items.append({**r, "conn": de.build_db(r["domain"], r["seed"])})
    return items


def gretel_items(n: int) -> list[dict]:
    from viki_slm_125m.sft.builders import build_sql_sft as bs
    from viki_slm_125m.sft import sft_sources as ss

    wanted = {}
    with open("data/sft/gretel_test_unseen.jsonl", encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            wanted[r["id"]] = r
    ids = sorted(wanted)
    random.Random(0).shuffle(ids)
    ids = set(ids[:n])
    items = []
    for rec in bs.load_rows():
        rid = str(rec.get("id", ""))
        if rid not in ids:
            continue
        try:
            conn = build_database(rec["sql_context"])
        except Exception:
            continue
        items.append({"id": rid, "domain": rec.get("domain"), "schema": ss.schema_from_context(rec["sql_context"]),
                      "question": rec["sql_prompt"], "gold_sql": rec["sql"], "conn": conn})
    return items


CHECKS = {
    "refusal": (de.build_refusal, ("read-only", "can't", "cannot")),
    "missing_info": (de.build_missing_info, ("no data", "can't answer", "schema")),
    "clarification": (de.build_clarification, ("clarify", "which one", "mean by")),
}


def behaviour_checks(model, tok, device: str, n: int = 40) -> dict:
    out = {}
    rng = random.Random(123)
    for name, (builder, keywords) in CHECKS.items():
        exs = [builder(rng.choice(["supply_chain", "finance"]), 9000 + i, rng) for i in range(n)]
        prompts = [ev.build_prompt(e["turns"][0]["text"], e["turns"][1]["text"]) for e in exs]
        gens = generate(model, tok, prompts, ["<|eos|>"], 120, device)
        ok_shape = [("<|insight|>" in g and "<|sql|>" not in g) for g in gens]
        ok_kw = [any(k in g.lower() for k in keywords) for g in gens]
        out[name] = {"n": n, "no_sql_and_insight": sum(ok_shape) / n,
                     "expected_wording": sum(ok_kw) / n, "sample": gens[0]}
        print(f"[{name}] no-sql+insight {sum(ok_shape) / n:.2f}, wording {sum(ok_kw) / n:.2f}")
    return out


UNSEEN_RULES = {"refusal": ("read-only", "can't", "cannot", "not able"), "missing": ("no data about",),
                "clarify": ("clarify",)}


def unseen_behaviour(model, tok, device: str) -> dict:
    """Refusal / missing-data / clarification on schemas from domains never used in training."""
    with open("data/sft/refusal_test_unseen.jsonl", encoding="utf-8") as fh:
        items = [json.loads(line) for line in fh]
    prompts = [ev.build_prompt(i["turns"][0]["text"], i["turns"][1]["text"]) for i in items]
    gens = generate(model, tok, prompts, ["<|eos|>", "<|/sql|>"], 120, device)
    out: dict = {}
    for kind, words in UNSEEN_RULES.items():
        idx = [i for i, it in enumerate(items) if it["kind"] == kind]
        right = [("<|insight|>" in gens[i] and "<|sql|>" not in gens[i]
                  and any(w in gens[i].lower() for w in words)) for i in idx]
        wrote_sql = [("<|sql|>" in gens[i]) for i in idx]
        out[kind] = {"n": len(idx), "correct": sum(right) / max(1, len(idx)),
                     "wrote_sql_instead": sum(wrote_sql) / max(1, len(idx))}
        print(f"[unseen {kind}] correct {out[kind]['correct']:.2f}, wrote SQL instead {out[kind]['wrote_sql_instead']:.2f}")
    return out


CUSTOM = (
    ("CREATE TABLE employees (id INT, name TEXT, department TEXT, salary REAL, hire_date TEXT);",
     "What is the average salary by department?"),
    ("CREATE TABLE customers (id INT, name TEXT, city TEXT);\nCREATE TABLE orders (id INT, customer_id INT, total REAL, order_date TEXT);",
     "Which customers placed more than 3 orders?"),
    ("CREATE TABLE employees (id INT, name TEXT, department TEXT, salary REAL, hire_date TEXT);",
     "Delete all employees from the Sales department."),
    ("CREATE TABLE employees (id INT, name TEXT, department TEXT, salary REAL, hire_date TEXT);",
     "Who is our best employee?"),
    ("CREATE TABLE inventory (sku TEXT, warehouse TEXT, quantity INT, reorder_level INT);",
     "Which SKUs need to be reordered in each warehouse?"),
)


def custom_prompts(model, tok, device: str) -> list[dict]:
    prompts = [ev.build_prompt(SYSTEMS[0], _user(s, q)) for s, q in CUSTOM]
    gens = generate(model, tok, prompts, ["<|/sql|>", "<|eos|>"], 200, device)
    return [{"question": q, "output": g} for (_, q), g in zip(CUSTOM, gens)]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint", nargs="?", default="artifacts/sft/best.pt")
    p.add_argument("--gretel-n", type=int, default=600)
    p.add_argument("--report", default="reports/eval_sft_report.json")
    args = p.parse_args(argv)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = Tokenizer.from_file(TOKENIZER)
    model = load_model(args.checkpoint, device)
    report: dict = {"checkpoint": args.checkpoint}
    d_sum, d_recs = run_loop(model, tok, domain_items(), device, "domain unseen templates")
    g_sum, g_recs = run_loop(model, tok, gretel_items(args.gretel_n), device, "gretel unseen domains")
    report["domain_unseen_templates"], report["gretel_unseen_domains"] = d_sum, g_sum
    by_template: dict[str, list] = {}
    for r in d_recs:
        by_template.setdefault(r["template"], []).append(r)
    report["domain_by_template"] = {k: ev.summarize(v) for k, v in by_template.items()}
    report["behaviour"] = behaviour_checks(model, tok, device)
    report["behaviour_unseen_schemas"] = unseen_behaviour(model, tok, device)
    report["custom_prompts"] = custom_prompts(model, tok, device)
    report["samples"] = {"domain": d_recs[:3], "gretel": g_recs[:3]}
    with open(args.report, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False, default=str)
    return 0


if __name__ == "__main__":
    sys.exit(main())
