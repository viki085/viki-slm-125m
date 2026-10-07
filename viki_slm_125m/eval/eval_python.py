"""Execution accuracy of the pandas (Python-tool) task: generated code is run and compared with gold code.

Questions come from the Python templates but on databases with unseen seeds, so this checks table/column
selection and computation, not memorised data. Templates were seen in training: this is NOT an unseen-template test.

Usage: python -m viki_slm_125m.eval.eval_python [checkpoint] [--seeds 2] [--report reports/eval_python_report.json]
"""

from __future__ import annotations

import argparse
import json
import random
import sys

from tokenizers import Tokenizer

from viki_slm_125m.sft.domain import domain_examples as de
from viki_slm_125m.eval import eval_sft as es
from viki_slm_125m.eval import evaluator as ev
from viki_slm_125m.sft import sandbox
from viki_slm_125m.app import ui_modes
from viki_slm_125m.sft.domain.domain_data import table_csv

SEED_BASE = 900_000


def build_items(seeds: int) -> list[dict]:
    items = []
    for t in de.python_templates():
        for k in range(seeds):
            seed = SEED_BASE + k
            conn = de.build_db(t.domain, seed)
            try:
                question, gold_code = de.instantiate(t, conn, random.Random(seed + 1))
                names = de._tables(t.domain)
                files = {f"{n}.csv": table_csv(conn, n) for n in names}
            finally:
                conn.close()
            gold = sandbox.run_python(gold_code, files=files, timeout_s=20)
            if gold.ok and gold.stdout.strip():
                items.append({"template": t.name, "question": question, "files": files,
                              "gold_rows": de.parse_pipe_output(gold.stdout)[1]})
    return items


def judge(pred_stdout: str, gold_rows: list) -> bool:
    _, rows = de.parse_pipe_output(pred_stdout)
    return [tuple(map(str, r)) for r in rows] == [tuple(map(str, r)) for r in gold_rows]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint", nargs="?", default="artifacts/sft_v3/best.pt")
    p.add_argument("--seeds", type=int, default=2)
    p.add_argument("--report", default="reports/reports/eval_python_report.json")
    args = p.parse_args(argv)
    device = "cuda"
    tok = Tokenizer.from_file(es.TOKENIZER)
    model = es.load_model(args.checkpoint, device)
    items = build_items(args.seeds)
    prompts = [ev.build_prompt(de.PY_SYSTEM_PROMPTS[0],
                               f"<|schema|>\n{ui_modes.header_for(it['files'])}\n<|/schema|>\n{it['question']}")
               for it in items]
    gens = es.generate(model, tok, prompts, ["<|/python|>", "<|eos|>"], 300, device)
    recs = []
    for it, g in zip(items, gens):
        code, warning = ui_modes.extract_code(g)
        status, ok = "no_code", False
        if code:
            res = sandbox.run_python(code, files=it["files"], timeout_s=20)
            status = "error" if not res.ok else "ran"
            ok = res.ok and judge(res.stdout, it["gold_rows"])
        recs.append({"template": it["template"], "question": it["question"], "status": status,
                     "match": ok, "strict_format": warning is None and code is not None, "code": code})
    n = len(recs)
    summary = {"checkpoint": args.checkpoint, "n": n,
               "execution_accuracy": sum(r["match"] for r in recs) / n,
               "runs_rate": sum(r["status"] == "ran" for r in recs) / n,
               "strict_format_rate": sum(r["strict_format"] for r in recs) / n}
    by_t: dict[str, list] = {}
    for r in recs:
        by_t.setdefault(r["template"], []).append(r["match"])
    summary["by_template"] = {k: sum(v) / len(v) for k, v in by_t.items()}
    print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in summary.items() if k != "by_template"}))
    with open(args.report, "w", encoding="utf-8") as fh:
        json.dump({"summary": summary, "records": recs}, fh, indent=2, ensure_ascii=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
