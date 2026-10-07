"""Local web UI for trying the SFT model (stdlib HTTP server, no extra dependencies).

Usage: python -m viki_slm_125m.app.ui_server [checkpoint] [--port 8000]   then open http://127.0.0.1:8000
Flow per question: schema + question -> model writes SQL -> SQL runs on a real SQLite DB -> model writes insight.
"""

from __future__ import annotations

import argparse
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from viki_slm_125m.eval import evaluator as ev
from viki_slm_125m.eval import eval_sft as es
from viki_slm_125m.sft import sft_data, sql_vote
from viki_slm_125m.sft.domain.domain_data import DOMAINS
from viki_slm_125m.sft.sft_sources import schema_from_context
from viki_slm_125m.sft.sql_verify import build_database, format_result, run_query
from viki_slm_125m.sft.domain import domain_examples as de
from viki_slm_125m.app import ui_modes

from tokenizers import Tokenizer
import torch

UI_DIR = Path(__file__).parent / "static"
MAX_QUESTION, MAX_SCHEMA = 1_000, 8_000
SAMPLE_DB_SEED = 1
_lock = threading.Lock()


def sample_databases() -> dict[str, str]:
    return {name: schema_from_context(ddl) for name, (ddl, _) in DOMAINS.items()}


def open_database(db: str, schema: str):
    """Sample DB by name, or an in-memory DB built from the user's CREATE TABLE / INSERT script."""
    if db in DOMAINS:
        return de.build_db(db, SAMPLE_DB_SEED), sample_databases()[db]
    conn = build_database(schema)
    return conn, schema_from_context(schema)


MAX_REPAIRS = 2
VOTE_SAMPLES = 8
VOTE_TEMPERATURE = 0.7


def _generate_sql(model, tok, device: str, transcript: str) -> str:
    with _lock:
        return es.generate(model, tok, [transcript], ["<|/sql|>", "<|eos|>"], 220, device)[0]


def _sample_sql(model, tok, device: str, prompt: str, n: int) -> list[str | None]:
    with _lock:
        texts = es.generate(model, tok, [prompt] * n, ["<|/sql|>", "<|eos|>"], 220, device, batch=n,
                            do_sample=True, temperature=VOTE_TEMPERATURE, top_p=0.95)
    return [ev.extract_block(t, "sql") for t in texts]


def _repair(model, tok, device: str, conn, out: dict, transcript: str, res):
    """Retry with the database error (trained repair turn). Returns (result, transcript)."""
    for _ in range(MAX_REPAIRS):
        if res.ok:
            break
        out["repairs"].append({"sql": out["sql"], "error": res.error})
        transcript += f"<|result|>\nError: {res.error.strip()}\n<|/result|><|assistant|>"
        current = _generate_sql(model, tok, device, transcript)
        transcript += current
        out["raw_stage1"] += "\n[repair attempt]\n" + current
        retry = ev.extract_block(current, "sql")
        if retry is None:
            break
        out["sql"] = retry
        res = run_query(conn, retry, max_rows=200)
    return res, transcript


def ask(model, tok, device: str, db: str, schema: str, question: str) -> dict:
    """SQL agent loop: greedy query plus sampled alternatives, vote on execution results, repair if none runs,
    then explain the result."""
    conn, schema_text = open_database(db, schema)
    prompt = ev.build_prompt(sft_data.SYSTEM_PROMPTS[0], es._user(schema_text, question))
    first = _generate_sql(model, tok, device, prompt)
    out = {"raw_stage1": first, "sql": ev.extract_block(first, "sql"), "result": None, "insight": None,
           "answer": None, "error": None, "repairs": [], "vote": None}
    if out["sql"] is None:
        out["answer"] = ev.extract_block(first, "insight") or first.replace("<|eos|>", "").strip()
        return out
    greedy_sql = out["sql"]
    choice = sql_vote.vote(conn, [greedy_sql] + _sample_sql(model, tok, device, prompt, VOTE_SAMPLES))
    if choice is not None:
        out["sql"] = choice.sql
        out["vote"] = {"votes": choice.votes, "executed": choice.executed, "of": VOTE_SAMPLES + 1,
                       "changed_from_greedy": choice.sql != greedy_sql}
    transcript = prompt + (first if out["sql"] == greedy_sql else f"<|sql|>\n{out['sql']}\n<|/sql|>")
    res = run_query(conn, out["sql"], max_rows=200)
    res, transcript = _repair(model, tok, device, conn, out, transcript, res)
    if not res.ok:
        out["error"] = res.error
        return out
    table = format_result(res.columns, res.rows[:10], 10, total_rows=len(res.rows))
    out["result"] = {"columns": list(res.columns), "rows": [list(r) for r in res.rows[:50]],
                     "total_rows": len(res.rows), "text": table}
    follow = transcript + f"<|result|>\n{table}\n<|/result|><|assistant|>"
    with _lock:
        second = es.generate(model, tok, [follow], ["<|eos|>"], 140, device)[0]
    out["insight"] = ev.extract_block(second, "insight") or second.replace("<|eos|>", "").strip()
    return out


def make_handler(model, tok, device: str):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code: int, obj: dict) -> None:
            self._send(code, json.dumps(obj, default=str).encode("utf-8"), "application/json; charset=utf-8")

        def do_GET(self):  # noqa: N802
            if self.path == "/api/databases":
                return self._json(200, {"sql": sample_databases(), "python": ui_modes.sample_headers()})
            if self.path in ("/", "/index.html"):
                return self._send(200, (UI_DIR / "index.html").read_bytes(), "text/html; charset=utf-8")
            self._json(404, {"error": "not found"})

        def do_POST(self):  # noqa: N802
            if self.path != "/api/ask":
                return self._json(404, {"error": "not found"})
            try:
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(min(length, 100_000)))
                question = str(body.get("question", "")).strip()
                schema = str(body.get("schema", "")).strip()
                db = str(body.get("db", "custom"))
                mode = str(body.get("mode", "sql"))
                if not question or len(question) > MAX_QUESTION:
                    return self._json(400, {"error": f"question must be 1-{MAX_QUESTION} characters"})
                if mode == "chat":
                    return self._json(200, ui_modes.ask_chat(model, tok, device, schema, question))
                if mode == "python":
                    return self._json(200, ui_modes.ask_python(model, tok, device, db, schema, question))
                if db not in DOMAINS and (not schema or len(schema) > MAX_SCHEMA):
                    return self._json(400, {"error": f"provide a schema (up to {MAX_SCHEMA} characters)"})
                self._json(200, ask(model, tok, device, db, schema, question))
            except Exception as exc:  # report to the UI instead of dropping the connection
                self._json(500, {"error": f"{type(exc).__name__}: {exc}"})

        def log_message(self, fmt, *args):  # quiet
            pass

    return Handler


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint", nargs="?", default="artifacts/sft_v5/best.pt")
    p.add_argument("--port", type=int, default=8000)
    args = p.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = Tokenizer.from_file(es.TOKENIZER)
    model = es.load_model(args.checkpoint, device)
    print(f"Loaded {args.checkpoint} on {device}. Open http://127.0.0.1:{args.port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(model, tok, device)).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
