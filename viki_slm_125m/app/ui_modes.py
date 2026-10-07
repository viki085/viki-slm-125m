"""Python-on-CSV and free-chat modes for the playground (SQL mode lives in ui_server.py)."""

from __future__ import annotations

import re
import threading

from viki_slm_125m.sft.domain import domain_examples as de
from viki_slm_125m.eval import eval_sft as es
from viki_slm_125m.eval import evaluator as ev
from viki_slm_125m.sft import sandbox
from viki_slm_125m.sft.domain.domain_data import DOMAINS, table_csv

SAMPLE_DB_SEED = 1
CHAT_MAX_NEW = 300
CHAT_GEN = {"repetition_penalty": 1.15, "no_repeat_ngram_size": 4}   # stops greedy decoding from looping
FILE_MARKER = re.compile(r"^---\s*(\S+\.csv)\s*$", re.MULTILINE)
lock = threading.Lock()


def extract_code(text: str) -> tuple[str | None, str | None]:
    """(code, format_warning). Tolerates a missing/wrong opening tag if the block is closed."""
    strict = ev.extract_block(text, "python")
    if strict is not None:
        return strict, None
    if "<|/python|>" in text:
        body = text.split("<|/python|>")[0]
        body = re.sub(r"<\|(?:think|python)\|>", "", body).strip()
        return (body or None), "model did not open the code block with <|python|>; code recovered leniently"
    return None, None


def sample_files(db: str) -> dict[str, str]:
    conn = de.build_db(db, SAMPLE_DB_SEED)
    try:
        tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")]
        return {f"{t}.csv": table_csv(conn, t) for t in tables}
    finally:
        conn.close()


def parse_custom_files(text: str) -> dict[str, str]:
    """'--- name.csv' marker lines followed by CSV content -> {name: content}."""
    parts = FILE_MARKER.split(text)
    files = {parts[i]: parts[i + 1].strip() + "\n" for i in range(1, len(parts) - 1, 2)}
    if not files:
        raise ValueError("custom data needs at least one block starting with a line like: --- sales.csv")
    return files


def header_for(files: dict[str, str]) -> str:
    return "\n".join(f"{name}: " + ", ".join(content.splitlines()[0].split(",")) for name, content in files.items())


def sample_headers() -> dict[str, str]:
    return {db: header_for(sample_files(db)) for db in DOMAINS}


def ask_python(model, tok, device: str, db: str, data: str, question: str) -> dict:
    files = sample_files(db) if db in DOMAINS else parse_custom_files(data)
    header = header_for(files)
    prompt = ev.build_prompt(de.PY_SYSTEM_PROMPTS[0], f"<|schema|>\n{header}\n<|/schema|>\n{question.strip()}")
    with lock:
        first = es.generate(model, tok, [prompt], ["<|/python|>", "<|eos|>"], 300, device)[0]
    code, warning = extract_code(first)
    out = {"raw_stage1": first, "code": code, "format_warning": warning, "stdout": None, "insight": None,
           "answer": None, "error": None}
    if out["code"] is None:
        out["answer"] = ev.extract_block(first, "insight") or first.replace("<|eos|>", "").strip()
        return out
    res = sandbox.run_python(out["code"], files=files, timeout_s=20)
    if not res.ok:
        out["error"] = (res.error or "execution failed").strip()[-1500:]
        return out
    out["stdout"] = res.stdout.strip()
    follow = prompt + first + f"<|output|>\n{out['stdout']}\n<|/output|><|assistant|>"
    with lock:
        second = es.generate(model, tok, [follow], ["<|eos|>"], 140, device)[0]
    out["insight"] = ev.extract_block(second, "insight") or second.replace("<|eos|>", "").strip()
    return out


def ask_chat(model, tok, device: str, system: str, question: str) -> dict:
    """General chat / coding help. Never executes anything."""
    prompt = "<|bos|>" + (f"<|system|>{system.strip()}" if system.strip() else "") \
        + f"<|user|>{question.strip().replace('<|', '< |')}<|assistant|>"
    with lock:
        text = es.generate(model, tok, [prompt], ["<|eos|>"], CHAT_MAX_NEW, device, **CHAT_GEN)[0]
    return {"raw_stage1": text, "answer": text.replace("<|eos|>", "").strip(), "error": None}
