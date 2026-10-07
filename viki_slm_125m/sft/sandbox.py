"""Restricted execution of generated Python for SFT data.

Defence in depth, NOT a security boundary: an AST allow-list (imports, banned names, dunder
access, URL/absolute-path literals), then a fresh interpreter in a temp directory with a
timeout and a minimal environment. Use only for code you generated or that passed the check.
Windows has no resource limits for child processes, so memory is not capped here.
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass

ALLOWED_IMPORTS = frozenset({
    "pandas", "numpy", "matplotlib", "sklearn", "scipy", "seaborn", "statsmodels", "math",
    "statistics", "itertools", "collections", "datetime", "json", "re", "random", "functools",
    "typing", "string", "decimal", "fractions", "operator", "time", "csv", "io", "warnings",
})
BANNED_NAMES = frozenset({"exec", "eval", "compile", "open", "__import__", "input", "breakpoint",
                          "globals", "locals", "vars", "getattr", "setattr", "delattr", "exit",
                          "quit", "memoryview"})
_URL_PREFIXES = ("http://", "https://", "ftp://", "file://", "s3://", "gs://")


@dataclass(frozen=True)
class ExecResult:
    ok: bool
    stdout: str
    error: str
    timed_out: bool


def _bad_string(value: str) -> str | None:
    low = value.strip().lower()
    if low.startswith(_URL_PREFIXES):
        return "url literal not allowed"
    if low.startswith(("/", "\\", "~")) or (len(low) > 2 and low[1] == ":" and low[2] in "/\\"):
        return "absolute path literal not allowed"
    if ".." in low.replace("\\", "/").split("/"):
        return "parent-directory path literal not allowed"
    return None


def check_code_safe(code: str) -> tuple[bool, str]:
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return False, f"syntax error: {exc.msg}"
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] not in ALLOWED_IMPORTS:
                    return False, f"import not allowed: {alias.name}"
        elif isinstance(node, ast.ImportFrom):
            if node.level or (node.module or "").split(".")[0] not in ALLOWED_IMPORTS:
                return False, f"import not allowed: {node.module}"
        elif isinstance(node, ast.Name) and node.id in BANNED_NAMES:
            return False, f"name not allowed: {node.id}"
        elif isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            return False, f"dunder attribute not allowed: {node.attr}"
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            problem = _bad_string(node.value)
            if problem:
                return False, problem
    return True, ""


def _truncate(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit] + "\n...[output truncated]"


def _error_tail(stderr: str, lines: int = 6) -> str:
    tail = [ln for ln in stderr.strip().splitlines() if ln.strip()][-lines:]
    return "\n".join(tail)


def run_python(code: str, files: dict[str, str] | None = None, timeout_s: float = 10.0,
               max_output: int = 4000) -> ExecResult:
    ok, reason = check_code_safe(code)
    if not ok:
        return ExecResult(False, "", f"rejected: {reason}", False)
    env = {"MPLBACKEND": "Agg", "PYTHONHASHSEED": "0", "PYTHONIOENCODING": "utf-8",
           "SYSTEMROOT": os.environ.get("SYSTEMROOT", ""), "PATH": os.environ.get("PATH", "")}
    with tempfile.TemporaryDirectory(prefix="sbx_") as tmp:
        for name, content in (files or {}).items():
            with open(os.path.join(tmp, name), "w", encoding="utf-8") as fh:
                fh.write(content)
        try:
            proc = subprocess.run([sys.executable, "-I", "-c", code], cwd=tmp, env=env,
                                  capture_output=True, text=True, encoding="utf-8",
                                  errors="replace", timeout=timeout_s)
        except subprocess.TimeoutExpired:
            return ExecResult(False, "", f"timeout: exceeded {timeout_s:.0f}s", True)
    out = _truncate(proc.stdout, max_output)
    if proc.returncode != 0:
        return ExecResult(False, out, _error_tail(proc.stderr), False)
    return ExecResult(True, out, "", False)
