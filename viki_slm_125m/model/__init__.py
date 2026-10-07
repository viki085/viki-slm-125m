"""Viki SLM 125M model code: architecture, checkpoint loading, text generation and a small facade.

    from viki_slm_125m.model import VikiSLM
    slm = VikiSLM.load("artifacts/sft_v5/best.pt")
    slm.sql_for("CREATE TABLE t(a INT);", "How many rows are there?")
"""

from viki_slm_125m.model.architecture import build_model, count_parameters
from viki_slm_125m.model.assistant import VikiSLM
from viki_slm_125m.model.checkpoint import load_model, load_tokenizer
from viki_slm_125m.model.generation import generate

__all__ = ["VikiSLM", "build_model", "count_parameters", "generate", "load_model", "load_tokenizer"]
