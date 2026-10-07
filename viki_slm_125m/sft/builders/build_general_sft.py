"""Build general instruction + Python examples from smoltalk configs (local, free).

Usage: python -m viki_slm_125m.sft.builders.build_general_sft [out_dir]  -> general_train.jsonl, general_val.jsonl, report
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import sys
import urllib.request

import pandas as pd
from tokenizers import Tokenizer

from viki_slm_125m.sft import sft_data
from viki_slm_125m.sft import sft_general as sg

log = logging.getLogger("build_general_sft")
CONFIGS = {  # smoltalk config -> max examples kept
    "self-oss-instruct": 30_000, "smol-summarize": 6_000, "smol-rewrite": 6_000,
    "everyday-conversations": 2_000, "systemchats-30k": 6_000,
}
MAX_TOKENS = 1_024
VAL_FRACTION = 0.01
TOKENIZER_PATH = "artifacts/tokenizer/tokenizer.json"


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "viki-slm-125m"})
    with urllib.request.urlopen(req, timeout=300) as resp:
        return resp.read()


def config_urls() -> dict[str, list[str]]:
    info = json.loads(_get("https://datasets-server.huggingface.co/parquet?dataset=HuggingFaceTB/smoltalk"))
    urls: dict[str, list[str]] = {}
    for f in info["parquet_files"]:
        if f["split"] == "train" and f["config"] in CONFIGS:
            urls.setdefault(f["config"], []).append(f["url"])
    return urls


def main(out_dir: str = "data/sft") -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    os.makedirs(out_dir, exist_ok=True)
    tok = Tokenizer.from_file(TOKENIZER_PATH)

    def enc(text):
        e = tok.encode(text)
        return e.ids, e.offsets

    examples, stats = [], {}
    for cfg, urls in config_urls().items():
        kept = seen = 0
        for url in urls:
            for rec in pd.read_parquet(io.BytesIO(_get(url))).to_dict("records"):
                seen += 1
                turns = sg.messages_to_turns(list(rec["messages"]))
                if turns is None or not all(sg.mostly_ascii(t.text) for t in turns):
                    continue
                if sft_data.encode_example(turns, enc, MAX_TOKENS) is None:
                    continue
                examples.append({"source": cfg, "id": f"{cfg}-{seen}",
                                 "turns": [{"role": t.role, "text": t.text} for t in turns]})
                kept += 1
        stats[cfg] = {"seen": seen, "kept": kept}
        log.info("%s: kept %d of %d", cfg, kept, seen)
    chosen = sg.select_per_source(examples, CONFIGS, seed=0)

    def is_val(i: str) -> bool:
        return int(hashlib.blake2b(i.encode(), digest_size=4).hexdigest(), 16) % 10_000 < VAL_FRACTION * 10_000

    splits = {"general_train": [e for e in chosen if not is_val(e["id"])],
              "general_val": [e for e in chosen if is_val(e["id"])]}
    for name, items in splits.items():
        with open(os.path.join(out_dir, f"{name}.jsonl"), "w", encoding="utf-8") as fh:
            for e in items:
                fh.write(json.dumps(e, ensure_ascii=False) + "\n")
    report = {"per_config": stats, "train": len(splits["general_train"]), "val": len(splits["general_val"])}
    with open(os.path.join(out_dir, "general_report.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    log.info(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(*sys.argv[1:2]))
