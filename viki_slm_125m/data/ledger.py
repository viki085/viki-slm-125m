"""Build the dataset license ledger from the Hugging Face API (run locally, no cost).

Usage: python -m viki_slm_125m.data.ledger [output.json]
"""

from __future__ import annotations

import json
import logging
import sys
import urllib.error
import urllib.parse
import urllib.request

from viki_slm_125m import config
from viki_slm_125m.data.measure import license_from_card

log = logging.getLogger("ledger")

EXTRA_IDS: tuple[str, ...] = (
    # SFT candidates
    "HuggingFaceTB/smol-smoltalk", "HuggingFaceTB/smoltalk",
    "b-mc2/sql-create-context", "gretelai/synthetic_text_to_sql",
    "xlangai/spider", "bigcode/self-oss-instruct-sc2-exec-filter-50k",
    "Salesforce/xlam-function-calling-60k",
    "gbharti/finance-alpaca", "Josephgflowers/Finance-Instruct-500k",
    # evaluation / decontamination sets
    "openai/openai_humaneval", "google-research-datasets/mbpp",
    "xlangai/DS-1000", "birdsql/bird_sql_dev_20251106",
)


def _get_json(url: str) -> dict | None:
    req = urllib.request.Request(url, headers={"User-Agent": "viki-slm-125m"})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.load(resp)
    except urllib.error.HTTPError as exc:
        log.warning("HTTP %s for %s", exc.code, url)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        log.warning("failed %s: %s", url, exc)
    return None


def _size_rows(dataset: str) -> int | None:
    url = "https://datasets-server.huggingface.co/size?dataset=" + urllib.parse.quote(dataset)
    data = _get_json(url)
    try:
        return int(data["size"]["dataset"]["num_rows"]) if data else None
    except (KeyError, TypeError, ValueError):
        return None


def entry_for(dataset: str, role: str) -> dict:
    info = _get_json("https://huggingface.co/api/datasets/" + dataset)
    if info is None:
        return {"id": dataset, "role": role, "exists": False}
    card = info.get("cardData") or {}
    return {
        "id": dataset,
        "role": role,
        "exists": True,
        "license": license_from_card(info),
        "gated": info.get("gated"),
        "downloads": info.get("downloads"),
        "configs": [c.get("config_name") for c in card.get("configs") or []],
        "rows": _size_rows(dataset),
    }


def build_ledger() -> list[dict]:
    seen: dict[str, str] = {}
    for src in config.DATA_MIX:
        seen.setdefault(src.hf_id, "pretrain")
    for ds in EXTRA_IDS:
        seen.setdefault(ds, "sft_or_eval")
    return [entry_for(ds, role) for ds, role in seen.items()]


def main(argv: list[str]) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    out = argv[1] if len(argv) > 1 else "reports/license_ledger.json"
    ledger = build_ledger()
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(ledger, fh, indent=2)
    for e in ledger:
        if not e["exists"]:
            log.info("MISSING  %s", e["id"])
        else:
            log.info("%-62s license=%-14s gated=%-5s rows=%s", e["id"], e["license"],
                     e["gated"], e["rows"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
