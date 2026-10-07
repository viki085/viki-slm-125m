"""Modal app for the Viki 125M SLM build. Stage 1 (feasibility) functions only so far."""

from __future__ import annotations

import modal

from viki_slm_125m import config
from viki_slm_125m.data import tokenizer_lib

app = modal.App(config.PROJECT)

_base = modal.Image.debian_slim(python_version="3.12").pip_install(
    "datasets==3.6.0",
    "huggingface_hub==0.34.4",
    "pyarrow==17.0.0",
    "boto3==1.35.99",
    "sqlglot==25.34.1",
    "langdetect==1.0.9",
    "numpy==2.1.3",
    "datasketch==1.6.5",
)
# All pip/apt steps MUST come before add_local_* (Modal rule).
cpu_image = _base.add_local_python_source("viki_slm_125m")

volume = modal.Volume.from_name(config.VOLUME_NAME, create_if_missing=True)
VOLUMES = {config.DATA_ROOT: volume}

SWH_BUCKET = "softwareheritage"
SWH_REGION = "us-east-1"


def _rows_for(dataset: str, config_name: str | None) -> int | None:
    import json
    import urllib.parse
    import urllib.request

    url = "https://datasets-server.huggingface.co/size?dataset=" + urllib.parse.quote(dataset)
    try:
        with urllib.request.urlopen(url, timeout=60) as resp:
            data = json.load(resp)
        for c in data["size"]["configs"]:
            if c["config"] == (config_name or "default"):
                return int(c["num_rows"])
    except Exception as exc:  # network/shape errors: report as unknown, never crash
        print(f"  [rows] {dataset}/{config_name}: {exc}")
    return None


@app.function(image=cpu_image, volumes=VOLUMES, timeout=60 * 20)
def measure_text_source(source_name: str, n_docs: int = 2000) -> dict:
    """Stream n_docs of a text/notebook source; report raw chars per doc and projected tokens.

    Raw (uncleaned) figures are an UPPER BOUND; Phase 2 cleaning will keep less.
    """
    from datasets import load_dataset

    from viki_slm_125m.data.measure import estimate_clean_tokens

    src = next(s for s in config.DATA_MIX if s.name == source_name)
    ds = load_dataset(src.hf_id, src.config_name, split=src.split, streaming=True)
    docs = chars = 0
    for rec in ds:
        text = rec.get(src.text_field) or ""
        chars += len(str(text))
        docs += 1
        if docs >= n_docs:
            break
    avg = chars / docs if docs else 0.0
    rows = _rows_for(src.hf_id, src.config_name)
    est = estimate_clean_tokens(rows, avg, config.CHARS_PER_TOKEN) if rows else None
    out = {"source": source_name, "docs_sampled": docs, "avg_chars": round(avg),
           "total_rows": rows, "raw_tokens_upper_bound": est,
           "planned_budget": src.token_budget}
    print(out)
    return out


@app.function(image=cpu_image, volumes=VOLUMES, timeout=60 * 30, memory=8_192)
def measure_stack_edu(config_name: str, n_rows: int = 300_000) -> dict:
    """Permissive-license share of stack-edu metadata (shuffled sample)."""
    from datasets import load_dataset

    from viki_slm_125m.data.measure import summarize_stack_edu

    ds = load_dataset("HuggingFaceTB/stack-edu", config_name, split="train", streaming=True)
    ds = ds.shuffle(seed=0, buffer_size=20_000)
    rows = []
    for rec in ds:
        rows.append((rec["license_type"], int(rec["length_bytes"]), int(rec["int_score"])))
        if len(rows) >= n_rows:
            break
    out = {"config": config_name, **summarize_stack_edu(rows)}
    print(out)
    return out


@app.function(image=cpu_image, volumes=VOLUMES, timeout=60 * 40, region="us-east", cpu=4.0)
def swh_fetch_test(config_name: str, n_files: int = 10_000, threads: int = 64) -> dict:
    """Time anonymous Software Heritage fetches for permissive, int_score>=4 files."""
    import gzip
    import time
    from concurrent.futures import ThreadPoolExecutor

    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config
    from botocore.exceptions import ClientError
    from datasets import load_dataset

    from viki_slm_125m.data.measure import summarize_fetch

    ds = load_dataset("HuggingFaceTB/stack-edu", config_name, split="train", streaming=True)
    ds = ds.shuffle(seed=1, buffer_size=20_000)
    blobs: list[tuple[str, str]] = []
    for rec in ds:
        if rec["license_type"] == "permissive" and rec["int_score"] >= 4 \
                and 200 <= rec["length_bytes"] <= 100_000:
            blobs.append((rec["blob_id"], rec["src_encoding"] or "utf-8"))
            if len(blobs) >= n_files:
                break

    s3 = boto3.client("s3", region_name=SWH_REGION,
                      config=Config(signature_version=UNSIGNED, max_pool_connections=threads,
                                    retries={"max_attempts": 3, "mode": "adaptive"}))

    def fetch(item: tuple[str, str]) -> tuple[str, int, str]:
        blob_id, enc = item
        try:
            obj = s3.get_object(Bucket=SWH_BUCKET, Key=f"content/{blob_id}")
            raw = gzip.decompress(obj["Body"].read())
            text = raw.decode(enc if enc else "utf-8", errors="ignore")
            return "ok", len(text), text[:200]
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code", "")
            return ("missing" if code in ("NoSuchKey", "404") else "error:" + code), 0, ""
        except Exception as exc:  # decode/lookup errors count as errors, not crashes
            return "error:" + type(exc).__name__, 0, ""

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=threads) as pool:
        results = list(pool.map(fetch, blobs))
    elapsed = time.time() - t0
    n_ok = sum(1 for r in results if r[0] == "ok")
    n_missing = sum(1 for r in results if r[0] == "missing")
    n_err = len(results) - n_ok - n_missing
    total_chars = sum(r[1] for r in results)
    errs: dict[str, int] = {}
    for r in results:
        if r[0].startswith("error"):
            errs[r[0]] = errs.get(r[0], 0) + 1
    out = {"config": config_name, "threads": threads, "error_kinds": errs,
           "avg_chars_ok": round(total_chars / n_ok) if n_ok else 0,
           **summarize_fetch(n_ok, n_missing, n_err, total_chars, elapsed)}
    print(out)
    return out


# ------------------------------------------------------------------ Stage 2: fetch + clean
PILOT_DIR = f"{config.DATA_ROOT}/pilot"
FETCH_THREADS = 64
FETCH_BATCH = 512
SAMPLE_CHARS = 400


def _parquet_urls(hf_id: str, config_name: str | None, split: str = "train") -> list[str]:
    import json
    import urllib.parse
    import urllib.request

    api = "https://datasets-server.huggingface.co/parquet?dataset=" + urllib.parse.quote(hf_id)
    req = urllib.request.Request(api, headers={"User-Agent": "viki-slm-125m"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.load(resp)
    cfg = config_name or "default"
    return sorted(f["url"] for f in data.get("parquet_files", [])
                  if f.get("config") == cfg and f.get("split") == split)


def _make_swh_client():
    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config

    return boto3.client("s3", region_name=SWH_REGION,
                        config=Config(signature_version=UNSIGNED, max_pool_connections=FETCH_THREADS,
                                      retries={"max_attempts": 3, "mode": "adaptive"}))


def _fetch_blobs(s3, items: list[tuple[str, str | None]]) -> list[str | None]:
    """Fetch and decode file texts from Software Heritage; None where unavailable."""
    import gzip
    from concurrent.futures import ThreadPoolExecutor

    from viki_slm_125m.data.pipeline import decode_text

    def one(item):
        blob_id, enc = item
        try:
            body = s3.get_object(Bucket=SWH_BUCKET, Key=f"content/{blob_id}")["Body"].read()
            return decode_text(gzip.decompress(body), enc)
        except Exception as exc:  # missing/throttled/corrupt: counted by the caller, never fatal
            print(f"  [swh] {blob_id}: {type(exc).__name__}")
            return None

    with ThreadPoolExecutor(max_workers=FETCH_THREADS) as pool:
        return list(pool.map(one, items))


def _records(source, url: str, worker: int, n_workers: int, limit: int):
    """Yield (text_or_None, meta) for this worker's rows; code rows are fetched in batches."""
    from datasets import load_dataset

    from viki_slm_125m.data.pipeline import accept_code_row, owns_row

    ds = load_dataset("parquet", data_files=url, split="train", streaming=True)
    if not source.needs_fetch:
        n = 0
        for i, rec in enumerate(ds):
            if owns_row(i, worker, n_workers):
                yield str(rec.get(source.text_field) or "")
                n += 1
                if limit and n >= limit:
                    return
        return
    s3 = _make_swh_client()
    batch: list[dict] = []
    taken = 0

    def flush():
        texts = _fetch_blobs(s3, [(r["blob_id"], r.get("src_encoding")) for r in batch])
        return [t if t is not None else "" for t in texts]

    for i, rec in enumerate(ds):
        if not owns_row(i, worker, n_workers) or not accept_code_row(rec):
            continue
        batch.append(rec)
        taken += 1
        if len(batch) >= FETCH_BATCH or (limit and taken >= limit):
            yield from flush()
            batch = []
            if limit and taken >= limit:
                return
    if batch:
        yield from flush()


@app.function(image=cpu_image, volumes=VOLUMES, timeout=60 * 60, cpu=4.0, memory=8_192)
def clean_shard(source_name: str, url: str, shard: int, worker: int, n_workers: int,
                token_cap: int, limit: int = 0, out_root: str = config.CLEAN_DIR) -> dict:
    """Stream one worker's slice of a parquet shard, clean it, write JSONL, report drops."""
    import json
    import os

    import logging

    from viki_slm_125m.data.cleaning import clean_document
    from viki_slm_125m.data.pipeline import est_tokens

    logging.getLogger("sqlglot").setLevel(logging.ERROR)  # silence 'falling back to Command' noise
    src = next(s for s in config.DATA_MIX if s.name == source_name)
    out_dir = f"{out_root}/{source_name}"
    os.makedirs(out_dir, exist_ok=True)
    path = f"{out_dir}/shard-{shard:03d}-{worker:02d}.jsonl"
    streamed = kept = clean_chars = 0
    reasons: dict[str, int] = {}
    samples = {"kept": [], "dropped": []}
    with open(path, "w", encoding="utf-8") as fh:
        for text in _records(src, url, worker, n_workers, limit):
            streamed += 1
            if not text:
                reasons["empty_or_unfetched"] = reasons.get("empty_or_unfetched", 0) + 1
                continue
            r = clean_document(text, src.kind)
            reasons[r.reason] = reasons.get(r.reason, 0) + 1
            bucket = samples["kept" if r.kept else "dropped"]
            if len(bucket) < 3:
                bucket.append({"reason": r.reason, "text": (r.text or text)[:SAMPLE_CHARS]})
            if not r.kept:
                continue
            fh.write(json.dumps({"text": r.text}, ensure_ascii=False) + "\n")
            kept += 1
            clean_chars += r.clean_chars
            if est_tokens(clean_chars, config.CHARS_PER_TOKEN) >= token_cap:
                break
    volume.commit()
    report = {"source": source_name, "shard": shard, "worker": worker, "streamed": streamed,
              "kept": kept, "est_tokens": est_tokens(clean_chars, config.CHARS_PER_TOKEN),
              "reasons": reasons, "samples": samples}
    print({k: v for k, v in report.items() if k != "samples"})
    return report


@app.local_entrypoint()
def clean_pilot(rows: int = 400):
    """Pilot: one worker, ~`rows` documents per source, written under /data/pilot."""
    import json

    handles = {}
    for s in config.DATA_MIX:
        urls = _parquet_urls(s.hf_id, s.config_name)
        if not urls:
            print(f"!! no parquet urls for {s.name}")
            continue
        handles[s.name] = clean_shard.spawn(s.name, urls[0], 0, 0, 1, 10**12, rows, PILOT_DIR)
    report = {name: h.get() for name, h in handles.items()}
    with open("reports/pilot_report.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    for name, r in report.items():
        rate = r["kept"] / r["streamed"] if r["streamed"] else 0
        print(f"{name:<12} streamed={r['streamed']:>5} kept={r['kept']:>5} ({rate:.0%}) "
              f"tokens~{r['est_tokens']:>9,} drops={r['reasons']}")


# (workers per parquet shard, max shards to use) per source: tuned for ~100 containers
CLEAN_PLAN = {
    "python": (8, 5), "sql": (16, 1), "notebooks": (4, 8), "finance-sec": (4, 8),
    "fineweb-edu": (1, 24), "cosmopedia": (1, 24), "math": (1, 24),
}


@app.function(image=cpu_image, volumes=VOLUMES)
def save_report(name: str, report: dict) -> None:
    import json
    import os

    os.makedirs(config.RAW_META_DIR, exist_ok=True)
    with open(f"{config.RAW_META_DIR}/{name}.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    volume.commit()


@app.local_entrypoint()
def clean(only: str = ""):
    """Full fetch + clean fan-out into /data/clean (run only after pilot review)."""
    from viki_slm_125m.data.pipeline import merge_reports, worker_budget

    work = []
    for s in config.DATA_MIX:
        if only and s.name != only:
            continue
        per_shard, max_shards = CLEAN_PLAN[s.name]
        urls = _parquet_urls(s.hf_id, s.config_name)[:max_shards]
        n_workers = len(urls) * per_shard
        cap = worker_budget(s.token_budget, n_workers)
        for shard, url in enumerate(urls):
            for w in range(per_shard):
                work.append((s.name, url, shard, w, per_shard, cap))
        print(f"{s.name:<12} {len(urls)} shard(s) x {per_shard} workers, cap ~{cap / 1e6:.1f}M tokens each")
    print(f"Launching {len(work)} clean workers...")
    results = list(clean_shard.starmap(work))
    by_source: dict[str, list] = {}
    for r in results:
        by_source.setdefault(r["source"], []).append({k: v for k, v in r.items() if k != "samples"})
    report = {name: merge_reports(rs) for name, rs in by_source.items()}
    print("PHASE 2 CLEAN REPORT")
    total = 0
    for name, a in report.items():
        total += a["est_tokens"]
        print(f"  {name:<12} streamed={a['streamed']:>9} kept={a['kept']:>9} "
              f"est_tokens={a['est_tokens'] / 1e9:.3f}B drops={a['reasons']}")
    print(f"  TOTAL est_clean_tokens={total / 1e9:.2f}B")
    save_report.remote("phase2_clean_report" + (f"_{only}" if only else ""), report)


# ------------------------------------------------------------ Stage 3: dedup + decontamination
TMP_DIR = f"{config.DATA_ROOT}/tmp"
EXACT_DUPS_PATH = f"{TMP_DIR}/exact_dups.json"
NEAR_DUPS_PATH = f"{TMP_DIR}/near_dups.json"
NEAR_DUP_SOURCES = {"finance-sec"}
# benchmark name -> (hf_id, parquet config, splits, text fields); dev/test only, never train splits
BENCHMARKS = {
    "humaneval": ("openai/openai_humaneval", "openai_humaneval", ("test",),
                  ("prompt", "canonical_solution", "test")),
    "mbpp": ("google-research-datasets/mbpp", "full", ("train", "validation", "test", "prompt"),
             ("text", "code", "test_list")),
    "spider": ("xlangai/spider", "spider", ("validation",), ("question", "query")),
    "bird": ("birdsql/bird_sql_dev_20251106", "default", ("dev_20251106",),
             ("question", "evidence", "SQL")),
    "ds1000": ("xlangai/DS-1000", "default", ("test",), ("prompt", "reference_code")),
}


def _kind_of(source_name: str) -> str:
    return next(s.kind for s in config.DATA_MIX if s.name == source_name)


def _benchmark_texts() -> dict[str, list[str]]:
    import json
    import urllib.parse
    import urllib.request

    from datasets import load_dataset

    from viki_slm_125m.data.dedup import record_text

    out: dict[str, list[str]] = {}
    for name, (hf_id, cfg, splits, fields) in BENCHMARKS.items():
        api = "https://datasets-server.huggingface.co/parquet?dataset=" + urllib.parse.quote(hf_id)
        with urllib.request.urlopen(api, timeout=60) as resp:
            files = json.load(resp)["parquet_files"]
        urls = [f["url"] for f in files if f["config"] == cfg and f["split"] in splits]
        texts: list[str] = []
        for url in urls:
            for rec in load_dataset("parquet", data_files=url, split="train"):
                texts.append(record_text(rec, fields))
        out[name] = texts
        print(f"  [decontam] {name}: {len(texts)} records from {len(urls)} file(s)")
    return out


@app.function(image=cpu_image, volumes=VOLUMES)
def list_clean_files(only: str = "") -> list[tuple[str, str]]:
    import os

    volume.reload()
    out = []
    for s in config.DATA_MIX:
        if only and s.name != only:
            continue
        d = f"{config.CLEAN_DIR}/{s.name}"
        if os.path.isdir(d):
            out += [(s.name, f) for f in sorted(os.listdir(d)) if f.endswith(".jsonl")]
    return out


@app.function(image=cpu_image, volumes=VOLUMES, timeout=60 * 30, cpu=2.0)
def hash_file(source_name: str, filename: str) -> int:
    """Phase 3a: write a uint64 hash per document of one clean file."""
    import json
    import os

    import numpy as np

    from viki_slm_125m.data.dedup import canonical_for_hash, exact_hash

    volume.reload()
    kind = _kind_of(source_name)
    hashes = []
    with open(f"{config.CLEAN_DIR}/{source_name}/{filename}", encoding="utf-8") as fh:
        for line in fh:
            hashes.append(exact_hash(canonical_for_hash(json.loads(line)["text"], kind)))
    out_dir = f"{TMP_DIR}/hashes/{source_name}"
    os.makedirs(out_dir, exist_ok=True)
    np.save(f"{out_dir}/{filename}.npy", np.asarray(hashes, dtype=np.uint64))
    volume.commit()
    return len(hashes)


@app.function(image=cpu_image, volumes=VOLUMES, timeout=60 * 30, memory=16_384)
def build_exact_dups(files: list[tuple[str, str]]) -> dict:
    """Phase 3b: global exact-duplicate indices (first occurrence wins, in file order)."""
    import json

    import numpy as np

    from viki_slm_125m.data.dedup import duplicate_indices

    volume.reload()
    arrays = [np.load(f"{TMP_DIR}/hashes/{s}/{f}.npy") for s, f in files]
    dups = duplicate_indices(arrays)
    mapping = {f"{s}/{f}": d.tolist() for (s, f), d in zip(files, dups) if len(d)}
    with open(EXACT_DUPS_PATH, "w", encoding="utf-8") as fh:
        json.dump(mapping, fh)
    volume.commit()
    total = int(sum(len(d) for d in dups))
    docs = int(sum(len(a) for a in arrays))
    print(f"[exact-dups] {total:,} duplicates of {docs:,} docs")
    return {"exact_duplicates": total, "docs": docs}


@app.function(image=cpu_image, volumes=VOLUMES, timeout=60 * 60, memory=8_192, cpu=2.0)
def build_near_dups(files: list[tuple[str, str]]) -> dict:
    """Phase 3c: MinHash near-duplicates for the sources in NEAR_DUP_SOURCES (streamed)."""
    import json

    from viki_slm_125m.data.dedup import near_duplicate_ids

    volume.reload()
    targets = [(s, f) for s, f in files if s in NEAR_DUP_SOURCES]

    def pairs():
        for s, f in targets:
            with open(f"{config.CLEAN_DIR}/{s}/{f}", encoding="utf-8") as fh:
                for idx, line in enumerate(fh):
                    yield f"{s}/{f}:{idx}", json.loads(line)["text"][:60_000]

    near: dict[str, list[int]] = {}
    for doc_id in near_duplicate_ids(pairs()):
        key, idx = doc_id.rsplit(":", 1)
        near.setdefault(key, []).append(int(idx))
    with open(NEAR_DUPS_PATH, "w", encoding="utf-8") as fh:
        json.dump(near, fh)
    volume.commit()
    total = sum(len(v) for v in near.values())
    print(f"[near-dups] {total:,} near-duplicates")
    return {"near_duplicates": total}


@app.function(image=cpu_image, volumes=VOLUMES, timeout=60 * 40, cpu=4.0, memory=8_192)
def write_corpus_shard(source_name: str, filename: str) -> dict:
    """Phase 3d: drop exact/near duplicates and benchmark-contaminated documents."""
    import json
    import os

    from viki_slm_125m.data.dedup import build_contamination_grams, find_contamination
    from viki_slm_125m.data.pipeline import est_tokens

    volume.reload()
    key = f"{source_name}/{filename}"
    with open(EXACT_DUPS_PATH, encoding="utf-8") as fh:
        exact = set(json.load(fh).get(key, []))
    with open(NEAR_DUPS_PATH, encoding="utf-8") as fh:
        near = set(json.load(fh).get(key, []))
    grams = build_contamination_grams(_benchmark_texts())
    out_dir = f"{config.CORPUS_DIR}/{source_name}"
    os.makedirs(out_dir, exist_ok=True)
    reasons: dict[str, int] = {"kept": 0, "exact_dup": 0, "near_dup": 0}
    contaminated: dict[str, int] = {}
    chars = 0
    with open(f"{config.CLEAN_DIR}/{key}", encoding="utf-8") as fin, \
            open(f"{out_dir}/{filename}", "w", encoding="utf-8") as fout:
        for idx, line in enumerate(fin):
            if idx in exact:
                reasons["exact_dup"] += 1
                continue
            if idx in near:
                reasons["near_dup"] += 1
                continue
            text = json.loads(line)["text"]
            hit = find_contamination(text, grams)
            if hit:
                contaminated[hit] = contaminated.get(hit, 0) + 1
                continue
            fout.write(line)
            reasons["kept"] += 1
            chars += len(text)
    volume.commit()
    report = {"source": source_name, "file": filename, "reasons": reasons,
              "contaminated": contaminated, "kept": reasons["kept"],
              "est_tokens": est_tokens(chars, config.CHARS_PER_TOKEN)}
    print(report)
    return report


@app.local_entrypoint()
def dedup(only: str = "", rewrite_only: bool = False):
    """Stage 3 end to end; `--only <source>` runs one source; `--rewrite-only` redoes step 3d."""
    import json

    files = list_clean_files.remote(only)
    print(f"{len(files)} clean files")
    exact: dict = {}
    near: dict = {}
    if not rewrite_only:
        print("3a hashing...")
        list(hash_file.starmap(files))
        print("3b exact duplicates...")
        exact = build_exact_dups.remote(files)
        print("3c near duplicates...")
        near = build_near_dups.remote(files)
    print("3d writing corpus...")
    results = list(write_corpus_shard.starmap(files))
    report: dict[str, dict] = {}
    for r in results:
        a = report.setdefault(r["source"], {"kept": 0, "est_tokens": 0, "exact_dup": 0,
                                            "near_dup": 0, "contaminated": {}})
        a["kept"] += r["kept"]
        a["est_tokens"] += r["est_tokens"]
        a["exact_dup"] += r["reasons"]["exact_dup"]
        a["near_dup"] += r["reasons"]["near_dup"]
        for b, n in r["contaminated"].items():
            a["contaminated"][b] = a["contaminated"].get(b, 0) + n
    print("PHASE 3 REPORT", exact, near)
    total = 0
    for name, a in report.items():
        total += a["est_tokens"]
        print(f"  {name:<12} kept={a['kept']:>9} est_tokens={a['est_tokens'] / 1e9:.3f}B "
              f"exact={a['exact_dup']} near={a['near_dup']} contaminated={a['contaminated']}")
    print(f"  TOTAL corpus est tokens: {total / 1e9:.2f}B")
    full = {"global": {**exact, **near}, "sources": report}
    save_report.remote("phase3_dedup_report" + (f"_{only}" if only else ""), full)
    with open("reports/phase3_dedup_report.json", "w", encoding="utf-8") as fh:
        json.dump(full, fh, indent=2)


# ------------------------------------------------------------------ Stage 4: tokenizer
ml_image = _base.pip_install("transformers==4.46.3").add_local_python_source("viki_slm_125m")
TOKENIZER_SAMPLE_CHARS = 1_200_000_000   # about 300M tokens
EVAL_DOCS_PER_SOURCE = 300


def _corpus_docs(path: str):
    import json

    with open(path, encoding="utf-8") as fh:
        for line in fh:
            yield json.loads(line)["text"]


def _gather(role: str, per_source_chars: dict[str, int], max_docs: int = 0):
    """Yield (source, text) from every corpus file, spreading each source's quota over its files."""
    import os

    for name, quota in per_source_chars.items():
        d = f"{config.CORPUS_DIR}/{name}"
        files = sorted(f for f in os.listdir(d) if f.endswith(".jsonl"))
        per_file = max(1, quota // len(files))
        n = 0
        for f in files:
            for text in tokenizer_lib.sample_docs(_corpus_docs(f"{d}/{f}"), role, per_file):
                yield name, text
                n += 1
                if max_docs and n >= max_docs:
                    break
            if max_docs and n >= max_docs:
                break


@app.function(image=ml_image, volumes=VOLUMES, timeout=60 * 60, cpu=16.0, memory=32_768)
def train_tokenizer(sample_chars: int = TOKENIZER_SAMPLE_CHARS) -> dict:
    import os

    from transformers import AutoTokenizer, PreTrainedTokenizerFast

    volume.reload()
    quotas = tokenizer_lib.source_quotas(sample_chars, tokenizer_lib.TOKENIZER_SHARES)
    specials = config.all_special_tokens()
    print("training BPE on", {k: f"{v / 1e6:.0f}M chars" for k, v in quotas.items()})
    tok = tokenizer_lib.build_tokenizer(
        (text for _, text in _gather("train", quotas)), config.MODEL.vocab_size, specials)
    ids = tokenizer_lib.special_token_ids(tok, specials)
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        bos_token=config.SPECIAL_TOKENS["bos_token"], eos_token=config.SPECIAL_TOKENS["eos_token"],
        pad_token=config.SPECIAL_TOKENS["pad_token"], unk_token=config.SPECIAL_TOKENS["unk_token"],
        additional_special_tokens=list(config.EXTRA_CHAT_TOKENS) + list(config.AGENT_TOKENS))
    os.makedirs(config.TOKENIZER_DIR, exist_ok=True)
    fast.save_pretrained(config.TOKENIZER_DIR)
    volume.commit()

    # --- checks on held-out documents (disjoint from the training sample)
    eval_quota = {k: 20_000 * EVAL_DOCS_PER_SOURCE for k in quotas}
    by_source: dict[str, list[str]] = {}
    for name, text in _gather("eval", eval_quota, max_docs=EVAL_DOCS_PER_SOURCE):
        by_source.setdefault(name, []).append(text)
    reloaded = AutoTokenizer.from_pretrained(config.TOKENIZER_DIR)
    baseline = AutoTokenizer.from_pretrained("gpt2")
    fertility = {}
    for name, texts in by_source.items():
        chars = sum(len(t) for t in texts)
        ours = sum(len(reloaded(t)["input_ids"]) for t in texts)
        base = sum(len(baseline(t)["input_ids"]) for t in texts)
        fertility[name] = {"docs": len(texts), "chars_per_token": round(chars / ours, 3),
                           "gpt2_chars_per_token": round(chars / base, 3),
                           "vs_gpt2": round(base / ours, 3)}
    all_eval = [t for ts in by_source.values() for t in ts]
    failures = tokenizer_lib.roundtrip_failures(tok, all_eval)
    hf_single = {t: reloaded.convert_tokens_to_ids(t) for t in specials}
    report = {
        "vocab_size": fast.vocab_size, "special_ids": ids,
        "special_ids_match_hf": all(hf_single[t] == i for t, i in ids.items()),
        "special_tokens_single_id": all(len(reloaded.encode(t, add_special_tokens=False)) == 1
                                        for t in specials),
        "roundtrip_failures": len(failures), "roundtrip_checked": len(all_eval),
        "fertility": fertility,
        "digit_split": reloaded.tokenize("12345"),
        "indent_12_spaces": reloaded.tokenize("\n" + " " * 12 + "x = 1"),
    }
    print(report)
    return report


@app.local_entrypoint()
def tokenizer():
    import json

    report = train_tokenizer.remote()
    with open("reports/phase4_tokenizer_report.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    save_report.remote("phase4_tokenizer_report", report)
    print(json.dumps(report, indent=2))


# ------------------------------------------------------- Stage 5: tokenize + pack into windows
ENCODE_BATCH = 1_000


@app.function(image=ml_image, volumes=VOLUMES)
def list_corpus_files(only: str = "", limit_files: int = 0) -> list[tuple[str, str]]:
    import os

    volume.reload()
    out: list[tuple[str, str]] = []
    for s in config.DATA_MIX:
        if only and s.name != only:
            continue
        d = f"{config.CORPUS_DIR}/{s.name}"
        files = sorted(f for f in os.listdir(d) if f.endswith(".jsonl")) if os.path.isdir(d) else []
        out += [(s.name, f) for f in (files[:limit_files] if limit_files else files)]
    return out


@app.function(image=ml_image, volumes=VOLUMES, timeout=60 * 40, cpu=4.0, memory=8_192)
def tokenize_file(source_name: str, filename: str) -> dict:
    """Tokenize one corpus file and write packed uint16 windows (99/1 train/val)."""
    import json
    import os

    from transformers import AutoTokenizer

    from viki_slm_125m.data.packing import pack_stream, route_window

    volume.reload()
    src = next(s for s in config.DATA_MIX if s.name == source_name)
    tok = AutoTokenizer.from_pretrained(config.TOKENIZER_DIR)
    eos_id = tok.convert_tokens_to_ids(config.SPECIAL_TOKENS["eos_token"])
    stem = filename[: -len(".jsonl")]
    os.makedirs(config.TRAIN_TOKENS_DIR, exist_ok=True)
    os.makedirs(config.VAL_TOKENS_DIR, exist_ok=True)
    train_path = f"{config.TRAIN_TOKENS_DIR}/{source_name}-{stem}.bin"
    val_path = f"{config.VAL_TOKENS_DIR}/{source_name}-{stem}.bin"

    def ids_iter():
        batch: list[str] = []
        with open(f"{config.CORPUS_DIR}/{source_name}/{filename}", encoding="utf-8") as fh:
            for line in fh:
                batch.append(json.loads(line)["text"])
                if len(batch) >= ENCODE_BATCH:
                    yield from tok(batch, add_special_tokens=False)["input_ids"]
                    batch = []
        if batch:
            yield from tok(batch, add_special_tokens=False)["input_ids"]

    n_train = n_val = 0
    with open(train_path, "wb") as ftr, open(val_path, "wb") as fva:
        for i, window in enumerate(pack_stream(ids_iter(), eos_id, config.SEQ_LEN)):
            if route_window(i, config.VAL_EVERY_N_WINDOWS) == "val":
                window.tofile(fva)
                n_val += 1
            else:
                window.tofile(ftr)
                n_train += 1
    volume.commit()
    report = {"source": source_name, "file": filename, "train_windows": n_train,
              "val_windows": n_val, "epochs": src.epochs}
    print(report)
    return report


@app.function(image=ml_image, volumes=VOLUMES, timeout=60 * 10)
def inspect_tokens(source_name: str, filename: str, n_windows: int = 2) -> dict:
    """Sanity-check a written shard: dtype/size, id range, decoded excerpt."""
    import numpy as np
    from transformers import AutoTokenizer

    volume.reload()
    tok = AutoTokenizer.from_pretrained(config.TOKENIZER_DIR)
    stem = filename[: -len(".jsonl")]
    path = f"{config.TRAIN_TOKENS_DIR}/{source_name}-{stem}.bin"
    data = np.fromfile(path, dtype=np.uint16)
    out = {"file": path, "bytes": os_size(path), "windows": len(data) // config.SEQ_LEN,
           "size_matches": len(data) % config.SEQ_LEN == 0,
           "max_id": int(data.max()), "vocab": config.MODEL.vocab_size,
           "eos_count_first_window": int((data[:config.SEQ_LEN] == tok.eos_token_id).sum())}
    out["excerpt"] = tok.decode(data[:config.SEQ_LEN][:120].tolist())
    return out


def os_size(path: str) -> int:
    import os

    return os.path.getsize(path)


@app.function(image=ml_image, volumes=VOLUMES)
def write_token_index(results: list, name: str) -> dict:
    import json

    from viki_slm_125m.data.packing import build_index

    index = build_index(results, config.SEQ_LEN, config.TOKENS_DTYPE)
    with open(f"{config.TOKENS_DIR}/{name}", "w", encoding="utf-8") as fh:
        json.dump(index, fh, indent=2)
    volume.commit()
    return index


@app.local_entrypoint()
def tokenize(only: str = "", limit_files: int = 0):
    """Stage 5. Pilot: `--only python --limit-files 1`. Full: no flags."""
    import json

    files = list_corpus_files.remote(only, limit_files)
    print(f"tokenizing {len(files)} corpus files")
    results = list(tokenize_file.starmap(files))
    name = "index_pilot.json" if (only or limit_files) else "index.json"
    index = write_token_index.remote(results, name)
    with open(name, "w", encoding="utf-8") as fh:
        json.dump(index, fh, indent=2)
    print(f"{name}: train={index['train_tokens'] / 1e9:.3f}B tok ({index['train_windows']} windows), "
          f"val={index['val_tokens'] / 1e6:.1f}M tok ({index['val_windows']} windows), "
          f"effective(with epochs)={index['effective_train_tokens'] / 1e9:.3f}B")
    for n, a in index["sources"].items():
        print(f"  {n:<12} train={a['train_tokens'] / 1e6:>9.1f}M val={a['val_tokens'] / 1e6:>7.2f}M "
              f"epochs={a['epochs']} mix={index['realized_mix'][n]:.1%}")
    if only or limit_files:
        for r in results[:2]:
            print(json.dumps(inspect_tokens.remote(r["source"], r["file"]), indent=2, ensure_ascii=False))


# ------------------------------------------------------------- Stage 6: pretraining
gpu_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.7.0", "transformers==4.46.3", "numpy==2.1.3")
    .add_local_python_source("viki_slm_125m")
)
PILOT_DIR_CKPT = f"{config.CKPT_DIR}/pilot"
LOCAL_TOKENS = "/tmp/tokens"
SMOKE_PROMPTS = (
    "def fibonacci(n):\n",
    "SELECT customer_id, SUM(amount) AS total FROM orders GROUP BY",
    "The Company reported that net revenues increased",
    "import pandas as pd\ndf = pd.read_csv('sales.csv')\n",
)


def _stage_tokens() -> float:
    """Copy packed tokens from the Volume to local disk (random access is much faster)."""
    import shutil
    import time

    t0 = time.time()
    shutil.copytree(config.TOKENS_DIR, LOCAL_TOKENS, dirs_exist_ok=True)
    return time.time() - t0


def _smoke_generate(model, device: str) -> dict[str, str]:
    import torch
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(config.TOKENIZER_DIR)
    model = model.to(torch.bfloat16).eval()
    out = {}
    for prompt in SMOKE_PROMPTS:
        ids = tok(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
        with torch.no_grad():
            gen = model.generate(ids, max_new_tokens=60, do_sample=False,
                                 pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
        out[prompt] = tok.decode(gen[0][ids.shape[1]:].tolist())
    return out


@app.function(image=gpu_image, volumes=VOLUMES, gpu="H100", timeout=60 * 120, cpu=8.0, memory=65_536)
def pretrain_run(tokens_b: float = 0.3, warmup_tokens_m: float = 20.0, compile_model: bool = True,
                 run_name: str = "pilot", cap_usd: float = 5.0, micro_batch: int = 32) -> dict:
    """Single-GPU pretraining run (used for the pilot). Resumable via the checkpoint on the Volume."""
    import json
    import os
    import time

    import torch

    from viki_slm_125m.pretrain import trainlib as tl

    volume.reload()
    print(torch.cuda.get_device_name(0), "torch", torch.__version__)
    copy_s = _stage_tokens()
    print(f"staged tokens to local disk in {copy_s:.0f}s")
    with open(f"{LOCAL_TOKENS}/index.json", encoding="utf-8") as fh:
        index = json.load(fh)
    refs = tl.refs_from_index(index, f"{LOCAL_TOKENS}/train")
    reader = tl.WindowReader([r.path for r in refs], config.SEQ_LEN)
    sampler = tl.BatchSampler(refs, config.TRAIN.seed)
    val_sets = tl.load_val_sets(index, f"{LOCAL_TOKENS}/val", config.SEQ_LEN, per_source=32)

    seqs_per_step = config.TRAIN.global_batch_tokens // config.SEQ_LEN        # 256
    total_steps = int(tokens_b * 1e9 // config.TRAIN.global_batch_tokens)
    run = tl.RunConfig(
        total_steps=total_steps, micro_batch=micro_batch, accum_steps=seqs_per_step // micro_batch,
        lr=config.TRAIN.lr, min_lr=config.TRAIN.min_lr,
        warmup_steps=int(warmup_tokens_m * 1e6 // config.TRAIN.global_batch_tokens),
        weight_decay=config.TRAIN.weight_decay, grad_clip=config.TRAIN.grad_clip,
        betas=(config.TRAIN.beta1, config.TRAIN.beta2), ckpt_every=200, eval_every=100,
        log_every=10, seed=config.TRAIN.seed, device="cuda", compile=compile_model)
    ckpt_dir = f"{config.CKPT_DIR}/{run_name}"
    os.makedirs(ckpt_dir, exist_ok=True)
    budget = tl.BudgetGuard(config.PRETRAIN_GPU_USD_PER_HOUR, 1, cap_usd)
    t0 = time.time()
    out = tl.train(config.MODEL, run, sampler, reader, ckpt_path=f"{ckpt_dir}/ckpt.pt",
                   metrics_path=f"{ckpt_dir}/metrics.jsonl", val_sets=val_sets, budget=budget)
    elapsed = time.time() - t0
    volume.commit()

    hist = out["history"]
    steady = [h["tokens_per_sec"] for h in hist if h["step"] > run.warmup_steps + 20] or \
             [h["tokens_per_sec"] for h in hist[1:]]
    tps = sum(steady) / len(steady)
    evals = [h for h in hist if "val_loss" in h]
    gens = _smoke_generate(out["model"], "cuda")
    full_tokens = 8.0e9
    summary = {
        "steps": out["final_step"], "total_steps": total_steps, "stopped_by_budget": out["stopped_by_budget"],
        "first_loss": hist[0]["loss"], "final_loss": hist[-1]["loss"],
        "val_loss_first": evals[0]["val_loss"] if evals else None,
        "val_loss_last": evals[-1]["val_loss"] if evals else None,
        "tokens_per_sec_1gpu": round(tps), "elapsed_s": round(elapsed),
        "pilot_cost_usd": round(config.PRETRAIN_GPU_USD_PER_HOUR * elapsed / 3600, 2),
        "projection_8gpu_hours_for_8B": round(full_tokens / (tps * 8 * 0.9) / 3600, 2),
        "projection_8gpu_usd_for_8B": round(full_tokens / (tps * 8 * 0.9) / 3600
                                            * config.PRETRAIN_GPU_USD_PER_HOUR * 8, 1),
        "generations": gens,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


@app.local_entrypoint()
def pretrain_pilot(tokens_b: float = 0.3, compile_model: bool = True):
    import json

    summary = pretrain_run.remote(tokens_b=tokens_b, compile_model=compile_model)
    with open("reports/phase6_pilot_report.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    save_report.remote("phase6_pilot_report", summary)


# ---- 8-GPU DDP run (trial and full)
@app.function(image=gpu_image, volumes=VOLUMES, gpu="H100:8", timeout=60 * 60 * 4, cpu=32.0,
              memory=131_072)
def pretrain_ddp(total_steps: int, warmup_steps: int, run_name: str, cap_usd: float,
                 ckpt_every: int = 500, eval_every: int = 500, no_compile: bool = False) -> dict:
    """Stage tokens once on local disk, then run torchrun across all 8 GPUs of this node."""
    import json
    import os
    import subprocess
    import time

    from viki_slm_125m.data.pipeline import copy_tree_parallel

    volume.reload()
    t0 = time.time()
    n_files = copy_tree_parallel(config.TOKENS_DIR, LOCAL_TOKENS, workers=48)
    staging_s = time.time() - t0
    print(f"staged {n_files} files in {staging_s:.0f}s")
    ckpt_dir = f"{config.CKPT_DIR}/{run_name}"
    os.makedirs(ckpt_dir, exist_ok=True)
    cmd = ["torchrun", "--standalone", "--nproc_per_node=8", "-m", "viki_slm_125m.pretrain.train_ddp",
           "--tokens-dir", LOCAL_TOKENS, "--ckpt-dir", ckpt_dir,
           "--total-steps", str(total_steps), "--warmup-steps", str(warmup_steps),
           "--ckpt-every", str(ckpt_every), "--eval-every", str(eval_every),
           "--cap-usd", str(cap_usd)]
    if no_compile:
        cmd.append("--no-compile")
    env = {**os.environ, "OMP_NUM_THREADS": "4"}
    t1 = time.time()
    result = subprocess.run(cmd, env=env)
    wall = time.time() - t0
    volume.commit()
    summary: dict = {"returncode": result.returncode, "staging_s": round(staging_s),
                     "wall_s": round(wall), "train_phase_s": round(time.time() - t1),
                     "node_cost_usd": round(config.PRETRAIN_GPU_USD_PER_HOUR * 8 * wall / 3600, 2)}
    path = f"{ckpt_dir}/summary.json"
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            summary.update(json.load(fh))
    print(json.dumps(summary, indent=2))
    return summary


@app.local_entrypoint()
def pretrain_8gpu(total_steps: int = 200, warmup_steps: int = 20, run_name: str = "trial8",
                  cap_usd: float = 6.0, ckpt_every: int = 100, eval_every: int = 100,
                  no_compile: bool = False):
    """8-GPU run. Trial: defaults (200 steps, ~105M tokens). Full: pass --total-steps etc."""
    import json

    summary = pretrain_ddp.remote(total_steps, warmup_steps, run_name, cap_usd, ckpt_every,
                                  eval_every, no_compile)
    with open(f"reports/phase6_{run_name}_report.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    save_report.remote(f"phase6_{run_name}_report", summary)


# ---- generation smoke test on a saved checkpoint
SMOKE_PROMPTS_V2 = (
    "def moving_average(values, window):\n",
    "import pandas as pd\ndf = pd.read_csv('orders.csv')\n# total revenue per region\n",
    "-- schema: orders(order_id, customer_id, amount, order_date)\n-- top 5 customers by total spend\nSELECT",
    "CREATE TABLE suppliers (\n",
    "The economic order quantity (EOQ) is the order size that minimizes",
    "Safety stock is held to protect against",
    "Net interest margin is calculated as",
    "Under Basel III, banks must maintain a minimum",
)


@app.function(image=gpu_image, volumes=VOLUMES, gpu="L4", timeout=60 * 20)
def smoke_generate(run_name: str, ckpt_name: str = "ckpt.pt") -> dict:
    import torch
    from transformers import AutoTokenizer

    from viki_slm_125m.pretrain import trainlib as tl

    volume.reload()
    state = torch.load(f"{config.CKPT_DIR}/{run_name}/{ckpt_name}", map_location="cpu",
                       weights_only=True)
    model = tl.build_model(config.MODEL)
    model.load_state_dict(state["model"])
    model = model.to(torch.bfloat16).to("cuda").eval()
    tok = AutoTokenizer.from_pretrained(config.TOKENIZER_DIR)
    out: dict = {"step": state["step"], "greedy": {}, "sampled": {}}
    torch.manual_seed(0)
    for prompt in SMOKE_PROMPTS_V2:
        ids = tok(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"].to("cuda")
        for mode, kw in (("greedy", {"do_sample": False, "repetition_penalty": 1.1}),
                         ("sampled", {"do_sample": True, "temperature": 0.7, "top_p": 0.9})):
            with torch.no_grad():
                gen = model.generate(ids, max_new_tokens=80, pad_token_id=tok.pad_token_id,
                                     eos_token_id=tok.eos_token_id, **kw)
            out[mode][prompt] = tok.decode(gen[0][ids.shape[1]:].tolist())
    return out


@app.local_entrypoint()
def smoke(run_name: str = "base-e1", ckpt_name: str = "ckpt.pt"):
    import json

    res = smoke_generate.remote(run_name, ckpt_name)
    with open(f"reports/smoke_{run_name}.json", "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2, ensure_ascii=False)
    print("step", res["step"])


@app.local_entrypoint()
def stage1(n_docs: int = 2000, fetch_files: int = 10_000):
    """Run every Stage 1 measurement in parallel and save the report."""
    import json

    text_sources = [s.name for s in config.DATA_MIX if not s.needs_fetch]
    handles = {
        "text": [measure_text_source.spawn(n, n_docs) for n in text_sources],
        "stack_edu": [measure_stack_edu.spawn(c) for c in ("Python", "SQL")],
        "fetch": [swh_fetch_test.spawn(c, fetch_files) for c in ("Python", "SQL")],
    }
    report = {k: [h.get() for h in v] for k, v in handles.items()}
    with open("reports/phase1_measurements.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(json.dumps(report, indent=2))
