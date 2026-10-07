# AGENT BRIEF: Replicate the 125M SLM Data Pipeline (Phases 0 to 4)\n\nYou are an AI coding agent. Follow this brief top to bottom to reproduce, from nothing, the training data for a 125M-parameter legal/financial small language model: a cleaned, deduplicated, decontaminated, tokenized corpus of about 2.19 billion tokens, stored on a Modal Volume. Everything here runs on CPU and costs well under 1 US dollar. Pretraining on GPU (Phase 5) is NOT part of this brief.\n\nThis document is self-contained. It gives you the exact accounts to create, the four source files to write verbatim, and the exact commands to run with their expected output. Do not improvise the data mix or the parameters; they were chosen by measurement and are fixed below.\n\n* * *\n\n## 1\\. HOW TO WORK (rules)\n\n1.  Go phase by phase. Run one phase, show the result, then continue. Do not chain all phases into one silent run.\n    \n2.  There is no GPU and no meaningful spend in Phases 0 to 4. Still, print the cost with `modal billing report` when asked.\n    \n3.  `config.py` is the single source of truth. Every other file imports from it.\n    \n4.  Write the four files in section 5 EXACTLY as given. They already encode every fix and threshold. Do not re-derive them.\n    \n5.  If a run is slow or gets preempted, do not switch to a single big container. The design is deliberately fanned out one worker per shard. Keep it that way.\n    \n\n* * *\n\n## 2\\. THE DATA (read carefully; the split is NOT 70/20/10)\n\n### 2.1 The three datasets\n\nAll three are public (ungated), streamed from HuggingFace (never fully downloaded), and parquet-native. You read only a text field from each:\n\nKeyHuggingFace idconfigsplittext fieldwhat it iscase-law`HFforLegal/case-law`default`usdocument`US court opinions (scanned; some OCR noise)sec`PleIAs/SEC`default`traintext`SEC filings (10-K etc.), born-digitalfineweb-edu`HuggingFaceFW/fineweb-edusample-10BTtraintext`general educational web text (fluency filler)\n\n### 2.2 The split is NOT 70/20/10. Here is why, and what it actually is.\n\nThe original idea was 70 percent case-law, 20 percent SEC, 10 percent web, at about 10 billion tokens. That is IMPOSSIBLE with these datasets, and you must not use it. Reason: the two legal sources are small. Measured yields (run the `measure` step in Phase 0 to confirm):\n\n*   case-law: about 0.8 to 1.0 billion clean tokens total (only ~282,000 documents).\n    \n*   sec: about 1.1 to 1.2 billion clean tokens total (only ~48,500 documents).\n    \n*   fineweb-edu: effectively unlimited (the sample-10BT slice holds ~11 billion).\n    \n\nSo the legal sources together cap at about 2 billion tokens. You cannot make case-law 70 percent of a 10 billion token corpus; it does not contain that much.\n\nThe actual strategy used here is \"legal-first\" (call it Choice A):\n\n*   Take ALL of case-law (budget cap 1.0B tokens; yields ~1.0B).\n    \n*   Take ALL of SEC (budget cap 1.3B tokens; yields ~1.1B).\n    \n*   Add a small web slice: fineweb-edu capped at 0.5B tokens.\n    \n\nThese caps live in `config.py` as `token_budget` per source. The streaming stops each source once its budget of clean tokens is reached (measured with a chars/4 proxy, since the real tokenizer does not exist until Phase 3).\n\nThe realized proportions after tokenization (Phase 4, real tokenizer counts):\n\n*   case-law: ~863M tokens (about 39 percent)\n    \n*   sec: ~861M tokens (about 39 percent)\n    \n*   fineweb-edu: ~465M tokens (about 21 percent)\n    \n\nSo the actual mix is roughly 40 / 40 / 20 (legal / legal / web), about 78 percent legal, NOT 70 / 20 / 10. Total: about 2.19 billion training tokens.\n\nYou get more \"tokens seen\" during pretraining by running multiple epochs over this fixed corpus (Phase 5), not by collecting more unique tokens.\n\n* * *\n\n## 3\\. PREREQUISITES: accounts and tokens\n\n### 3.1 Create a Modal account and authenticate the CLI\n\n1.  Sign up at https://modal.com (free tier includes monthly credits; this whole pipeline costs well under 1 dollar).\n    \n2.  Install and authenticate:\n    \n    ```bash\n    pip install modal\n    modal token new\n    ```\n    \n    `modal token new` opens a browser to authorize and writes `~/.modal.toml`. To do it non-interactively instead, create an API token in the dashboard (Settings, then API Tokens) and run:\n    \n    ```bash\n    modal token set --token-id ak-XXXXXXXX --token-secret as-XXXXXXXX\n    ```\n    \n3.  Verify:\n    \n    ```bash\n    modal profile current\n    ```\n    \n\n### 3.2 Create a HuggingFace token\n\n*   Only needed to push the finished MODEL later (Phase 6). All three datasets are ungated, so reading data needs no token.\n    \n*   Go to https://huggingface.co/settings/tokens, create a token with the WRITE role.\n    \n\n### 3.3 Put credentials in a git-ignored file\n\nCreate `.env.local` in the working directory (never commit it):\n\n```bash\nMODAL_TOKEN_ID=ak-XXXXXXXX\nMODAL_TOKEN_SECRET=as-XXXXXXXX\nHUGGINGFACE_TOKEN=hf_XXXXXXXX\n```\n\nAdd `.env.local` to `.gitignore`. Load it before every run:\n\n```bash\nsource .env.local && export MODAL_TOKEN_ID MODAL_TOKEN_SECRET\n```\n\n### 3.4 Create the persistent Volume\n\nOne Volume holds every durable artifact, mounted at `/data`:\n\n```bash\nmodal volume create slm-125m\n```\n\n(The app also creates it on first use, so this is optional but explicit.)\n\n* * *\n\n## 4\\. FILE LAYOUT\n\nCreate these four files in the working directory:\n\n*   `config.py` single source of truth (model, data mix, budgets, paths, thresholds)\n    \n*   `cleaning.py` pure deterministic 6-step cleaning chain\n    \n*   `dedup.py` pure helpers for dedup and decontamination\n    \n*   `modal_app.py` the Modal app: image, Volume, and one function per phase\n    \n\nOn-Volume layout that the runs produce:\n\n```\n/data/clean/<source>/shard-XX.txt      Phase 1: cleaned text, one doc per line\n/data/corpus/<source>/shard-XX.txt     Phase 2: deduped + decontaminated corpus\n/data/tokenizer/                       Phase 3: the 16K byte-level BPE tokenizer\n/data/tokens/train/*.bin               Phase 4: 99 percent packed uint16 windows\n/data/tokens/val/*.bin                 Phase 4: 1 percent packed uint16 windows\n/data/tokens/index.json                Phase 4: counts + dtype + seq_len\n```\n\nSanity-check config locally after writing it (no Modal needed):\n\n```bash\npython3 config.py\n# Expected first line: slm-125m\n# model: 125,847,552 params (~125.8M) | vocab 16384 | 12L/768d/12h kv=12\n```\n\n* * *\n\n## 5\\. THE FOUR SOURCE FILES (write each verbatim)\n\n### 5.1 config.py\n\n```python\n\"\"\"Single source of truth for the from-scratch 125M SLM build.\"\"\"\n\nfrom __future__ import annotations\n\nfrom dataclasses import dataclass\nfrom typing import Mapping\n\n# Project identity / paths (all paths are relative to the Modal Volume mount).\nPROJECT = \"slm-125m\"\nHF_REPO = \"your-user/slm-125m-base\"  # set to your own HF namespace for Phase 6\n\nVOLUME_NAME = \"slm-125m\"\nDATA_ROOT = \"/data\"\nCLEAN_DIR = f\"{DATA_ROOT}/clean\"          # Phase 1\nCORPUS_DIR = f\"{DATA_ROOT}/corpus\"        # Phase 2\nTOKENIZER_DIR = f\"{DATA_ROOT}/tokenizer\"  # Phase 3\nTOKENS_DIR = f\"{DATA_ROOT}/tokens\"        # Phase 4\nTRAIN_TOKENS_DIR = f\"{TOKENS_DIR}/train\"\nVAL_TOKENS_DIR = f\"{TOKENS_DIR}/val\"\nCKPT_DIR = f\"{DATA_ROOT}/checkpoints\"     # Phase 5 (not in this brief)\nBASE_CKPT_DIR = f\"{CKPT_DIR}/base\"\nRESUME_CKPT_PATH = f\"{CKPT_DIR}/ckpt.pt\"\nMETRICS_PATH = f\"{CKPT_DIR}/metrics.jsonl\"\n\nHF_SECRET_NAME = \"huggingface-token\"\n\n\n@dataclass(frozen=True)\nclass ModelConfig:\n    \"\"\"Maps 1:1 to transformers.LlamaConfig. ~125.8M params with tied embeddings.\"\"\"\n\n    vocab_size: int = 16_384\n    hidden_size: int = 768\n    intermediate_size: int = 3_072        # SwiGLU inner\n    num_hidden_layers: int = 12\n    num_attention_heads: int = 12         # head dim 64\n    num_key_value_heads: int = 12         # == heads -> MHA\n    max_position_embeddings: int = 1_024  # context length\n    rope_theta: float = 10_000.0\n    rms_norm_eps: float = 1e-5\n    hidden_act: str = \"silu\"              # SwiGLU\n    tie_word_embeddings: bool = True\n    attention_bias: bool = False\n\n    def to_llama_kwargs(self) -> dict:\n        return {\n            \"vocab_size\": self.vocab_size,\n            \"hidden_size\": self.hidden_size,\n            \"intermediate_size\": self.intermediate_size,\n            \"num_hidden_layers\": self.num_hidden_layers,\n            \"num_attention_heads\": self.num_attention_heads,\n            \"num_key_value_heads\": self.num_key_value_heads,\n            \"max_position_embeddings\": self.max_position_embeddings,\n            \"rope_theta\": self.rope_theta,\n            \"rms_norm_eps\": self.rms_norm_eps,\n            \"hidden_act\": self.hidden_act,\n            \"tie_word_embeddings\": self.tie_word_embeddings,\n            \"attention_bias\": self.attention_bias,\n        }\n\n    def approx_params(self) -> int:\n        e = self.vocab_size * self.hidden_size\n        h, i = self.hidden_size, self.intermediate_size\n        kv = self.num_key_value_heads * (h // self.num_attention_heads)\n        attn = h * h + 2 * (h * kv) + h * h\n        mlp = 3 * h * i\n        per_layer = attn + mlp + 2 * h\n        return e + self.num_hidden_layers * per_layer\n\n\nMODEL = ModelConfig()\n\nSPECIAL_TOKENS: Mapping[str, str] = {\n    \"bos_token\": \"<|bos|>\",\n    \"eos_token\": \"<|eos|>\",\n    \"pad_token\": \"<|pad|>\",\n    \"unk_token\": \"<|unk|>\",\n}\nEXTRA_CHAT_TOKENS: tuple[str, ...] = (\"<|user|>\", \"<|assistant|>\", \"<|system|>\")\n\n\n@dataclass(frozen=True)\nclass Source:\n    name: str\n    hf_id: str\n    token_budget: int      # stop streaming this source at ~this many clean tokens\n    text_field: str\n    split: str = \"train\"\n    config_name: str | None = None\n    strict_ocr: bool = False\n\n\n# Legal-first mix (Choice A). Legal sources cap ~2B tokens, so take all of them\n# and cap web at 0.5B. NOT 70/20/10. Realized ~40/40/20 (case-law/sec/web).\nDATA_MIX: tuple[Source, ...] = (\n    Source(\"case-law\", \"HFforLegal/case-law\", 1_000_000_000, \"document\",\n           split=\"us\", strict_ocr=True),\n    Source(\"sec\", \"PleIAs/SEC\", 1_300_000_000, \"text\", split=\"train\"),\n    Source(\"fineweb-edu\", \"HuggingFaceFW/fineweb-edu\", 500_000_000, \"text\",\n           split=\"train\", config_name=\"sample-10BT\"),\n)\n\nTARGET_TOKENS: int = 2_500_000_000\nCHARS_PER_TOKEN: float = 4.0\n\n# Held OUT of training; Phase 2 strips docs resembling these.\nEVAL_HOLDOUT: tuple[str, ...] = (\"coastalcph/lex_glue\", \"casehold/casehold\")\n\n\n@dataclass(frozen=True)\nclass CleanConfig:\n    min_line_chars: int = 40\n    max_nonalnum_ratio: float = 0.30\n    min_doc_chars: int = 600\n    repetition_top_k: int = 10\n    max_repetition_ratio: float = 0.50\n    ngram_n: int = 4\n    lang_sample_chars: int = 5_000\n    nonword_ratio_max: float = 0.20     # OCR gate: drop if >20% words non-dictionary\n    ocr_min_tokens: int = 50\n    dict_path: str = \"/usr/share/dict/words\"\n\n\nCLEAN = CleanConfig()\n\nSEQ_LEN: int = 1_024\nVAL_EVERY_N_WINDOWS: int = 100          # every 100th window -> val (99/1 split)\nTOKENS_DTYPE: str = \"uint16\"\n\n\n@dataclass(frozen=True)\nclass TrainConfig:\n    seq_len: int = SEQ_LEN\n    micro_batch_size: int = 32\n    global_batch_tokens: int = 524_288\n    lr: float = 6e-4\n    min_lr: float = 6e-5\n    warmup_tokens: int = 200_000_000\n    weight_decay: float = 0.1\n    grad_clip: float = 1.0\n    beta1: float = 0.9\n    beta2: float = 0.95\n    ckpt_every_steps: int = 500\n    log_every_steps: int = 20\n    eval_every_steps: int = 1_000\n    seed: int = 1337\n\n\nTRAIN = TrainConfig()\n\nPRETRAIN_GPU = \"H100\"\nPRETRAIN_GPU_COUNT = 8\nBUDGET_CAP_USD = 40.0\n\nSTAGES: tuple[str, ...] = (\n    \"setup\", \"clean\", \"dedup\", \"tokenizer\", \"tokenize\", \"pretrain\", \"deploy\",\n)\n\n\nif __name__ == \"__main__\":\n    p = MODEL.approx_params()\n    print(f\"{PROJECT}\")\n    print(f\"model: {p:,} params (~{p/1e6:.1f}M) | vocab {MODEL.vocab_size} | \"\n          f\"{MODEL.num_hidden_layers}L/{MODEL.hidden_size}d/\"\n          f\"{MODEL.num_attention_heads}h kv={MODEL.num_key_value_heads}\")\n    print(f\"target tokens: {TARGET_TOKENS/1e9:.1f}B (~{TARGET_TOKENS/p:.0f} tok/param)\")\n    print(f\"stages: {' -> '.join(STAGES)}\")\n```\n\n### 5.2 cleaning.py\n\n```python\n\"\"\"The fixed, rule-based, deterministic cleaning pipeline (pure functions).\"\"\"\n\nfrom __future__ import annotations\n\nimport re\nfrom collections import Counter\nfrom dataclasses import dataclass\n\nfrom config import CLEAN\n\n_BOILERPLATE_PATTERNS: tuple[re.Pattern[str], ...] = tuple(\n    re.compile(p, re.IGNORECASE)\n    for p in (\n        r\"^\\s*form\\s+10[-\\s]?[kq]\\b.*$\",\n        r\"^\\s*page\\s+\\d+\\s+of\\s+\\d+\\s*$\",\n        r\"^\\s*table\\s+of\\s+contents\\s*$\",\n        r\"^\\s*/s/\\s*.*$\",\n        r\"^\\s*all\\s+rights\\s+reserved.*$\",\n        r\"^\\s*united\\s+states\\s+securities\\s+and\\s+exchange\\s+commission\\s*$\",\n        r\"^\\s*securities\\s+and\\s+exchange\\s+commission\\s*$\",\n        r\"^\\s*washington,?\\s+d\\.?\\s?c\\.?\\s+\\d{5}\\s*$\",\n        r\"^\\s*\\[?\\s*x\\s*\\]?\\s*$\",\n    )\n)\n\n_WHITESPACE = re.compile(r\"\\s+\")\n_WORD = re.compile(r\"[A-Za-z]+\")\n_ALNUM = re.compile(r\"[A-Za-z0-9]\")\n\n\n@dataclass(frozen=True)\nclass CleanResult:\n    kept: bool\n    text: str\n    reason: str\n    raw_chars: int\n    clean_chars: int\n\n\ndef _nonalnum_ratio(line: str) -> float:\n    if not line:\n        return 1.0\n    alnum = sum(1 for c in line if _ALNUM.match(c))\n    return 1.0 - alnum / len(line)\n\n\ndef filter_lines(text: str) -> str:\n    out: list[str] = []\n    for raw in text.splitlines():\n        line = _WHITESPACE.sub(\" \", raw).strip()\n        if len(line) < CLEAN.min_line_chars:\n            continue\n        if _nonalnum_ratio(line) > CLEAN.max_nonalnum_ratio:\n            continue\n        out.append(line)\n    return \"\\n\".join(out)\n\n\ndef strip_boilerplate(text: str) -> str:\n    return \"\\n\".join(\n        line\n        for line in text.splitlines()\n        if not any(p.match(line) for p in _BOILERPLATE_PATTERNS)\n    )\n\n\ndef is_repetitive(text: str) -> bool:\n    words = text.split()\n    n = CLEAN.ngram_n\n    if len(words) < n * 2:\n        return False\n    grams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]\n    if not grams:\n        return False\n    counts = Counter(grams)\n    top = sum(c for _, c in counts.most_common(CLEAN.repetition_top_k))\n    return top / len(grams) > CLEAN.max_repetition_ratio\n\n\ndef _ascii_ratio(text: str) -> float:\n    if not text:\n        return 0.0\n    return sum(1 for c in text if ord(c) < 128) / len(text)\n\n\ndef is_english(text: str) -> bool:\n    \"\"\"ASCII-ratio first; langdetect only on the ambiguous 90-99% band.\"\"\"\n    sample = text[: CLEAN.lang_sample_chars]\n    ratio = _ascii_ratio(sample)\n    if ratio >= 0.99:\n        return True\n    if ratio < 0.90:\n        return False\n    try:\n        from langdetect import DetectorFactory, detect\n\n        DetectorFactory.seed = 0\n        return detect(sample) == \"en\"\n    except Exception:\n        return ratio > 0.95\n\n\n_OCR_TOKEN = re.compile(r\"[A-Za-z]{3,}\")\n_ENGLISH_WORDS: frozenset[str] | None = None\n\n\ndef _english_words() -> frozenset[str]:\n    global _ENGLISH_WORDS\n    if _ENGLISH_WORDS is None:\n        try:\n            with open(CLEAN.dict_path, encoding=\"utf-8\", errors=\"ignore\") as fh:\n                _ENGLISH_WORDS = frozenset(\n                    w.strip().lower() for w in fh if w.strip().isalpha()\n                )\n        except OSError:\n            _ENGLISH_WORDS = frozenset()\n    return _ENGLISH_WORDS\n\n\ndef nonword_ratio(text: str) -> float:\n    words = _english_words()\n    if not words:\n        return 0.0\n    toks = [t.lower() for t in _OCR_TOKEN.findall(text)]\n    if len(toks) < CLEAN.ocr_min_tokens:\n        return 0.0\n    nonword = sum(1 for t in toks if t not in words)\n    return nonword / len(toks)\n\n\ndef is_ocr_garble(text: str) -> bool:\n    return nonword_ratio(text) > CLEAN.nonword_ratio_max\n\n\ndef clean_document(text: str, *, strict_ocr: bool = False) -> CleanResult:\n    \"\"\"Run one document through the full deterministic chain.\"\"\"\n    raw_chars = len(text)\n    step1 = filter_lines(text)\n    step2 = strip_boilerplate(step1)\n    if len(step2) < CLEAN.min_doc_chars:\n        return CleanResult(False, \"\", \"too_short\", raw_chars, len(step2))\n    if is_repetitive(step2):\n        return CleanResult(False, \"\", \"repetitive\", raw_chars, len(step2))\n    if not is_english(step2):\n        return CleanResult(False, \"\", \"non_english\", raw_chars, len(step2))\n    if strict_ocr and is_ocr_garble(step2):\n        return CleanResult(False, \"\", \"ocr\", raw_chars, len(step2))\n    return CleanResult(True, step2, \"kept\", raw_chars, len(step2))\n```\n\n### 5.3 dedup.py\n\n```python\n\"\"\"Pure helpers for Phase 2 (dedup + contamination strip).\"\"\"\n\nfrom __future__ import annotations\n\nimport hashlib\nimport re\n\n_WS = re.compile(r\"\\s+\")\n_WORD = re.compile(r\"[a-z0-9]+\")\n\n\ndef normalize(text: str) -> str:\n    return _WS.sub(\" \", text.lower()).strip()\n\n\ndef words(text: str) -> list[str]:\n    return _WORD.findall(normalize(text))\n\n\ndef exact_hash(text: str) -> str:\n    return hashlib.blake2b(normalize(text).encode(\"utf-8\"), digest_size=16).hexdigest()\n\n\ndef word_ngrams(tokens: list[str], n: int) -> set[int]:\n    \"\"\"Fast native hash of word n-grams (contam set and doc grams share a process).\"\"\"\n    if len(tokens) < n:\n        return set()\n    return {hash(tuple(tokens[i : i + n])) for i in range(len(tokens) - n + 1)}\n\n\ndef shingles(tokens: list[str], k: int) -> set[bytes]:\n    if len(tokens) < k:\n        return {\" \".join(tokens).encode(\"utf-8\")} if tokens else set()\n    return {\" \".join(tokens[i : i + k]).encode(\"utf-8\") for i in range(len(tokens) - k + 1)}\n```\n\n### 5.4 modal\\_app.py\n\n```python\n\"\"\"Modal App for the from-scratch 125M SLM build (Phases 0 to 4).\"\"\"\n\nfrom __future__ import annotations\n\nimport modal\n\nimport config\n\napp = modal.App(config.PROJECT)\n\n# CPU base. All pip/apt build steps MUST come before add_local_* (Modal rule).\n_cpu_base = (\n    modal.Image.debian_slim(python_version=\"3.12\")\n    .apt_install(\"wamerican\")  # /usr/share/dict/words for the OCR gate\n    .pip_install(\n        \"datasets==3.6.0\",\n        \"huggingface_hub==0.34.4\",\n        \"langdetect==1.0.9\",\n        \"pyarrow==17.0.0\",\n        \"datasketch==1.6.5\",\n    )\n)\ncpu_image = _cpu_base.add_local_python_source(\"config\", \"cleaning\", \"dedup\")\n\nvolume = modal.Volume.from_name(config.VOLUME_NAME, create_if_missing=True)\nVOLUMES = {config.DATA_ROOT: volume}\n\n\ndef _stream_source(source: \"config.Source\", n: int):\n    from datasets import load_dataset\n\n    ds = load_dataset(source.hf_id, source.config_name, split=source.split, streaming=True)\n    for i, record in enumerate(ds):\n        if i >= n:\n            break\n        yield record\n\n\n@app.function(image=cpu_image, volumes=VOLUMES, timeout=60 * 15)\ndef smoke_test(n_per_source: int = 10) -> dict:\n    from cleaning import clean_document\n\n    summary: dict[str, dict] = {}\n    for source in config.DATA_MIX:\n        print(\"\\n\" + \"=\" * 78)\n        print(f\"SOURCE: {source.name}  ({source.hf_id}, split={source.split}, \"\n              f\"field='{source.text_field}')\")\n        print(\"=\" * 78)\n        kept = 0\n        reasons: dict[str, int] = {}\n        for i, record in enumerate(_stream_source(source, n_per_source)):\n            text = record.get(source.text_field) or \"\"\n            if not isinstance(text, str):\n                text = str(text)\n            result = clean_document(text)\n            reasons[result.reason] = reasons.get(result.reason, 0) + 1\n            kept += int(result.kept)\n            excerpt = (result.text[:240] if result.kept else text[:160]).replace(\"\\n\", \" / \")\n            print(f\"\\n[{source.name} #{i}] raw={result.raw_chars:>7} clean={result.clean_chars:>7} \"\n                  f\"-> {result.reason.upper()}\")\n            print(f\"    {excerpt}\")\n        summary[source.name] = {\"streamed\": n_per_source, \"kept\": kept, \"reasons\": reasons}\n    print(\"\\nSMOKE TEST SUMMARY\")\n    for name, s in summary.items():\n        print(f\"  {name:<12} kept {s['kept']}/{s['streamed']}  reasons={s['reasons']}\")\n    return summary\n\n\n@app.function(image=cpu_image, volumes=VOLUMES, timeout=60 * 20)\ndef measure_sources(n_per_source: int = 2000) -> dict:\n    from cleaning import clean_document\n\n    TOTAL_ROWS = {\"case-law\": 282_390, \"sec\": 48_543, \"fineweb-edu\": 9_670_000}\n    out: dict[str, dict] = {}\n    for source in config.DATA_MIX:\n        clean_chars = kept = 0\n        for record in _stream_source(source, n_per_source):\n            text = record.get(source.text_field) or \"\"\n            if not isinstance(text, str):\n                text = str(text)\n            r = clean_document(text)\n            if r.kept:\n                kept += 1\n                clean_chars += r.clean_chars\n        avg_clean = clean_chars / n_per_source if n_per_source else 0\n        total = TOTAL_ROWS[source.name]\n        est = total * avg_clean / config.CHARS_PER_TOKEN\n        out[source.name] = {\"est_clean_tokens\": int(est), \"keep_rate\": round(kept / n_per_source, 3)}\n        print(f\"{source.name:<12} keep={kept/n_per_source:.0%}  avg_clean={avg_clean:>7.0f} ch/doc  \"\n              f\"rows={total:>9,}  est_clean_tokens={est/1e9:.2f}B\")\n    print(f\"TOTAL est clean tokens: {sum(v['est_clean_tokens'] for v in out.values())/1e9:.2f}B\")\n    return out\n\n\n# ---- Phase 1: stream + clean, one worker per parquet shard ----\n_SOURCE_BY_NAME = {s.name: s for s in config.DATA_MIX}\n\n\n@app.function(image=cpu_image, volumes=VOLUMES, timeout=60 * 60)\ndef clean_shard(source_name: str, url: str, shard_index: int, token_cap: int) -> dict:\n    import os\n\n    from datasets import load_dataset\n\n    from cleaning import clean_document\n\n    source = _SOURCE_BY_NAME[source_name]\n    out_dir = f\"{config.CLEAN_DIR}/{source_name}\"\n    os.makedirs(out_dir, exist_ok=True)\n    out_path = f\"{out_dir}/shard-{shard_index:03d}.txt\"\n    ds = load_dataset(\"parquet\", data_files=url, split=\"train\", streaming=True)\n    streamed = kept = clean_chars = 0\n    reasons: dict[str, int] = {}\n    with open(out_path, \"w\", encoding=\"utf-8\") as fh:\n        for record in ds:\n            streamed += 1\n            text = record.get(source.text_field) or \"\"\n            if not isinstance(text, str):\n                text = str(text)\n            r = clean_document(text, strict_ocr=source.strict_ocr)\n            reasons[r.reason] = reasons.get(r.reason, 0) + 1\n            if r.kept:\n                fh.write(r.text.replace(\"\\n\", \" \").strip() + \"\\n\")\n                kept += 1\n                clean_chars += r.clean_chars\n                if clean_chars / config.CHARS_PER_TOKEN >= token_cap:\n                    break\n    volume.commit()\n    est_tokens = int(clean_chars / config.CHARS_PER_TOKEN)\n    print(f\"[{source_name} shard {shard_index:03d}] streamed={streamed} kept={kept} \"\n          f\"est_tokens={est_tokens/1e6:.1f}M reasons={reasons}\")\n    return {\"source\": source_name, \"shard\": shard_index, \"streamed\": streamed,\n            \"kept\": kept, \"est_tokens\": est_tokens, \"reasons\": reasons}\n\n\ndef _parquet_urls(hf_id: str, config_name: str, split: str) -> list[str]:\n    import json\n    import urllib.request\n\n    api = f\"https://datasets-server.huggingface.co/parquet?dataset={hf_id}\"\n    req = urllib.request.Request(api, headers={\"User-Agent\": \"slm-125m\"})\n    with urllib.request.urlopen(req, timeout=60) as resp:\n        data = json.load(resp)\n    return [f[\"url\"] for f in data.get(\"parquet_files\", [])\n            if f.get(\"config\") == config_name and f.get(\"split\") == split]\n\n\n@app.local_entrypoint()\ndef clean(fineweb_shards: int = 1, only: str = \"\"):\n    def cfg(s):\n        return s.config_name or \"default\"\n\n    sources = [s for s in config.DATA_MIX if not only or s.name == only]\n    work = []\n    for s in sources:\n        urls = _parquet_urls(s.hf_id, cfg(s), s.split)\n        if s.name == \"fineweb-edu\":\n            urls = urls[:fineweb_shards]\n        per_shard_cap = s.token_budget // max(1, len(urls))\n        for i, url in enumerate(urls):\n            work.append((s.name, url, i, per_shard_cap))\n        print(f\"{s.name:<12} {len(urls)} shard(s), per-shard cap ~{per_shard_cap/1e6:.0f}M tokens\")\n    print(f\"Launching {len(work)} clean workers...\")\n    results = list(clean_shard.starmap(work))\n    report: dict[str, dict] = {}\n    for r in results:\n        agg = report.setdefault(r[\"source\"], {\"streamed\": 0, \"kept\": 0, \"est_tokens\": 0, \"reasons\": {}})\n        agg[\"streamed\"] += r[\"streamed\"]\n        agg[\"kept\"] += r[\"kept\"]\n        agg[\"est_tokens\"] += r[\"est_tokens\"]\n        for k, v in r[\"reasons\"].items():\n            agg[\"reasons\"][k] = agg[\"reasons\"].get(k, 0) + v\n    print(\"PHASE 1 DROP REPORT\")\n    total = 0\n    for name, a in report.items():\n        total += a[\"est_tokens\"]\n        print(f\"  {name:<12} streamed={a['streamed']:>8} kept={a['kept']:>8} \"\n              f\"est_tokens={a['est_tokens']/1e9:.2f}B drops={a['reasons']}\")\n    print(f\"  TOTAL est_clean_tokens={total/1e9:.2f}B\")\n    save_report.remote(report)\n\n\nocr_image = cpu_image\n\n# ---- Phase 2: dedup + contamination strip ----\nSHINGLE_K = 5\nMINHASH_PERM = 32\nMINHASH_THRESHOLD = 0.8\nDECONTAM_NGRAM = 13\nSIG_DIR = f\"{config.DATA_ROOT}/tmp/minhash_sigs\"\nNEAR_DUPS_PATH = f\"{config.DATA_ROOT}/tmp/near_dups.json\"\nDECONTAM_SOURCES = {\"case-law\", \"sec\"}\nCLEAN_SHARDS = {\"case-law\": 10, \"sec\": 5, \"fineweb-edu\": 5}\n\n\ndef _build_contamination_ngrams() -> set:\n    from datasets import load_dataset\n\n    from dedup import word_ngrams, words\n\n    grams: set = set()\n    for hf_id, cfg_name in [(\"casehold/casehold\", None), (\"coastalcph/lex_glue\", \"case_hold\")]:\n        try:\n            urls = _parquet_urls(hf_id, cfg_name or \"default\", \"test\")\n            if not urls:\n                urls = _parquet_urls(hf_id, cfg_name or \"default\", \"train\")\n            ds = load_dataset(\"parquet\", data_files=urls, split=\"train\", streaming=True)\n            for rec in ds:\n                text = \" \".join(str(v) for v in rec.values() if isinstance(v, str))\n                grams |= word_ngrams(words(text), DECONTAM_NGRAM)\n        except Exception as e:\n            print(f\"  [decontam] could not load {hf_id}: {e}\")\n    print(f\"  [decontam] {len(grams):,} eval 13-grams loaded\")\n    return grams\n\n\n@app.function(image=cpu_image, volumes=VOLUMES, timeout=60 * 20, cpu=4.0, memory=4_096)\ndef minhash_shard(shard_basename: str) -> dict:\n    import os\n\n    import numpy as np\n    from datasketch import MinHash\n\n    from dedup import shingles, words\n\n    path = f\"{config.CLEAN_DIR}/case-law/{shard_basename}\"\n    sigs, idxs = [], []\n    with open(path, encoding=\"utf-8\") as fh:\n        for idx, line in enumerate(fh):\n            line = line.rstrip(\"\\n\")\n            if not line:\n                continue\n            m = MinHash(num_perm=MINHASH_PERM)\n            sh = list(shingles(words(line), SHINGLE_K))\n            if sh:\n                m.update_batch(sh)\n            sigs.append(m.hashvalues.astype(np.uint64))\n            idxs.append(idx)\n    os.makedirs(SIG_DIR, exist_ok=True)\n    np.savez(f\"{SIG_DIR}/{shard_basename}.npz\",\n             sigs=np.vstack(sigs), idxs=np.asarray(idxs, dtype=np.int64))\n    volume.commit()\n    print(f\"[minhash {shard_basename}] {len(idxs):,} docs\")\n    return {\"shard\": shard_basename, \"n\": len(idxs)}\n\n\n@app.function(image=cpu_image, volumes=VOLUMES, timeout=60 * 20, memory=8_192)\ndef build_near_dups() -> int:\n    import glob\n    import json\n    import os\n\n    import numpy as np\n    from datasketch import MinHash, MinHashLSH\n\n    near: dict[str, list[int]] = {}\n    lsh = MinHashLSH(threshold=MINHASH_THRESHOLD, num_perm=MINHASH_PERM)\n    for npz_path in sorted(glob.glob(f\"{SIG_DIR}/*.npz\")):\n        shard = os.path.basename(npz_path)[: -len(\".npz\")]\n        data = np.load(npz_path)\n        for row, idx in zip(data[\"sigs\"], data[\"idxs\"]):\n            m = MinHash(num_perm=MINHASH_PERM, hashvalues=row)\n            if lsh.query(m):\n                near.setdefault(shard, []).append(int(idx))\n            else:\n                lsh.insert(f\"{shard}:{int(idx)}\", m)\n    os.makedirs(os.path.dirname(NEAR_DUPS_PATH), exist_ok=True)\n    with open(NEAR_DUPS_PATH, \"w\", encoding=\"utf-8\") as fh:\n        json.dump(near, fh)\n    volume.commit()\n    total = sum(len(v) for v in near.values())\n    print(f\"[near-dups] {total:,} case-law near-duplicates\")\n    return total\n\n\n@app.function(image=cpu_image, volumes=VOLUMES, timeout=60 * 30, cpu=4.0, memory=8_192)\ndef write_corpus_shard(source_name: str, shard_basename: str) -> dict:\n    import json\n    import os\n\n    from dedup import exact_hash, word_ngrams, words\n\n    near: set[int] = set()\n    if source_name == \"case-law\":\n        with open(NEAR_DUPS_PATH, encoding=\"utf-8\") as fh:\n            near = set(json.load(fh).get(shard_basename, []))\n    contam = _build_contamination_ngrams() if source_name in DECONTAM_SOURCES else None\n    in_path = f\"{config.CLEAN_DIR}/{source_name}/{shard_basename}\"\n    out_dir = f\"{config.CORPUS_DIR}/{source_name}\"\n    os.makedirs(out_dir, exist_ok=True)\n    seen: set[str] = set()\n    kept = clean_chars = 0\n    reasons = {\"near_dup\": 0, \"exact_dup\": 0, \"contaminated\": 0, \"kept\": 0}\n    with open(in_path, encoding=\"utf-8\") as fin, \\\n            open(f\"{out_dir}/{shard_basename}\", \"w\", encoding=\"utf-8\") as fout:\n        for idx, line in enumerate(fin):\n            text = line.rstrip(\"\\n\")\n            if not text:\n                continue\n            if idx in near:\n                reasons[\"near_dup\"] += 1\n                continue\n            h = exact_hash(text)\n            if h in seen:\n                reasons[\"exact_dup\"] += 1\n                continue\n            if contam and (word_ngrams(words(text), DECONTAM_NGRAM) & contam):\n                reasons[\"contaminated\"] += 1\n                continue\n            seen.add(h)\n            fout.write(text + \"\\n\")\n            kept += 1\n            clean_chars += len(text)\n            reasons[\"kept\"] += 1\n    volume.commit()\n    print(f\"[corpus {source_name}/{shard_basename}] kept={kept} drops={reasons}\")\n    return {\"source\": source_name, \"shard\": shard_basename, \"kept\": kept,\n            \"est_tokens\": int(clean_chars / config.CHARS_PER_TOKEN), \"reasons\": reasons}\n\n\n@app.function(image=cpu_image, volumes=VOLUMES)\ndef write_phase2_report(results: list) -> dict:\n    import json\n\n    report: dict[str, dict] = {}\n    for r in results:\n        if not r:\n            continue\n        agg = report.setdefault(r[\"source\"], {\"kept\": 0, \"est_tokens\": 0,\n              \"reasons\": {\"near_dup\": 0, \"exact_dup\": 0, \"contaminated\": 0, \"kept\": 0}})\n        agg[\"kept\"] += r[\"kept\"]\n        agg[\"est_tokens\"] += r[\"est_tokens\"]\n        for k, v in r[\"reasons\"].items():\n            agg[\"reasons\"][k] = agg[\"reasons\"].get(k, 0) + v\n    total = sum(v[\"est_tokens\"] for v in report.values())\n    print(\"PHASE 2 REPORT\")\n    for name, a in report.items():\n        print(f\"  {name:<12} kept={a['kept']:>8} est_tokens={a['est_tokens']/1e9:.2f}B drops={a['reasons']}\")\n    print(f\"  TOTAL corpus est tokens: {total/1e9:.2f}B\")\n    with open(f\"{config.CORPUS_DIR}/phase2_report.json\", \"w\", encoding=\"utf-8\") as fh:\n        json.dump(report, fh, indent=2)\n    volume.commit()\n    return report\n\n\n@app.local_entrypoint()\ndef dedup(compute_sigs: bool = True):\n    if compute_sigs:\n        names = [f\"shard-{i:03d}.txt\" for i in range(CLEAN_SHARDS[\"case-law\"])]\n        print(f\"1/3 MinHash signatures for {len(names)} case-law shards...\")\n        list(minhash_shard.map(names))\n    print(\"2/3 building near-dup set (LSH)...\")\n    build_near_dups.remote()\n    work = [(src, f\"shard-{i:03d}.txt\") for src, n in CLEAN_SHARDS.items() for i in range(n)]\n    print(f\"3/3 writing final corpus ({len(work)} shards, parallel)...\")\n    results = list(write_corpus_shard.starmap(work))\n    write_phase2_report.remote(results)\n\n\n# ---- Phase 3: train the 16K byte-level BPE tokenizer ----\nml_image = _cpu_base.pip_install(\"transformers==4.46.3\").add_local_python_source(\n    \"config\", \"cleaning\", \"dedup\")\n\n\ndef _corpus_line_iter():\n    import glob\n\n    for path in sorted(glob.glob(f\"{config.CORPUS_DIR}/*/*.txt\")):\n        with open(path, encoding=\"utf-8\") as fh:\n            for line in fh:\n                line = line.rstrip(\"\\n\")\n                if line:\n                    yield line\n\n\n@app.function(image=ml_image, volumes=VOLUMES, timeout=60 * 40, cpu=8.0, memory=16_384)\ndef train_tokenizer() -> dict:\n    import os\n\n    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers\n    from transformers import PreTrainedTokenizerFast\n\n    specials = list(config.SPECIAL_TOKENS.values()) + list(config.EXTRA_CHAT_TOKENS)\n    tok = Tokenizer(models.BPE(unk_token=config.SPECIAL_TOKENS[\"unk_token\"]))\n    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)\n    tok.decoder = decoders.ByteLevel()\n    trainer = trainers.BpeTrainer(\n        vocab_size=config.MODEL.vocab_size, special_tokens=specials,\n        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=True)\n    print(\"training BPE...\")\n    tok.train_from_iterator(_corpus_line_iter(), trainer=trainer)\n    fast = PreTrainedTokenizerFast(\n        tokenizer_object=tok,\n        bos_token=config.SPECIAL_TOKENS[\"bos_token\"],\n        eos_token=config.SPECIAL_TOKENS[\"eos_token\"],\n        pad_token=config.SPECIAL_TOKENS[\"pad_token\"],\n        unk_token=config.SPECIAL_TOKENS[\"unk_token\"],\n        additional_special_tokens=list(config.EXTRA_CHAT_TOKENS))\n    os.makedirs(config.TOKENIZER_DIR, exist_ok=True)\n    fast.save_pretrained(config.TOKENIZER_DIR)\n    volume.commit()\n    for s in [\"The plaintiff shall bear the burden of proof by a preponderance of the evidence.\",\n              \"The Company's net revenues increased 12% year over year pursuant to the agreement.\"]:\n        ids = fast.encode(s)\n        print(f\"  '{s[:40]}...' -> {len(ids)} tokens | roundtrip={fast.decode(ids).strip() == s}\")\n    print(f\"vocab_size={fast.vocab_size}\")\n    return {\"vocab_size\": fast.vocab_size}\n\n\n@app.local_entrypoint()\ndef tokenizer():\n    train_tokenizer.remote()\n\n\n# ---- Phase 4: tokenize + pack into uint16 1024-token windows, split 99/1 ----\nTOKENIZE_SHARDS = {\"case-law\": 4, \"sec\": 6, \"fineweb-edu\": 4}\nENCODE_BATCH = 1_000\n\n\n@app.function(image=ml_image, volumes=VOLUMES, timeout=60 * 40, cpu=8.0, memory=16_384)\ndef tokenize_shard(source_name: str, shard_index: int, num_shards: int) -> dict:\n    import glob\n    import os\n\n    import numpy as np\n    from transformers import AutoTokenizer\n\n    tok = AutoTokenizer.from_pretrained(config.TOKENIZER_DIR)\n    eos_id = tok.convert_tokens_to_ids(config.SPECIAL_TOKENS[\"eos_token\"])\n    seq_len = config.SEQ_LEN\n    os.makedirs(config.TRAIN_TOKENS_DIR, exist_ok=True)\n    os.makedirs(config.VAL_TOKENS_DIR, exist_ok=True)\n    train_path = f\"{config.TRAIN_TOKENS_DIR}/{source_name}-{shard_index:03d}.bin\"\n    val_path = f\"{config.VAL_TOKENS_DIR}/{source_name}-{shard_index:03d}.bin\"\n    buf: list[int] = []\n    win_count = n_train = n_val = 0\n    corpus_files = sorted(glob.glob(f\"{config.CORPUS_DIR}/{source_name}/*.txt\"))\n\n    def _doc_iter():\n        for path in corpus_files:\n            with open(path, encoding=\"utf-8\") as fh:\n                for idx, line in enumerate(fh):\n                    if idx % num_shards == shard_index:\n                        line = line.rstrip(\"\\n\")\n                        if line:\n                            yield line\n\n    with open(train_path, \"wb\") as ftr, open(val_path, \"wb\") as fva:\n        batch: list[str] = []\n\n        def _flush():\n            nonlocal win_count, n_train, n_val\n            if not batch:\n                return\n            for ids in tok(batch, add_special_tokens=False)[\"input_ids\"]:\n                buf.extend(ids)\n                buf.append(eos_id)\n            while len(buf) >= seq_len:\n                window = np.asarray(buf[:seq_len], dtype=np.uint16)\n                del buf[:seq_len]\n                if win_count % config.VAL_EVERY_N_WINDOWS == 0:\n                    window.tofile(fva)\n                    n_val += 1\n                else:\n                    window.tofile(ftr)\n                    n_train += 1\n                win_count += 1\n\n        for doc in _doc_iter():\n            batch.append(doc)\n            if len(batch) >= ENCODE_BATCH:\n                _flush()\n                batch = []\n        _flush()\n    volume.commit()\n    print(f\"[{source_name} {shard_index:03d}] train_win={n_train} val_win={n_val} \"\n          f\"train_tok={n_train*seq_len/1e6:.1f}M\")\n    return {\"source\": source_name, \"shard\": shard_index, \"train_windows\": n_train,\n            \"val_windows\": n_val, \"train_tokens\": n_train * seq_len, \"val_tokens\": n_val * seq_len}\n\n\n@app.function(image=ml_image, volumes=VOLUMES)\ndef write_token_index(results: list) -> dict:\n    import json\n\n    shards = [r for r in results if r]\n    total = {\"seq_len\": config.SEQ_LEN, \"dtype\": config.TOKENS_DTYPE,\n             \"train_windows\": sum(r[\"train_windows\"] for r in shards),\n             \"val_windows\": sum(r[\"val_windows\"] for r in shards),\n             \"train_tokens\": sum(r[\"train_tokens\"] for r in shards),\n             \"val_tokens\": sum(r[\"val_tokens\"] for r in shards), \"shards\": shards}\n    with open(f\"{config.TOKENS_DIR}/index.json\", \"w\", encoding=\"utf-8\") as fh:\n        json.dump(total, fh, indent=2)\n    volume.commit()\n    print(f\"index: train={total['train_tokens']/1e9:.2f}B tok ({total['train_windows']} win), \"\n          f\"val={total['val_tokens']/1e6:.1f}M tok ({total['val_windows']} win)\")\n    return total\n\n\n@app.local_entrypoint()\ndef tokenize():\n    work = [(name, i, n) for name, n in TOKENIZE_SHARDS.items() for i in range(n)]\n    print(f\"Launching {len(work)} tokenize workers...\")\n    results = list(tokenize_shard.starmap(work))\n    write_token_index.remote(results)\n\n\n# ---- OCR-threshold analysis (optional; informs config.CLEAN.nonword_ratio_max) ----\n@app.function(image=ocr_image, timeout=60 * 15)\ndef ocr_sample(n_docs: int = 3000) -> dict:\n    import re\n\n    from cleaning import clean_document\n\n    with open(\"/usr/share/dict/words\", encoding=\"utf-8\", errors=\"ignore\") as fh:\n        vocab = {w.strip().lower() for w in fh if w.strip().isalpha()}\n    tokre = re.compile(r\"[A-Za-z]{3,}\")\n    source = _SOURCE_BY_NAME[\"case-law\"]\n    ratios: list[float] = []\n    for record in _stream_source(source, n_docs):\n        text = record.get(source.text_field) or \"\"\n        if not isinstance(text, str):\n            text = str(text)\n        r = clean_document(text)\n        if not r.kept:\n            continue\n        toks = [t.lower() for t in tokre.findall(r.text)]\n        if len(toks) < 50:\n            continue\n        ratios.append(sum(1 for t in toks if t not in vocab) / len(toks))\n    ratios.sort()\n    n = len(ratios)\n    for t in [0.10, 0.15, 0.20, 0.25, 0.30]:\n        d = sum(1 for x in ratios if x > t)\n        print(f\"  drop if non-word ratio >{int(t*100)}%: {d} docs ({d/n:.1%})\")\n    return {\"scored\": n}\n\n\n@app.local_entrypoint()\ndef ocr(n_docs: int = 3000):\n    ocr_sample.remote(n_docs)\n\n\n@app.function(image=cpu_image, volumes=VOLUMES)\ndef save_report(report: dict) -> None:\n    import json\n\n    with open(f\"{config.CLEAN_DIR}/phase1_report.json\", \"w\", encoding=\"utf-8\") as fh:\n        json.dump(report, fh, indent=2)\n    volume.commit()\n\n\n@app.local_entrypoint()\ndef main(n_per_source: int = 10):\n    smoke_test.remote(n_per_source)\n\n\n@app.local_entrypoint()\ndef measure(n_per_source: int = 2000):\n    measure_sources.remote(n_per_source)\n```\n\n* * *\n\n## 6\\. RUN THE PIPELINE (phase by phase)\n\nBefore every command:\n\n```bash\nsource .env.local && export MODAL_TOKEN_ID MODAL_TOKEN_SECRET\n```\n\n### Phase 0: smoke test + measure (cost about 0)\n\n```bash\nmodal run modal_app.py            # smoke test: 10 docs/source, clean, print\nmodal run modal_app.py::measure   # project true token yield per source\n```\n\nExpect the smoke test to keep about 9 or 10 of 10 docs per source. Expect `measure` to report roughly case-law ~0.8B, sec ~1.1B, fineweb ~11B available, which is why the mix is legal-first, not 70/20/10.\n\n### Phase 1: stream + clean (cost about 0; a few minutes)\n\n```bash\nmodal run modal_app.py::clean --fineweb-shards 5\n```\n\nFans out 16 workers (case-law 10 + sec 5 + fineweb 5), one per parquet shard. Writes `/data/clean/<source>/shard-XX.txt`. To redo one source only: `modal run modal_app.py::clean --only case-law`. Expect about 718,000 docs streamed, about 698,000 kept (~97 percent), roughly 2.68B proxy tokens. The OCR gate (case-law only) drops ~1,400 scanned-garbage docs.\n\nOptional OCR-threshold check (already set to 0.20 in config):\n\n```bash\nmodal run modal_app.py::ocr\n```\n\n### Phase 2: dedup + decontaminate (cost about 0; about 6 minutes)\n\n```bash\nmodal run modal_app.py::dedup\n```\n\nThree parallel stages: MinHash signatures per case-law shard, one fast LSH near-dup pass, then one writer per shard (near-dup + exact-dup + contamination). Writes `/data/corpus/<source>/shard-XX.txt`. Expect ~24,000 case-law docs removed as CaseHOLD-contaminated, ~1,600 near-dups, ~2,000 SEC exact-dups. Corpus ~670,000 docs, ~2.40B proxy tokens. To reuse signatures already computed: `--no-compute-sigs`.\n\n### Phase 3: train the tokenizer (cost about 0; about 4 minutes)\n\n```bash\nmodal run modal_app.py::tokenizer\n```\n\nTrains a fresh 16,384 byte-level BPE on the whole corpus, saves to `/data/tokenizer/`. Expect `vocab_size=16384` and two `roundtrip=True` lines.\n\n### Phase 4: tokenize + pack + split 99/1 (cost about 0; about 10 minutes)\n\n```bash\nmodal run modal_app.py::tokenize\n```\n\n14 workers encode, append `<|eos|>` after each doc, pack into 1024-token uint16 windows, route every 100th window to val. Writes `/data/tokens/train/*.bin`, `/data/tokens/val/*.bin`, `/data/tokens/index.json`. Expect: `index: train=2.19B tok (~2.14M win), val=22.1M tok (~21.6K win)`.\n\n### Verify\n\n```bash\nmodal volume ls slm-125m /tokens       # train/ val/ index.json\nmodal volume ls slm-125m /tokenizer    # tokenizer.json + configs\nmodal volume get slm-125m /tokens/index.json ./index.json\n```\n\n* * *\n\n## 7\\. EXPECTED FINAL RESULT AND COST\n\nFinal training data on the Volume:\n\n*   train: about 2.19 billion tokens (about 2,138,970 windows of 1024)\n    \n*   val: about 22.1 million tokens (about 21,614 windows), a clean 1.0 percent split\n    \n*   realized mix: case-law ~863M, sec ~861M, fineweb ~465M (about 40 / 40 / 20)\n    \n\nCost of Phases 0 to 4: well under 1 US dollar (this run billed about 0.18 USD), all CPU. Wall-clock: about 40 minutes of useful compute. Check spend:\n\n```bash\nmodal billing report --start 2026-07-08 --json \\\n  | python3 -c \"import sys,json; print(sum(float(r['cost']) for r in json.load(sys.stdin)))\"\n```\n\n* * *\n\n## 8\\. GOTCHAS (do not relearn these)\n\n1.  The data split is legal-first, NOT 70/20/10. The legal sources only hold ~2B tokens; see section 2.\n    \n2.  Modal image rule: all `pip_install` and `apt_install` steps must come BEFORE `add_local_python_source`, or the image build errors.\n    \n3.  `langdetect` is slow per document. `is_english` is ASCII-first and only calls `langdetect` on the ambiguous 90 to 99 percent ASCII band. Keep that ordering.\n    \n4.  The OCR gate needs the system wordlist. The base image installs the apt package `wamerican` which provides `/usr/share/dict/words`.\n    \n5.  Modal can preempt a long single container and restart it from zero. Keep every heavy step fanned out one worker per shard (Phases 1, 2, 4 already do this).\n    \n6.  Token counts in Phases 1 and 2 are a chars/4 PROXY. Only Phase 4 (the real tokenizer) gives true counts; they came in about 8 percent lower than the proxy.\n    \n7.  `casehold/casehold` parquet may not resolve; the LexGLUE `case_hold` config covers the same CaseHOLD benchmark for decontamination, so this is fine.\n    \n8.  To change how much data you keep, edit `token_budget` per source in `config.py`, then re-run Phase 1 and every phase after it in order.# AGENT BRIEF: Replicate the 125M SLM Data Pipeline (Phases 0 to 4)

You are an AI coding agent. Follow this brief top to bottom to reproduce, from nothing, the training data for a 125M-parameter legal/financial small language model: a cleaned, deduplicated, decontaminated, tokenized corpus of about 2.19 billion tokens, stored on a Modal Volume. Everything here runs on CPU and costs well under 1 US dollar. Pretraining on GPU (Phase 5) is NOT part of this brief.

This document is self-contained. It gives you the exact accounts to create, the four source files to write verbatim, and the exact commands to run with their expected output. Do not improvise the data mix or the parameters; they were chosen by measurement and are fixed below.

* * *

## 1\. HOW TO WORK (rules)

1.  Go phase by phase. Run one phase, show the result, then continue. Do not chain all phases into one silent run.
    
2.  There is no GPU and no meaningful spend in Phases 0 to 4. Still, print the cost with `modal billing report` when asked.
    
3.  `config.py` is the single source of truth. Every other file imports from it.
    
4.  Write the four files in section 5 EXACTLY as given. They already encode every fix and threshold. Do not re-derive them.
    
5.  If a run is slow or gets preempted, do not switch to a single big container. The design is deliberately fanned out one worker per shard. Keep it that way.
    

* * *

## 2\. THE DATA (read carefully; the split is NOT 70/20/10)

### 2.1 The three datasets

All three are public (ungated), streamed from HuggingFace (never fully downloaded), and parquet-native. You read only a text field from each:

KeyHuggingFace idconfigsplittext fieldwhat it iscase-law`HFforLegal/case-law`default`usdocument`US court opinions (scanned; some OCR noise)sec`PleIAs/SEC`default`traintext`SEC filings (10-K etc.), born-digitalfineweb-edu`HuggingFaceFW/fineweb-edusample-10BTtraintext`general educational web text (fluency filler)

### 2.2 The split is NOT 70/20/10. Here is why, and what it actually is.

The original idea was 70 percent case-law, 20 percent SEC, 10 percent web, at about 10 billion tokens. That is IMPOSSIBLE with these datasets, and you must not use it. Reason: the two legal sources are small. Measured yields (run the `measure` step in Phase 0 to confirm):

*   case-law: about 0.8 to 1.0 billion clean tokens total (only ~282,000 documents).
    
*   sec: about 1.1 to 1.2 billion clean tokens total (only ~48,500 documents).
    
*   fineweb-edu: effectively unlimited (the sample-10BT slice holds ~11 billion).
    

So the legal sources together cap at about 2 billion tokens. You cannot make case-law 70 percent of a 10 billion token corpus; it does not contain that much.

The actual strategy used here is "legal-first" (call it Choice A):

*   Take ALL of case-law (budget cap 1.0B tokens; yields ~1.0B).
    
*   Take ALL of SEC (budget cap 1.3B tokens; yields ~1.1B).
    
*   Add a small web slice: fineweb-edu capped at 0.5B tokens.
    

These caps live in `config.py` as `token_budget` per source. The streaming stops each source once its budget of clean tokens is reached (measured with a chars/4 proxy, since the real tokenizer does not exist until Phase 3).

The realized proportions after tokenization (Phase 4, real tokenizer counts):

*   case-law: ~863M tokens (about 39 percent)
    
*   sec: ~861M tokens (about 39 percent)
    
*   fineweb-edu: ~465M tokens (about 21 percent)
    

So the actual mix is roughly 40 / 40 / 20 (legal / legal / web), about 78 percent legal, NOT 70 / 20 / 10. Total: about 2.19 billion training tokens.

You get more "tokens seen" during pretraining by running multiple epochs over this fixed corpus (Phase 5), not by collecting more unique tokens.

* * *

## 3\. PREREQUISITES: accounts and tokens

### 3.1 Create a Modal account and authenticate the CLI

1.  Sign up at https://modal.com (free tier includes monthly credits; this whole pipeline costs well under 1 dollar).
    
2.  Install and authenticate:
    
    ```bash
    pip install modal
    modal token new
    ```
    
    `modal token new` opens a browser to authorize and writes `~/.modal.toml`. To do it non-interactively instead, create an API token in the dashboard (Settings, then API Tokens) and run:
    
    ```bash
    modal token set --token-id ak-XXXXXXXX --token-secret as-XXXXXXXX
    ```
    
3.  Verify:
    
    ```bash
    modal profile current
    ```
    

### 3.2 Create a HuggingFace token

*   Only needed to push the finished MODEL later (Phase 6). All three datasets are ungated, so reading data needs no token.
    
*   Go to https://huggingface.co/settings/tokens, create a token with the WRITE role.
    

### 3.3 Put credentials in a git-ignored file

Create `.env.local` in the working directory (never commit it):

```bash
MODAL_TOKEN_ID=ak-XXXXXXXX
MODAL_TOKEN_SECRET=as-XXXXXXXX
HUGGINGFACE_TOKEN=hf_XXXXXXXX
```

Add `.env.local` to `.gitignore`. Load it before every run:

```bash
source .env.local && export MODAL_TOKEN_ID MODAL_TOKEN_SECRET
```

### 3.4 Create the persistent Volume

One Volume holds every durable artifact, mounted at `/data`:

```bash
modal volume create slm-125m
```

(The app also creates it on first use, so this is optional but explicit.)

* * *

## 4\. FILE LAYOUT

Create these four files in the working directory:

*   `config.py` single source of truth (model, data mix, budgets, paths, thresholds)
    
*   `cleaning.py` pure deterministic 6-step cleaning chain
    
*   `dedup.py` pure helpers for dedup and decontamination
    
*   `modal_app.py` the Modal app: image, Volume, and one function per phase
    

On-Volume layout that the runs produce:

```
/data/clean/<source>/shard-XX.txt      Phase 1: cleaned text, one doc per line
/data/corpus/<source>/shard-XX.txt     Phase 2: deduped + decontaminated corpus
/data/tokenizer/                       Phase 3: the 16K byte-level BPE tokenizer
/data/tokens/train/*.bin               Phase 4: 99 percent packed uint16 windows
/data/tokens/val/*.bin                 Phase 4: 1 percent packed uint16 windows
/data/tokens/index.json                Phase 4: counts + dtype + seq_len
```

Sanity-check config locally after writing it (no Modal needed):

```bash
python3 config.py
# Expected first line: slm-125m
# model: 125,847,552 params (~125.8M) | vocab 16384 | 12L/768d/12h kv=12
```

* * *

## 5\. THE FOUR SOURCE FILES (write each verbatim)

### 5.1 config.py

```python
"""Single source of truth for the from-scratch 125M SLM build."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

# Project identity / paths (all paths are relative to the Modal Volume mount).
PROJECT = "slm-125m"
HF_REPO = "your-user/slm-125m-base"  # set to your own HF namespace for Phase 6

VOLUME_NAME = "slm-125m"
DATA_ROOT = "/data"
CLEAN_DIR = f"{DATA_ROOT}/clean"          # Phase 1
CORPUS_DIR = f"{DATA_ROOT}/corpus"        # Phase 2
TOKENIZER_DIR = f"{DATA_ROOT}/tokenizer"  # Phase 3
TOKENS_DIR = f"{DATA_ROOT}/tokens"        # Phase 4
TRAIN_TOKENS_DIR = f"{TOKENS_DIR}/train"
VAL_TOKENS_DIR = f"{TOKENS_DIR}/val"
CKPT_DIR = f"{DATA_ROOT}/checkpoints"     # Phase 5 (not in this brief)
BASE_CKPT_DIR = f"{CKPT_DIR}/base"
RESUME_CKPT_PATH = f"{CKPT_DIR}/ckpt.pt"
METRICS_PATH = f"{CKPT_DIR}/metrics.jsonl"

HF_SECRET_NAME = "huggingface-token"


@dataclass(frozen=True)
class ModelConfig:
    """Maps 1:1 to transformers.LlamaConfig. ~125.8M params with tied embeddings."""

    vocab_size: int = 16_384
    hidden_size: int = 768
    intermediate_size: int = 3_072        # SwiGLU inner
    num_hidden_layers: int = 12
    num_attention_heads: int = 12         # head dim 64
    num_key_value_heads: int = 12         # == heads -> MHA
    max_position_embeddings: int = 1_024  # context length
    rope_theta: float = 10_000.0
    rms_norm_eps: float = 1e-5
    hidden_act: str = "silu"              # SwiGLU
    tie_word_embeddings: bool = True
    attention_bias: bool = False

    def to_llama_kwargs(self) -> dict:
        return {
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "max_position_embeddings": self.max_position_embeddings,
            "rope_theta": self.rope_theta,
            "rms_norm_eps": self.rms_norm_eps,
            "hidden_act": self.hidden_act,
            "tie_word_embeddings": self.tie_word_embeddings,
            "attention_bias": self.attention_bias,
        }

    def approx_params(self) -> int:
        e = self.vocab_size * self.hidden_size
        h, i = self.hidden_size, self.intermediate_size
        kv = self.num_key_value_heads * (h // self.num_attention_heads)
        attn = h * h + 2 * (h * kv) + h * h
        mlp = 3 * h * i
        per_layer = attn + mlp + 2 * h
        return e + self.num_hidden_layers * per_layer


MODEL = ModelConfig()

SPECIAL_TOKENS: Mapping[str, str] = {
    "bos_token": "<|bos|>",
    "eos_token": "<|eos|>",
    "pad_token": "<|pad|>",
    "unk_token": "<|unk|>",
}
EXTRA_CHAT_TOKENS: tuple[str, ...] = ("<|user|>", "<|assistant|>", "<|system|>")


@dataclass(frozen=True)
class Source:
    name: str
    hf_id: str
    token_budget: int      # stop streaming this source at ~this many clean tokens
    text_field: str
    split: str = "train"
    config_name: str | None = None
    strict_ocr: bool = False


# Legal-first mix (Choice A). Legal sources cap ~2B tokens, so take all of them
# and cap web at 0.5B. NOT 70/20/10. Realized ~40/40/20 (case-law/sec/web).
DATA_MIX: tuple[Source, ...] = (
    Source("case-law", "HFforLegal/case-law", 1_000_000_000, "document",
           split="us", strict_ocr=True),
    Source("sec", "PleIAs/SEC", 1_300_000_000, "text", split="train"),
    Source("fineweb-edu", "HuggingFaceFW/fineweb-edu", 500_000_000, "text",
           split="train", config_name="sample-10BT"),
)

TARGET_TOKENS: int = 2_500_000_000
CHARS_PER_TOKEN: float = 4.0

# Held OUT of training; Phase 2 strips docs resembling these.
EVAL_HOLDOUT: tuple[str, ...] = ("coastalcph/lex_glue", "casehold/casehold")


@dataclass(frozen=True)
class CleanConfig:
    min_line_chars: int = 40
    max_nonalnum_ratio: float = 0.30
    min_doc_chars: int = 600
    repetition_top_k: int = 10
    max_repetition_ratio: float = 0.50
    ngram_n: int = 4
    lang_sample_chars: int = 5_000
    nonword_ratio_max: float = 0.20     # OCR gate: drop if >20% words non-dictionary
    ocr_min_tokens: int = 50
    dict_path: str = "/usr/share/dict/words"


CLEAN = CleanConfig()

SEQ_LEN: int = 1_024
VAL_EVERY_N_WINDOWS: int = 100          # every 100th window -> val (99/1 split)
TOKENS_DTYPE: str = "uint16"


@dataclass(frozen=True)
class TrainConfig:
    seq_len: int = SEQ_LEN
    micro_batch_size: int = 32
    global_batch_tokens: int = 524_288
    lr: float = 6e-4
    min_lr: float = 6e-5
    warmup_tokens: int = 200_000_000
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    beta1: float = 0.9
    beta2: float = 0.95
    ckpt_every_steps: int = 500
    log_every_steps: int = 20
    eval_every_steps: int = 1_000
    seed: int = 1337


TRAIN = TrainConfig()

PRETRAIN_GPU = "H100"
PRETRAIN_GPU_COUNT = 8
BUDGET_CAP_USD = 40.0

STAGES: tuple[str, ...] = (
    "setup", "clean", "dedup", "tokenizer", "tokenize", "pretrain", "deploy",
)


if __name__ == "__main__":
    p = MODEL.approx_params()
    print(f"{PROJECT}")
    print(f"model: {p:,} params (~{p/1e6:.1f}M) | vocab {MODEL.vocab_size} | "
          f"{MODEL.num_hidden_layers}L/{MODEL.hidden_size}d/"
          f"{MODEL.num_attention_heads}h kv={MODEL.num_key_value_heads}")
    print(f"target tokens: {TARGET_TOKENS/1e9:.1f}B (~{TARGET_TOKENS/p:.0f} tok/param)")
    print(f"stages: {' -> '.join(STAGES)}")
```

### 5.2 cleaning.py

```python
"""The fixed, rule-based, deterministic cleaning pipeline (pure functions)."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

from config import CLEAN

_BOILERPLATE_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"^\s*form\s+10[-\s]?[kq]\b.*$",
        r"^\s*page\s+\d+\s+of\s+\d+\s*$",
        r"^\s*table\s+of\s+contents\s*$",
        r"^\s*/s/\s*.*$",
        r"^\s*all\s+rights\s+reserved.*$",
        r"^\s*united\s+states\s+securities\s+and\s+exchange\s+commission\s*$",
        r"^\s*securities\s+and\s+exchange\s+commission\s*$",
        r"^\s*washington,?\s+d\.?\s?c\.?\s+\d{5}\s*$",
        r"^\s*\[?\s*x\s*\]?\s*$",
    )
)

_WHITESPACE = re.compile(r"\s+")
_WORD = re.compile(r"[A-Za-z]+")
_ALNUM = re.compile(r"[A-Za-z0-9]")


@dataclass(frozen=True)
class CleanResult:
    kept: bool
    text: str
    reason: str
    raw_chars: int
    clean_chars: int


def _nonalnum_ratio(line: str) -> float:
    if not line:
        return 1.0
    alnum = sum(1 for c in line if _ALNUM.match(c))
    return 1.0 - alnum / len(line)


def filter_lines(text: str) -> str:
    out: list[str] = []
    for raw in text.splitlines():
        line = _WHITESPACE.sub(" ", raw).strip()
        if len(line) < CLEAN.min_line_chars:
            continue
        if _nonalnum_ratio(line) > CLEAN.max_nonalnum_ratio:
            continue
        out.append(line)
    return "\n".join(out)


def strip_boilerplate(text: str) -> str:
    return "\n".join(
        line
        for line in text.splitlines()
        if not any(p.match(line) for p in _BOILERPLATE_PATTERNS)
    )


def is_repetitive(text: str) -> bool:
    words = text.split()
    n = CLEAN.ngram_n
    if len(words) < n * 2:
        return False
    grams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
    if not grams:
        return False
    counts = Counter(grams)
    top = sum(c for _, c in counts.most_common(CLEAN.repetition_top_k))
    return top / len(grams) > CLEAN.max_repetition_ratio


def _ascii_ratio(text: str) -> float:
    if not text:
        return 0.0
    return sum(1 for c in text if ord(c) < 128) / len(text)


def is_english(text: str) -> bool:
    """ASCII-ratio first; langdetect only on the ambiguous 90-99% band."""
    sample = text[: CLEAN.lang_sample_chars]
    ratio = _ascii_ratio(sample)
    if ratio >= 0.99:
        return True
    if ratio < 0.90:
        return False
    try:
        from langdetect import DetectorFactory, detect

        DetectorFactory.seed = 0
        return detect(sample) == "en"
    except Exception:
        return ratio > 0.95


_OCR_TOKEN = re.compile(r"[A-Za-z]{3,}")
_ENGLISH_WORDS: frozenset[str] | None = None


def _english_words() -> frozenset[str]:
    global _ENGLISH_WORDS
    if _ENGLISH_WORDS is None:
        try:
            with open(CLEAN.dict_path, encoding="utf-8", errors="ignore") as fh:
                _ENGLISH_WORDS = frozenset(
                    w.strip().lower() for w in fh if w.strip().isalpha()
                )
        except OSError:
            _ENGLISH_WORDS = frozenset()
    return _ENGLISH_WORDS


def nonword_ratio(text: str) -> float:
    words = _english_words()
    if not words:
        return 0.0
    toks = [t.lower() for t in _OCR_TOKEN.findall(text)]
    if len(toks) < CLEAN.ocr_min_tokens:
        return 0.0
    nonword = sum(1 for t in toks if t not in words)
    return nonword / len(toks)


def is_ocr_garble(text: str) -> bool:
    return nonword_ratio(text) > CLEAN.nonword_ratio_max


def clean_document(text: str, *, strict_ocr: bool = False) -> CleanResult:
    """Run one document through the full deterministic chain."""
    raw_chars = len(text)
    step1 = filter_lines(text)
    step2 = strip_boilerplate(step1)
    if len(step2) < CLEAN.min_doc_chars:
        return CleanResult(False, "", "too_short", raw_chars, len(step2))
    if is_repetitive(step2):
        return CleanResult(False, "", "repetitive", raw_chars, len(step2))
    if not is_english(step2):
        return CleanResult(False, "", "non_english", raw_chars, len(step2))
    if strict_ocr and is_ocr_garble(step2):
        return CleanResult(False, "", "ocr", raw_chars, len(step2))
    return CleanResult(True, step2, "kept", raw_chars, len(step2))
```

### 5.3 dedup.py

```python
"""Pure helpers for Phase 2 (dedup + contamination strip)."""

from __future__ import annotations

import hashlib
import re

_WS = re.compile(r"\s+")
_WORD = re.compile(r"[a-z0-9]+")


def normalize(text: str) -> str:
    return _WS.sub(" ", text.lower()).strip()


def words(text: str) -> list[str]:
    return _WORD.findall(normalize(text))


def exact_hash(text: str) -> str:
    return hashlib.blake2b(normalize(text).encode("utf-8"), digest_size=16).hexdigest()


def word_ngrams(tokens: list[str], n: int) -> set[int]:
    """Fast native hash of word n-grams (contam set and doc grams share a process)."""
    if len(tokens) < n:
        return set()
    return {hash(tuple(tokens[i : i + n])) for i in range(len(tokens) - n + 1)}


def shingles(tokens: list[str], k: int) -> set[bytes]:
    if len(tokens) < k:
        return {" ".join(tokens).encode("utf-8")} if tokens else set()
    return {" ".join(tokens[i : i + k]).encode("utf-8") for i in range(len(tokens) - k + 1)}
```

### 5.4 modal\_app.py

```python
"""Modal App for the from-scratch 125M SLM build (Phases 0 to 4)."""

from __future__ import annotations

import modal

import config

app = modal.App(config.PROJECT)

# CPU base. All pip/apt build steps MUST come before add_local_* (Modal rule).
_cpu_base = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("wamerican")  # /usr/share/dict/words for the OCR gate
    .pip_install(
        "datasets==3.6.0",
        "huggingface_hub==0.34.4",
        "langdetect==1.0.9",
        "pyarrow==17.0.0",
        "datasketch==1.6.5",
    )
)
cpu_image = _cpu_base.add_local_python_source("config", "cleaning", "dedup")

volume = modal.Volume.from_name(config.VOLUME_NAME, create_if_missing=True)
VOLUMES = {config.DATA_ROOT: volume}


def _stream_source(source: "config.Source", n: int):
    from datasets import load_dataset

    ds = load_dataset(source.hf_id, source.config_name, split=source.split, streaming=True)
    for i, record in enumerate(ds):
        if i >= n:
            break
        yield record


@app.function(image=cpu_image, volumes=VOLUMES, timeout=60 * 15)
def smoke_test(n_per_source: int = 10) -> dict:
    from cleaning import clean_document

    summary: dict[str, dict] = {}
    for source in config.DATA_MIX:
        print("\n" + "=" * 78)
        print(f"SOURCE: {source.name}  ({source.hf_id}, split={source.split}, "
              f"field='{source.text_field}')")
        print("=" * 78)
        kept = 0
        reasons: dict[str, int] = {}
        for i, record in enumerate(_stream_source(source, n_per_source)):
            text = record.get(source.text_field) or ""
            if not isinstance(text, str):
                text = str(text)
            result = clean_document(text)
            reasons[result.reason] = reasons.get(result.reason, 0) + 1
            kept += int(result.kept)
            excerpt = (result.text[:240] if result.kept else text[:160]).replace("\n", " / ")
            print(f"\n[{source.name} #{i}] raw={result.raw_chars:>7} clean={result.clean_chars:>7} "
                  f"-> {result.reason.upper()}")
            print(f"    {excerpt}")
        summary[source.name] = {"streamed": n_per_source, "kept": kept, "reasons": reasons}
    print("\nSMOKE TEST SUMMARY")
    for name, s in summary.items():
        print(f"  {name:<12} kept {s['kept']}/{s['streamed']}  reasons={s['reasons']}")
    return summary


@app.function(image=cpu_image, volumes=VOLUMES, timeout=60 * 20)
def measure_sources(n_per_source: int = 2000) -> dict:
    from cleaning import clean_document

    TOTAL_ROWS = {"case-law": 282_390, "sec": 48_543, "fineweb-edu": 9_670_000}
    out: dict[str, dict] = {}
    for source in config.DATA_MIX:
        clean_chars = kept = 0
        for record in _stream_source(source, n_per_source):
            text = record.get(source.text_field) or ""
            if not isinstance(text, str):
                text = str(text)
            r = clean_document(text)
            if r.kept:
                kept += 1
                clean_chars += r.clean_chars
        avg_clean = clean_chars / n_per_source if n_per_source else 0
        total = TOTAL_ROWS[source.name]
        est = total * avg_clean / config.CHARS_PER_TOKEN
        out[source.name] = {"est_clean_tokens": int(est), "keep_rate": round(kept / n_per_source, 3)}
        print(f"{source.name:<12} keep={kept/n_per_source:.0%}  avg_clean={avg_clean:>7.0f} ch/doc  "
              f"rows={total:>9,}  est_clean_tokens={est/1e9:.2f}B")
    print(f"TOTAL est clean tokens: {sum(v['est_clean_tokens'] for v in out.values())/1e9:.2f}B")
    return out


# ---- Phase 1: stream + clean, one worker per parquet shard ----
_SOURCE_BY_NAME = {s.name: s for s in config.DATA_MIX}


@app.function(image=cpu_image, volumes=VOLUMES, timeout=60 * 60)
def clean_shard(source_name: str, url: str, shard_index: int, token_cap: int) -> dict:
    import os

    from datasets import load_dataset

    from cleaning import clean_document

    source = _SOURCE_BY_NAME[source_name]
    out_dir = f"{config.CLEAN_DIR}/{source_name}"
    os.makedirs(out_dir, exist_ok=True)
    out_path = f"{out_dir}/shard-{shard_index:03d}.txt"
    ds = load_dataset("parquet", data_files=url, split="train", streaming=True)
    streamed = kept = clean_chars = 0
    reasons: dict[str, int] = {}
    with open(out_path, "w", encoding="utf-8") as fh:
        for record in ds:
            streamed += 1
            text = record.get(source.text_field) or ""
            if not isinstance(text, str):
                text = str(text)
            r = clean_document(text, strict_ocr=source.strict_ocr)
            reasons[r.reason] = reasons.get(r.reason, 0) + 1
            if r.kept:
                fh.write(r.text.replace("\n", " ").strip() + "\n")
                kept += 1
                clean_chars += r.clean_chars
                if clean_chars / config.CHARS_PER_TOKEN >= token_cap:
                    break
    volume.commit()
    est_tokens = int(clean_chars / config.CHARS_PER_TOKEN)
    print(f"[{source_name} shard {shard_index:03d}] streamed={streamed} kept={kept} "
          f"est_tokens={est_tokens/1e6:.1f}M reasons={reasons}")
    return {"source": source_name, "shard": shard_index, "streamed": streamed,
            "kept": kept, "est_tokens": est_tokens, "reasons": reasons}


def _parquet_urls(hf_id: str, config_name: str, split: str) -> list[str]:
    import json
    import urllib.request

    api = f"https://datasets-server.huggingface.co/parquet?dataset={hf_id}"
    req = urllib.request.Request(api, headers={"User-Agent": "slm-125m"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.load(resp)
    return [f["url"] for f in data.get("parquet_files", [])
            if f.get("config") == config_name and f.get("split") == split]


@app.local_entrypoint()
def clean(fineweb_shards: int = 1, only: str = ""):
    def cfg(s):
        return s.config_name or "default"

    sources = [s for s in config.DATA_MIX if not only or s.name == only]
    work = []
    for s in sources:
        urls = _parquet_urls(s.hf_id, cfg(s), s.split)
        if s.name == "fineweb-edu":
            urls = urls[:fineweb_shards]
        per_shard_cap = s.token_budget // max(1, len(urls))
        for i, url in enumerate(urls):
            work.append((s.name, url, i, per_shard_cap))
        print(f"{s.name:<12} {len(urls)} shard(s), per-shard cap ~{per_shard_cap/1e6:.0f}M tokens")
    print(f"Launching {len(work)} clean workers...")
    results = list(clean_shard.starmap(work))
    report: dict[str, dict] = {}
    for r in results:
        agg = report.setdefault(r["source"], {"streamed": 0, "kept": 0, "est_tokens": 0, "reasons": {}})
        agg["streamed"] += r["streamed"]
        agg["kept"] += r["kept"]
        agg["est_tokens"] += r["est_tokens"]
        for k, v in r["reasons"].items():
            agg["reasons"][k] = agg["reasons"].get(k, 0) + v
    print("PHASE 1 DROP REPORT")
    total = 0
    for name, a in report.items():
        total += a["est_tokens"]
        print(f"  {name:<12} streamed={a['streamed']:>8} kept={a['kept']:>8} "
              f"est_tokens={a['est_tokens']/1e9:.2f}B drops={a['reasons']}")
    print(f"  TOTAL est_clean_tokens={total/1e9:.2f}B")
    save_report.remote(report)


ocr_image = cpu_image

# ---- Phase 2: dedup + contamination strip ----
SHINGLE_K = 5
MINHASH_PERM = 32
MINHASH_THRESHOLD = 0.8
DECONTAM_NGRAM = 13
SIG_DIR = f"{config.DATA_ROOT}/tmp/minhash_sigs"
NEAR_DUPS_PATH = f"{config.DATA_ROOT}/tmp/near_dups.json"
DECONTAM_SOURCES = {"case-law", "sec"}
CLEAN_SHARDS = {"case-law": 10, "sec": 5, "fineweb-edu": 5}


def _build_contamination_ngrams() -> set:
    from datasets import load_dataset

    from dedup import word_ngrams, words

    grams: set = set()
    for hf_id, cfg_name in [("casehold/casehold", None), ("coastalcph/lex_glue", "case_hold")]:
        try:
            urls = _parquet_urls(hf_id, cfg_name or "default", "test")
            if not urls:
                urls = _parquet_urls(hf_id, cfg_name or "default", "train")
            ds = load_dataset("parquet", data_files=urls, split="train", streaming=True)
            for rec in ds:
                text = " ".join(str(v) for v in rec.values() if isinstance(v, str))
                grams |= word_ngrams(words(text), DECONTAM_NGRAM)
        except Exception as e:
            print(f"  [decontam] could not load {hf_id}: {e}")
    print(f"  [decontam] {len(grams):,} eval 13-grams loaded")
    return grams


@app.function(image=cpu_image, volumes=VOLUMES, timeout=60 * 20, cpu=4.0, memory=4_096)
def minhash_shard(shard_basename: str) -> dict:
    import os

    import numpy as np
    from datasketch import MinHash

    from dedup import shingles, words

    path = f"{config.CLEAN_DIR}/case-law/{shard_basename}"
    sigs, idxs = [], []
    with open(path, encoding="utf-8") as fh:
        for idx, line in enumerate(fh):
            line = line.rstrip("\n")
            if not line:
                continue
            m = MinHash(num_perm=MINHASH_PERM)
            sh = list(shingles(words(line), SHINGLE_K))
            if sh:
                m.update_batch(sh)
            sigs.append(m.hashvalues.astype(np.uint64))
            idxs.append(idx)
    os.makedirs(SIG_DIR, exist_ok=True)
    np.savez(f"{SIG_DIR}/{shard_basename}.npz",
             sigs=np.vstack(sigs), idxs=np.asarray(idxs, dtype=np.int64))
    volume.commit()
    print(f"[minhash {shard_basename}] {len(idxs):,} docs")
    return {"shard": shard_basename, "n": len(idxs)}


@app.function(image=cpu_image, volumes=VOLUMES, timeout=60 * 20, memory=8_192)
def build_near_dups() -> int:
    import glob
    import json
    import os

    import numpy as np
    from datasketch import MinHash, MinHashLSH

    near: dict[str, list[int]] = {}
    lsh = MinHashLSH(threshold=MINHASH_THRESHOLD, num_perm=MINHASH_PERM)
    for npz_path in sorted(glob.glob(f"{SIG_DIR}/*.npz")):
        shard = os.path.basename(npz_path)[: -len(".npz")]
        data = np.load(npz_path)
        for row, idx in zip(data["sigs"], data["idxs"]):
            m = MinHash(num_perm=MINHASH_PERM, hashvalues=row)
            if lsh.query(m):
                near.setdefault(shard, []).append(int(idx))
            else:
                lsh.insert(f"{shard}:{int(idx)}", m)
    os.makedirs(os.path.dirname(NEAR_DUPS_PATH), exist_ok=True)
    with open(NEAR_DUPS_PATH, "w", encoding="utf-8") as fh:
        json.dump(near, fh)
    volume.commit()
    total = sum(len(v) for v in near.values())
    print(f"[near-dups] {total:,} case-law near-duplicates")
    return total


@app.function(image=cpu_image, volumes=VOLUMES, timeout=60 * 30, cpu=4.0, memory=8_192)
def write_corpus_shard(source_name: str, shard_basename: str) -> dict:
    import json
    import os

    from dedup import exact_hash, word_ngrams, words

    near: set[int] = set()
    if source_name == "case-law":
        with open(NEAR_DUPS_PATH, encoding="utf-8") as fh:
            near = set(json.load(fh).get(shard_basename, []))
    contam = _build_contamination_ngrams() if source_name in DECONTAM_SOURCES else None
    in_path = f"{config.CLEAN_DIR}/{source_name}/{shard_basename}"
    out_dir = f"{config.CORPUS_DIR}/{source_name}"
    os.makedirs(out_dir, exist_ok=True)
    seen: set[str] = set()
    kept = clean_chars = 0
    reasons = {"near_dup": 0, "exact_dup": 0, "contaminated": 0, "kept": 0}
    with open(in_path, encoding="utf-8") as fin, \
            open(f"{out_dir}/{shard_basename}", "w", encoding="utf-8") as fout:
        for idx, line in enumerate(fin):
            text = line.rstrip("\n")
            if not text:
                continue
            if idx in near:
                reasons["near_dup"] += 1
                continue
            h = exact_hash(text)
            if h in seen:
                reasons["exact_dup"] += 1
                continue
            if contam and (word_ngrams(words(text), DECONTAM_NGRAM) & contam):
                reasons["contaminated"] += 1
                continue
            seen.add(h)
            fout.write(text + "\n")
            kept += 1
            clean_chars += len(text)
            reasons["kept"] += 1
    volume.commit()
    print(f"[corpus {source_name}/{shard_basename}] kept={kept} drops={reasons}")
    return {"source": source_name, "shard": shard_basename, "kept": kept,
            "est_tokens": int(clean_chars / config.CHARS_PER_TOKEN), "reasons": reasons}


@app.function(image=cpu_image, volumes=VOLUMES)
def write_phase2_report(results: list) -> dict:
    import json

    report: dict[str, dict] = {}
    for r in results:
        if not r:
            continue
        agg = report.setdefault(r["source"], {"kept": 0, "est_tokens": 0,
              "reasons": {"near_dup": 0, "exact_dup": 0, "contaminated": 0, "kept": 0}})
        agg["kept"] += r["kept"]
        agg["est_tokens"] += r["est_tokens"]
        for k, v in r["reasons"].items():
            agg["reasons"][k] = agg["reasons"].get(k, 0) + v
    total = sum(v["est_tokens"] for v in report.values())
    print("PHASE 2 REPORT")
    for name, a in report.items():
        print(f"  {name:<12} kept={a['kept']:>8} est_tokens={a['est_tokens']/1e9:.2f}B drops={a['reasons']}")
    print(f"  TOTAL corpus est tokens: {total/1e9:.2f}B")
    with open(f"{config.CORPUS_DIR}/phase2_report.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    volume.commit()
    return report


@app.local_entrypoint()
def dedup(compute_sigs: bool = True):
    if compute_sigs:
        names = [f"shard-{i:03d}.txt" for i in range(CLEAN_SHARDS["case-law"])]
        print(f"1/3 MinHash signatures for {len(names)} case-law shards...")
        list(minhash_shard.map(names))
    print("2/3 building near-dup set (LSH)...")
    build_near_dups.remote()
    work = [(src, f"shard-{i:03d}.txt") for src, n in CLEAN_SHARDS.items() for i in range(n)]
    print(f"3/3 writing final corpus ({len(work)} shards, parallel)...")
    results = list(write_corpus_shard.starmap(work))
    write_phase2_report.remote(results)


# ---- Phase 3: train the 16K byte-level BPE tokenizer ----
ml_image = _cpu_base.pip_install("transformers==4.46.3").add_local_python_source(
    "config", "cleaning", "dedup")


def _corpus_line_iter():
    import glob

    for path in sorted(glob.glob(f"{config.CORPUS_DIR}/*/*.txt")):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.rstrip("\n")
                if line:
                    yield line


@app.function(image=ml_image, volumes=VOLUMES, timeout=60 * 40, cpu=8.0, memory=16_384)
def train_tokenizer() -> dict:
    import os

    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast

    specials = list(config.SPECIAL_TOKENS.values()) + list(config.EXTRA_CHAT_TOKENS)
    tok = Tokenizer(models.BPE(unk_token=config.SPECIAL_TOKENS["unk_token"]))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=config.MODEL.vocab_size, special_tokens=specials,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=True)
    print("training BPE...")
    tok.train_from_iterator(_corpus_line_iter(), trainer=trainer)
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        bos_token=config.SPECIAL_TOKENS["bos_token"],
        eos_token=config.SPECIAL_TOKENS["eos_token"],
        pad_token=config.SPECIAL_TOKENS["pad_token"],
        unk_token=config.SPECIAL_TOKENS["unk_token"],
        additional_special_tokens=list(config.EXTRA_CHAT_TOKENS))
    os.makedirs(config.TOKENIZER_DIR, exist_ok=True)
    fast.save_pretrained(config.TOKENIZER_DIR)
    volume.commit()
    for s in ["The plaintiff shall bear the burden of proof by a preponderance of the evidence.",
              "The Company's net revenues increased 12% year over year pursuant to the agreement."]:
        ids = fast.encode(s)
        print(f"  '{s[:40]}...' -> {len(ids)} tokens | roundtrip={fast.decode(ids).strip() == s}")
    print(f"vocab_size={fast.vocab_size}")
    return {"vocab_size": fast.vocab_size}


@app.local_entrypoint()
def tokenizer():
    train_tokenizer.remote()


# ---- Phase 4: tokenize + pack into uint16 1024-token windows, split 99/1 ----
TOKENIZE_SHARDS = {"case-law": 4, "sec": 6, "fineweb-edu": 4}
ENCODE_BATCH = 1_000


@app.function(image=ml_image, volumes=VOLUMES, timeout=60 * 40, cpu=8.0, memory=16_384)
def tokenize_shard(source_name: str, shard_index: int, num_shards: int) -> dict:
    import glob
    import os

    import numpy as np
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(config.TOKENIZER_DIR)
    eos_id = tok.convert_tokens_to_ids(config.SPECIAL_TOKENS["eos_token"])
    seq_len = config.SEQ_LEN
    os.makedirs(config.TRAIN_TOKENS_DIR, exist_ok=True)
    os.makedirs(config.VAL_TOKENS_DIR, exist_ok=True)
    train_path = f"{config.TRAIN_TOKENS_DIR}/{source_name}-{shard_index:03d}.bin"
    val_path = f"{config.VAL_TOKENS_DIR}/{source_name}-{shard_index:03d}.bin"
    buf: list[int] = []
    win_count = n_train = n_val = 0
    corpus_files = sorted(glob.glob(f"{config.CORPUS_DIR}/{source_name}/*.txt"))

    def _doc_iter():
        for path in corpus_files:
            with open(path, encoding="utf-8") as fh:
                for idx, line in enumerate(fh):
                    if idx % num_shards == shard_index:
                        line = line.rstrip("\n")
                        if line:
                            yield line

    with open(train_path, "wb") as ftr, open(val_path, "wb") as fva:
        batch: list[str] = []

        def _flush():
            nonlocal win_count, n_train, n_val
            if not batch:
                return
            for ids in tok(batch, add_special_tokens=False)["input_ids"]:
                buf.extend(ids)
                buf.append(eos_id)
            while len(buf) >= seq_len:
                window = np.asarray(buf[:seq_len], dtype=np.uint16)
                del buf[:seq_len]
                if win_count % config.VAL_EVERY_N_WINDOWS == 0:
                    window.tofile(fva)
                    n_val += 1
                else:
                    window.tofile(ftr)
                    n_train += 1
                win_count += 1

        for doc in _doc_iter():
            batch.append(doc)
            if len(batch) >= ENCODE_BATCH:
                _flush()
                batch = []
        _flush()
    volume.commit()
    print(f"[{source_name} {shard_index:03d}] train_win={n_train} val_win={n_val} "
          f"train_tok={n_train*seq_len/1e6:.1f}M")
    return {"source": source_name, "shard": shard_index, "train_windows": n_train,
            "val_windows": n_val, "train_tokens": n_train * seq_len, "val_tokens": n_val * seq_len}


@app.function(image=ml_image, volumes=VOLUMES)
def write_token_index(results: list) -> dict:
    import json

    shards = [r for r in results if r]
    total = {"seq_len": config.SEQ_LEN, "dtype": config.TOKENS_DTYPE,
             "train_windows": sum(r["train_windows"] for r in shards),
             "val_windows": sum(r["val_windows"] for r in shards),
             "train_tokens": sum(r["train_tokens"] for r in shards),
             "val_tokens": sum(r["val_tokens"] for r in shards), "shards": shards}
    with open(f"{config.TOKENS_DIR}/index.json", "w", encoding="utf-8") as fh:
        json.dump(total, fh, indent=2)
    volume.commit()
    print(f"index: train={total['train_tokens']/1e9:.2f}B tok ({total['train_windows']} win), "
          f"val={total['val_tokens']/1e6:.1f}M tok ({total['val_windows']} win)")
    return total


@app.local_entrypoint()
def tokenize():
    work = [(name, i, n) for name, n in TOKENIZE_SHARDS.items() for i in range(n)]
    print(f"Launching {len(work)} tokenize workers...")
    results = list(tokenize_shard.starmap(work))
    write_token_index.remote(results)


# ---- OCR-threshold analysis (optional; informs config.CLEAN.nonword_ratio_max) ----
@app.function(image=ocr_image, timeout=60 * 15)
def ocr_sample(n_docs: int = 3000) -> dict:
    import re

    from cleaning import clean_document

    with open("/usr/share/dict/words", encoding="utf-8", errors="ignore") as fh:
        vocab = {w.strip().lower() for w in fh if w.strip().isalpha()}
    tokre = re.compile(r"[A-Za-z]{3,}")
    source = _SOURCE_BY_NAME["case-law"]
    ratios: list[float] = []
    for record in _stream_source(source, n_docs):
        text = record.get(source.text_field) or ""
        if not isinstance(text, str):
            text = str(text)
        r = clean_document(text)
        if not r.kept:
            continue
        toks = [t.lower() for t in tokre.findall(r.text)]
        if len(toks) < 50:
            continue
        ratios.append(sum(1 for t in toks if t not in vocab) / len(toks))
    ratios.sort()
    n = len(ratios)
    for t in [0.10, 0.15, 0.20, 0.25, 0.30]:
        d = sum(1 for x in ratios if x > t)
        print(f"  drop if non-word ratio >{int(t*100)}%: {d} docs ({d/n:.1%})")
    return {"scored": n}


@app.local_entrypoint()
def ocr(n_docs: int = 3000):
    ocr_sample.remote(n_docs)


@app.function(image=cpu_image, volumes=VOLUMES)
def save_report(report: dict) -> None:
    import json

    with open(f"{config.CLEAN_DIR}/phase1_report.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    volume.commit()


@app.local_entrypoint()
def main(n_per_source: int = 10):
    smoke_test.remote(n_per_source)


@app.local_entrypoint()
def measure(n_per_source: int = 2000):
    measure_sources.remote(n_per_source)
```

* * *

## 6\. RUN THE PIPELINE (phase by phase)

Before every command:

```bash
source .env.local && export MODAL_TOKEN_ID MODAL_TOKEN_SECRET
```

### Phase 0: smoke test + measure (cost about 0)

```bash
modal run modal_app.py            # smoke test: 10 docs/source, clean, print
modal run modal_app.py::measure   # project true token yield per source
```

Expect the smoke test to keep about 9 or 10 of 10 docs per source. Expect `measure` to report roughly case-law ~0.8B, sec ~1.1B, fineweb ~11B available, which is why the mix is legal-first, not 70/20/10.

### Phase 1: stream + clean (cost about 0; a few minutes)

```bash
modal run modal_app.py::clean --fineweb-shards 5
```

Fans out 16 workers (case-law 10 + sec 5 + fineweb 5), one per parquet shard. Writes `/data/clean/<source>/shard-XX.txt`. To redo one source only: `modal run modal_app.py::clean --only case-law`. Expect about 718,000 docs streamed, about 698,000 kept (~97 percent), roughly 2.68B proxy tokens. The OCR gate (case-law only) drops ~1,400 scanned-garbage docs.

Optional OCR-threshold check (already set to 0.20 in config):

```bash
modal run modal_app.py::ocr
```

### Phase 2: dedup + decontaminate (cost about 0; about 6 minutes)

```bash
modal run modal_app.py::dedup
```

Three parallel stages: MinHash signatures per case-law shard, one fast LSH near-dup pass, then one writer per shard (near-dup + exact-dup + contamination). Writes `/data/corpus/<source>/shard-XX.txt`. Expect ~24,000 case-law docs removed as CaseHOLD-contaminated, ~1,600 near-dups, ~2,000 SEC exact-dups. Corpus ~670,000 docs, ~2.40B proxy tokens. To reuse signatures already computed: `--no-compute-sigs`.

### Phase 3: train the tokenizer (cost about 0; about 4 minutes)

```bash
modal run modal_app.py::tokenizer
```

Trains a fresh 16,384 byte-level BPE on the whole corpus, saves to `/data/tokenizer/`. Expect `vocab_size=16384` and two `roundtrip=True` lines.

### Phase 4: tokenize + pack + split 99/1 (cost about 0; about 10 minutes)

```bash
modal run modal_app.py::tokenize
```

14 workers encode, append `<|eos|>` after each doc, pack into 1024-token uint16 windows, route every 100th window to val. Writes `/data/tokens/train/*.bin`, `/data/tokens/val/*.bin`, `/data/tokens/index.json`. Expect: `index: train=2.19B tok (~2.14M win), val=22.1M tok (~21.6K win)`.

### Verify

```bash
modal volume ls slm-125m /tokens       # train/ val/ index.json
modal volume ls slm-125m /tokenizer    # tokenizer.json + configs
modal volume get slm-125m /tokens/index.json ./index.json
```

* * *

## 7\. EXPECTED FINAL RESULT AND COST

Final training data on the Volume:

*   train: about 2.19 billion tokens (about 2,138,970 windows of 1024)
    
*   val: about 22.1 million tokens (about 21,614 windows), a clean 1.0 percent split
    
*   realized mix: case-law ~863M, sec ~861M, fineweb ~465M (about 40 / 40 / 20)
    

Cost of Phases 0 to 4: well under 1 US dollar (this run billed about 0.18 USD), all CPU. Wall-clock: about 40 minutes of useful compute. Check spend:

```bash
modal billing report --start 2026-07-08 --json \
  | python3 -c "import sys,json; print(sum(float(r['cost']) for r in json.load(sys.stdin)))"
```

* * *

## 8\. GOTCHAS (do not relearn these)

1.  The data split is legal-first, NOT 70/20/10. The legal sources only hold ~2B tokens; see section 2.
    
2.  Modal image rule: all `pip_install` and `apt_install` steps must come BEFORE `add_local_python_source`, or the image build errors.
    
3.  `langdetect` is slow per document. `is_english` is ASCII-first and only calls `langdetect` on the ambiguous 90 to 99 percent ASCII band. Keep that ordering.
    
4.  The OCR gate needs the system wordlist. The base image installs the apt package `wamerican` which provides `/usr/share/dict/words`.
    
5.  Modal can preempt a long single container and restart it from zero. Keep every heavy step fanned out one worker per shard (Phases 1, 2, 4 already do this).
    
6.  Token counts in Phases 1 and 2 are a chars/4 PROXY. Only Phase 4 (the real tokenizer) gives true counts; they came in about 8 percent lower than the proxy.
    
7.  `casehold/casehold` parquet may not resolve; the LexGLUE `case_hold` config covers the same CaseHOLD benchmark for decontamination, so this is fine.
    
8.  To change how much data you keep, edit `token_budget` per source in `config.py`, then re-run Phase 1 and every phase after it in order.