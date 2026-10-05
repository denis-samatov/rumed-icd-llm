"""Methods 1–2: zero-shot, few-shot and RAG prompting through the DeepSeek API.

All three methods share one prompt template, one model, temperature 0 and thinking disabled.
They differ only in the examples placed in the prompt:
  zero_shot — none;
  few_shot  — the same K random training cases for every query (seed fixed);
  rag       — the K training cases most similar to the query (TF-IDF char n-gram cosine).
Retrieval only ever searches the training split.
"""

from __future__ import annotations

import hashlib
import http.client
import json
import os
import random
import re
import time
import urllib.error
import urllib.request
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from rumed_icd.data import Record

ROOT = Path(__file__).resolve().parents[2]
API_URL = "https://api.deepseek.com/chat/completions"
MODEL = "deepseek-flash"
# USD per 1M tokens, peak hours (upper bound), api-docs.deepseek.com, checked 2026-10-04
PRICE_PEAK = {"cache_hit": 0.006, "cache_miss": 0.3, "output": 1.2}

SYSTEM = (
    "Ты — врач-кодировщик МКБ-10. По жалобам пациента на приёме у врача поликлиники выбери "
    "три наиболее вероятных кода диагноза, от самого вероятного к менее вероятному. "
    "Используй только коды из списка разрешённых (трёхзначные рубрики МКБ-10). "
    'Ответь JSON-объектом вида {"codes": ["X00", "Y00", "Z00"]} без пояснений.'
)


def load_api_key() -> str:
    key = os.environ.get("DEEPSEEK_API_KEY")
    env = ROOT / ".env"
    if not key and env.exists():
        for line in env.read_text().splitlines():
            if line.startswith("DEEPSEEK_API_KEY="):
                key = line.split("=", 1)[1].strip()
    if not key:
        raise RuntimeError("DEEPSEEK_API_KEY is not set (environment or .env)")
    return key


class Retriever:
    def __init__(self, train: Sequence[Record]) -> None:
        self.train = list(train)
        self.vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2,
                                   sublinear_tf=True)
        self.matrix = self.vec.fit_transform([r.text for r in self.train])

    def top_k(self, text: str, k: int) -> list[Record]:
        sims = (self.matrix @ self.vec.transform([text]).T).toarray().ravel()
        return [self.train[i] for i in np.argsort(-sims)[:k]]


def build_messages(query: str, examples: Sequence[Record], codes: Sequence[str]) -> list[dict]:
    parts = ["Разрешённые коды: " + ", ".join(codes)]
    if examples:
        parts.append("Примеры случаев с установленным кодом:")
        parts += [f"Жалобы: {e.text}\nКод: {e.code}" for e in examples]
    parts.append(f"Жалобы пациента: {query}")
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "\n\n".join(parts)}]


CODE = re.compile(r"[A-Z]\d{2}")


def parse_codes(content: str) -> list[str]:
    """Extract up to 3 distinct 3-character codes; tolerate subcodes like 'M54.5'."""
    try:
        raw = json.loads(content).get("codes", [])
        raw = [str(c) for c in raw] if isinstance(raw, list) else [str(raw)]
    except (json.JSONDecodeError, AttributeError):
        raw = [content]
    out: list[str] = []
    for item in raw:
        for m in CODE.findall(item.upper()):
            if m not in out:
                out.append(m)
    return out[:3]


@dataclass
class Usage:
    calls: int = 0
    cache_hit: int = 0
    cache_miss: int = 0
    output: int = 0
    errors: list[str] = field(default_factory=list)

    def add(self, u: dict) -> None:
        self.calls += 1
        self.cache_hit += u.get("prompt_cache_hit_tokens", 0)
        self.cache_miss += u.get("prompt_cache_miss_tokens", u.get("prompt_tokens", 0))
        self.output += u.get("completion_tokens", 0)

    def cost_usd_peak(self) -> float:
        p = PRICE_PEAK
        return (self.cache_hit * p["cache_hit"] + self.cache_miss * p["cache_miss"]
                + self.output * p["output"]) / 1e6


def call_api(messages: list[dict], key: str, retries: int = 5) -> dict:
    body = json.dumps({
        "model": MODEL,
        "messages": messages,
        "temperature": 0,
        "max_tokens": 64,
        "thinking": {"type": "disabled"},
        "response_format": {"type": "json_object"},
    }).encode()
    req = urllib.request.Request(API_URL, data=body, method="POST", headers={
        "Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            if e.code not in (429, 500, 502, 503, 504) or attempt == retries - 1:
                raise RuntimeError(f"HTTP {e.code}: {e.read()[:300]!r}") from e
        except (urllib.error.URLError, http.client.HTTPException, OSError, json.JSONDecodeError):
            if attempt == retries - 1:
                raise
        time.sleep(2 ** attempt)
    raise RuntimeError("unreachable")


def examples_for(method: str, rec: Record, train: Sequence[Record], retriever: Retriever | None,
                 k: int, seed: int) -> list[Record]:
    if method == "zero_shot":
        return []
    if method == "few_shot":
        return random.Random(seed).sample(list(train), k)
    if method == "rag":
        assert retriever is not None
        # most similar case last, closest to the query
        return list(reversed(retriever.top_k(rec.text, k)))
    raise ValueError(f"unknown method {method!r}")


def predict(method: str, split: str, train: Sequence[Record], target: Sequence[Record],
            k: int = 15, seed: int = 0, workers: int = 8) -> tuple[list[list[str]], Usage]:
    codes = sorted({r.code for r in train})
    retriever = Retriever(train) if method == "rag" else None
    key = load_api_key()
    cache_path = ROOT / "results" / "cache" / f"{method}_{split}.jsonl"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache: dict[str, dict] = {}
    if cache_path.exists():
        for line in cache_path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            cache[row["key"]] = row

    jobs = []
    for rec in target:
        msgs = build_messages(rec.text, examples_for(method, rec, train, retriever, k, seed), codes)
        h = hashlib.sha256(json.dumps([MODEL, msgs], ensure_ascii=False).encode()).hexdigest()
        jobs.append((rec, msgs, f"{rec.idx}:{h[:16]}"))

    usage = Usage()
    todo = [j for j in jobs if j[2] not in cache]

    def run(job: tuple) -> dict:
        rec, msgs, ck = job
        resp = call_api(msgs, key)
        return {"key": ck, "idx": rec.idx,
                "content": resp["choices"][0]["message"]["content"], "usage": resp.get("usage", {})}

    with ThreadPoolExecutor(workers) as pool, cache_path.open("a", encoding="utf-8") as fh:
        for row in pool.map(run, todo):
            cache[row["key"]] = row
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()

    preds = []
    for _, _, ck in jobs:
        row = cache[ck]
        usage.add(row["usage"])
        preds.append(parse_codes(row["content"]))
    return preds, usage
