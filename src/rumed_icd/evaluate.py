"""Run a method on a split and write metrics to results/<method>_<split>.json.

Usage: uv run python -m rumed_icd.evaluate --method tfidf --split dev
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

from rumed_icd.data import load_split
from rumed_icd.metrics import report

RESULTS = Path(__file__).resolve().parents[2] / "results"


LLM_METHODS = ("zero_shot", "few_shot", "rag")
LOCAL_METHODS = ("local_zero_shot", "local_few_shot", "local_rag", "local_lora")


def run(method: str, split: str, limit: int | None = None) -> dict:
    if split not in ("dev", "test"):
        raise ValueError("evaluation split must be dev or test")
    if limit is not None and limit < 1:
        raise ValueError("limit must be a positive integer")
    if method not in ("tfidf", *LLM_METHODS, *LOCAL_METHODS):
        raise ValueError(f"unknown method {method!r}")
    train = load_split("train")
    target = load_split(split)[:limit]
    t0 = time.perf_counter()
    extra: dict = {}
    if method == "tfidf":
        from rumed_icd.baselines.tfidf import TfidfBaseline

        preds = TfidfBaseline().fit(train).predict_topk(target, k=3)
    elif method in LLM_METHODS:
        from rumed_icd import llm

        preds, usage = llm.predict(method, split, train, target)
        extra = {
            "model": llm.MODEL,
            "k_examples": 0 if method == "zero_shot" else 15,
            "tokens": {"cache_hit": usage.cache_hit, "cache_miss": usage.cache_miss,
                       "output": usage.output},
            "cost_usd_peak_upper_bound": round(usage.cost_usd_peak(), 4),
            "new_requests": usage.new_requests,
            "reused_responses": usage.calls - usage.new_requests,
        }
    elif method in LOCAL_METHODS:
        from rumed_icd import local_llm

        preds, extra = local_llm.predict(method.removeprefix("local_"), split, train, target)
    else:
        raise ValueError(f"unknown method {method!r}")
    labels = {r.code for r in train}
    metrics = report(preds, [r.code for r in target])
    metrics["outside_label_set_top1_rate"] = round(
        100 * sum((p[0] if p else "") not in labels for p in preds) / len(preds), 2)
    metrics.update(
        method=method,
        split=split,
        n_train=len(train),
        n_classes_train=len(labels),
        seconds=round(time.perf_counter() - t0, 1),
        python=platform.python_version(),
        **extra,
    )
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="tfidf", choices=["tfidf", *LLM_METHODS, *LOCAL_METHODS])
    ap.add_argument("--split", default="dev", choices=["dev", "test"])
    ap.add_argument("--limit", type=int, default=None, help="first N examples (smoke runs)")
    ap.add_argument("--output-dir", type=Path, default=RESULTS,
                    help="where to write metrics; use a scratch directory for reruns")
    args = ap.parse_args()
    metrics = run(args.method, args.split, args.limit)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_first{args.limit}" if args.limit else ""
    out = args.output_dir / f"{args.method}_{args.split}{suffix}.json"
    out.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
