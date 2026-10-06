"""Recompute the published Qwen3 metrics and paired CIs without MLX or model weights.

Usage: uv run python scripts/verify_qwen3_results.py
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from rumed_icd.data import RAW_DIR, load_split
from rumed_icd.metrics import bootstrap_ci, hits, is_valid_icd10, report

ROOT = Path(__file__).resolve().parents[1]


def verify(root: Path = ROOT) -> None:
    summary = json.loads((root / "results/qwen3_summary.json").read_text())
    seed, resamples = summary["seed"], summary["bootstrap_resamples"]
    hashes = {name.lstrip("*").removeprefix("./"): digest for digest, name in
              (line.split() for line in (root / "data/raw/SHA256SUMS").read_text().splitlines())}
    for split in ("dev", "test"):
        saved = summary["splits"][split]
        assert saved["raw_dataset_sha256"] == hashes[f"{split}_v1.jsonl"]
        path = root / "results" / f"qwen3_predictions_{split}.jsonl"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == saved["predictions_sha256"]
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        assert len(rows) == saved["n"]
        assert len({row["idx"] for row in rows}) == len(rows)
        gold = [row["gold"] for row in rows]
        # Downloaded raw data is optional; when present, also check labels and target order.
        if (RAW_DIR / f"{split}_v1.jsonl").exists():
            records = load_split(split)
            assert [(r.idx, r.code) for r in records] == [(r["idx"], r["gold"]) for r in rows]
        values = {}
        for method, field in (("local_zero_shot", "base_top3"), ("local_lora", "lora_top3")):
            predictions = [row[field] for row in rows]
            assert all(len(p) == len(set(p)) == 3 for p in predictions)
            assert all(is_valid_icd10(code) for p in predictions for code in p)
            actual = report(predictions, gold, seed=seed)
            metrics = json.loads((root / "results" / f"{method}_{split}.json").read_text())
            assert metrics == saved["methods"][method]
            assert metrics["model_revision"] == summary["revision"]
            expected_sha = summary["adapter_sha256"] if method == "local_lora" else None
            assert metrics["adapter_sha256"] == expected_sha
            assert all(metrics[key] == value for key, value in actual.items())
            values[method] = {}
            for k in (1, 3):
                h = hits(predictions, gold, k)
                values[method][k] = h
                assert int(h.sum()) == metrics[f"correct@{k}"]
        for k in (1, 3):
            delta = values["local_lora"][k] - values["local_zero_shot"][k]
            lo, hi = bootstrap_ci(delta, n_resamples=resamples, seed=seed)
            paired = saved["paired_difference"][f"hit@{k}"]
            assert paired["difference_pp"] == round(float(delta.mean()) * 100, 2)
            assert paired["ci95_pp"] == [round(lo * 100, 2), round(hi * 100, 2)]
            a, b = values["local_lora"][k], values["local_zero_shot"][k]
            for name, mask in (("lora_only_correct", (a == 1) & (b == 0)),
                               ("base_only_correct", (a == 0) & (b == 1)),
                               ("both_correct", (a == 1) & (b == 1)),
                               ("both_wrong", (a == 0) & (b == 0))):
                assert paired[name] == int(mask.sum())
        print(f"{split}: {len(rows)} predictions, metrics and paired CIs verified")


if __name__ == "__main__":
    verify()
