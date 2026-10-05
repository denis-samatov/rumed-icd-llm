"""Hit@k with bootstrap confidence intervals, and ICD-10 format checks."""

from __future__ import annotations

import re
from collections.abc import Sequence

import numpy as np

# Letter + two digits, optional subcategory (".x" or "x"), e.g. E11, I25.1, M54.5
ICD10 = re.compile(r"^[A-Z]\d{2}(\.?\d{1,2})?$")


def is_valid_icd10(code: str) -> bool:
    return bool(ICD10.match(code.strip().upper()))


def hits(preds: Sequence[Sequence[str]], gold: Sequence[str], k: int) -> np.ndarray:
    if len(preds) != len(gold):
        raise ValueError(f"{len(preds)} predictions for {len(gold)} gold labels")
    if k < 1:
        raise ValueError("k must be >= 1")
    return np.array([g in list(p)[:k] for p, g in zip(preds, gold, strict=True)], dtype=float)


def hit_at_k(preds: Sequence[Sequence[str]], gold: Sequence[str], k: int) -> float:
    return float(hits(preds, gold, k).mean())


def bootstrap_ci(
    values: np.ndarray, n_resamples: int = 2000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float]:
    """Percentile bootstrap CI of the mean over examples."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_resamples, len(values)))
    means = values[idx].mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def report(preds: Sequence[Sequence[str]], gold: Sequence[str], seed: int = 0) -> dict:
    out: dict = {"n": len(gold)}
    for k in (1, 3):
        h = hits(preds, gold, k)
        lo, hi = bootstrap_ci(h, seed=seed)
        out[f"hit@{k}"] = round(float(h.mean()) * 100, 2)
        out[f"hit@{k}_ci95"] = [round(lo * 100, 2), round(hi * 100, 2)]
    top1 = [p[0] if p else "" for p in preds]
    out["invalid_top1_rate"] = round(100 * sum(not is_valid_icd10(c) for c in top1) / len(top1), 2)
    return out
