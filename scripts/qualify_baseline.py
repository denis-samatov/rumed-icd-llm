"""One fixed baseline + independent metric/record-overlap checks; no clinical validation.

Run from the repository with PYTHONPATH=src. Outputs contain IDs and codes, never complaints.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
from pathlib import Path

import numpy as np
import sklearn

from rumed_icd.baselines.tfidf import TfidfBaseline
from rumed_icd.data import SPLITS, load_split
from rumed_icd.metrics import report
from rumed_icd.split_audit import audit_splits, normalize_text


def independent_statistics(predictions, gold, k):
    """Scalar membership + row-wise bootstrap, separate from repository vectorized metrics."""
    if not gold or len(predictions) != len(gold):
        raise ValueError("nonempty aligned predictions and gold required")
    flags = [int(label in list(top)[:k]) for top, label in zip(predictions, gold, strict=True)]
    n, successes = len(flags), sum(flags)
    proportion = successes / n
    rng = np.random.default_rng(0)
    means = [sum(flags[i] for i in rng.integers(0, n, n)) / n for _ in range(2000)]
    lo, hi = np.quantile(means, [0.025, 0.975])
    z = 1.959963984540054
    denominator = 1 + z * z / n
    center = (proportion + z * z / (2 * n)) / denominator
    half = z * math.sqrt(proportion * (1 - proportion) / n + z * z / (4 * n * n)) / denominator
    return {
        "n": n,
        "successes": successes,
        "rate_percent": round(100 * proportion, 2),
        "bootstrap_ci95_percent": [round(100 * lo, 2), round(100 * hi, 2)],
        "wilson_ci95_percent": [100 * (center - half), 100 * (center + half)],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("use a fresh output directory to preserve previous receipts")
    splits = {split: load_split(split, args.raw_dir) for split in SPLITS}
    result = {
        "purpose": "Exposed public benchmark/software qualification; no clinical holdout.",
        "audit": audit_splits(splits),
        "input_sha256": {
            f"{s}_v1.jsonl": hashlib.sha256(
                (args.raw_dir / f"{s}_v1.jsonl").read_bytes()
            ).hexdigest()
            for s in SPLITS
        },
        "config": {
            "C": 10,
            "seed": 0,
            "fit_split": "train",
            "topk": 3,
            "bootstrap_resamples": 2000,
            "bootstrap_unit": "record",
        },
        "runtime": {
            "python": sys.version,
            "numpy": np.__version__,
            "sklearn": sklearn.__version__,
            "platform": platform.platform(),
        },
        "splits": {},
    }
    train_labels = {}
    for record in splits["train"]:
        train_labels.setdefault(normalize_text(record.text), set()).add(record.code)
    baseline = TfidfBaseline(c=10, seed=0).fit(splits["train"])
    predictions = []
    for split in ("dev", "test"):
        records = splits[split]
        top = baseline.predict_topk(records, k=3)
        gold = [r.code for r in records]
        canonical = report(top, gold, seed=0)
        independent = {f"hit@{k}": independent_statistics(top, gold, k) for k in (1, 3)}
        for k in (1, 3):
            assert independent[f"hit@{k}"]["rate_percent"] == canonical[f"hit@{k}"]
            assert independent[f"hit@{k}"]["bootstrap_ci95_percent"] == canonical[f"hit@{k}_ci95"]
        groups = {
            "normalized_train_match": [],
            "no_normalized_train_match": [],
            "train_text_has_conflicting_labels": [],
        }
        for i, record in enumerate(records):
            labels = train_labels.get(normalize_text(record.text), set())
            groups["normalized_train_match" if labels else "no_normalized_train_match"].append(i)
            if labels and len(labels | {record.code}) > 1:
                groups["train_text_has_conflicting_labels"].append(i)
            predictions.append(
                {"split": split, "idx": record.idx, "gold": record.code, "top3": list(top[i])}
            )
        subsets = {}
        for name, indices in groups.items():
            subsets[name] = {
                "n": len(indices),
                **{
                    f"hit@{k}_percent": (
                        100 * sum(gold[i] in top[i][:k] for i in indices) / len(indices)
                    )
                    if indices
                    else None
                    for k in (1, 3)
                },
            }
        assert subsets["normalized_train_match"]["n"] + subsets["no_normalized_train_match"][
            "n"
        ] == len(records)
        result["splits"][split] = {
            "canonical": canonical,
            "independent": independent,
            "descriptive_subsets": subsets,
        }
    args.output_dir.mkdir(parents=True)
    pred_bytes = "".join(json.dumps(p, sort_keys=True) + "\n" for p in predictions).encode()
    (args.output_dir / "predictions.jsonl").write_bytes(pred_bytes)
    result["prediction_sha256"] = hashlib.sha256(pred_bytes).hexdigest()
    result["limits"] = (
        "Repeated benchmark records, not independent clinical validation. "
        "Lexical matches are not proof of patient identity. Record bootstrap/Wilson "
        "assume independent records; patient/site clustering cannot be verified. "
        "Descriptive subsets are exposed, not newly held-out populations."
    )
    (args.output_dir / "qualification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "input_sha256": result["input_sha256"],
                "audit": result["audit"],
                "splits": result["splits"],
                "prediction_sha256": result["prediction_sha256"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
