"""Local Kev-0.8B zero-shot evaluation with all 105 options in one choice question.

Requires the separately installed, pinned Kev runtime (Python 3.12). Metrics and
evidence verification need only this project's ordinary CPU dependencies.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np

from rumed_icd.data import load_split
from rumed_icd.metrics import bootstrap_ci, hits, report

ROOT = Path(__file__).resolve().parents[2]
MODEL = "jaredpalmer/kev-0.8b"
REVISION = "bf75a6a8848ea6960ff2ed108d9ed44c2941174f"
RUNTIME_REVISION = "5e42a7a03f28134853dd3ff77461457e921e5ec1"
BASE = "Qwen/Qwen3.5-0.8B-Base"
BASE_REVISION = "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68"
CHECKPOINT_HASHES = {
    "adapter_model.safetensors": "9b908623acb162118575f4e7a94524f9c139c335be4bfb74d6cfceca01e1885a",
    "head.pt": "f400bd12802b2b105ae45d6b03774a158a3db4fccff42413734ddca2e5c920b6",
    "adapter_config.json": "748acb2cda88454cb1ba69d745ba336f3fcb5486eac349e90960c8b8d8d3e854",
}
INSTRUCTIONS = (
    "По жалобам пациента выбери наиболее вероятный код диагноза МКБ-10. "
    "Используй только предложенные варианты с описаниями диагнозов."
)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha(value: dict) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True).encode()
    return hashlib.sha256(raw).hexdigest()


def criteria_for(labels: list[dict], codes: list[str]) -> dict[str, str]:
    if len(codes) < 3 or len(codes) != len(set(codes)) or codes != sorted(codes):
        raise ValueError("need at least three unique, sorted train codes")
    if [r["code"] for r in labels] != codes:
        raise ValueError("dictionary must match every sorted train code exactly")
    if any(not isinstance(r["description"], str) or not r["description"].strip()
           for r in labels):
        raise ValueError("empty label description")
    return {r["code"]: r["description"] for r in labels}


def rank_probabilities(probabilities: list[float], codes: list[str]) -> list[str]:
    values = np.asarray(probabilities, dtype=float)
    if (values.shape != (len(codes),) or not np.isfinite(values).all()
            or np.any(values < 0) or np.any(values > 1)
            or not np.isclose(values.sum(), 1, atol=1e-5) or len(codes) < 3):
        raise ValueError("expected a complete normalized probability distribution")
    return [codes[i] for i in np.argsort(-values, kind="stable")[:3]]


def paired_report(a: list[list[str]], b: list[list[str]], gold: list[str]) -> dict:
    result = {}
    for k in (1, 3):
        delta = hits(a, gold, k) - hits(b, gold, k)
        lo, hi = bootstrap_ci(delta)
        result[f"hit@{k}"] = {"difference_pp": round(float(delta.mean()) * 100, 2),
                               "ci95_pp": [round(lo * 100, 2), round(hi * 100, 2)]}
    return result


def calibration_report(probabilities: list[list[float]], gold: list[str], codes: list[str]) -> dict:
    p = np.asarray(probabilities, dtype=float)
    if p.shape != (len(gold), len(codes)) or not gold:
        raise ValueError("probability matrix must match gold and codes")
    for row in probabilities:
        rank_probabilities(row, codes)
    positions = np.array([codes.index(g) for g in gold])
    confidence = p.max(axis=1)
    correct = p.argmax(axis=1) == positions
    # Multiclass Brier is the sum over classes, not its class-normalized version.
    brier = (p * p).sum(axis=1) - 2 * p[np.arange(len(gold)), positions] + 1
    bins, ece = [], 0.0
    bucket = np.minimum((confidence * 10).astype(int), 9)
    for i in range(10):
        mask = bucket == i
        n = int(mask.sum())
        acc = float(correct[mask].mean()) if n else None
        conf = float(confidence[mask].mean()) if n else None
        if n:
            ece += n / len(gold) * abs(acc - conf)
        bins.append({"lower": i / 10, "upper": (i + 1) / 10, "n": n,
                     "accuracy": acc, "mean_confidence": conf})
    return {"multiclass_brier": round(float(brier.mean()), 6),
            "ece_10_equal_width_bins": round(ece, 6),
            "mean_top1_probability": round(float(confidence.mean()), 6), "bins": bins,
            "note": "shipped temperature; no RuMed calibration fitted; no confidence guarantee"}


class KevScorer:
    def __init__(self, checkpoint_path: Path, criteria: dict[str, str]):
        import kev
        import torch
        from huggingface_hub import try_to_load_from_cache
        from kev.checkpoint import Checkpoint, LoadOptions

        runtime_root = Path(kev.__file__).resolve().parents[1]
        actual = subprocess.check_output(
            ["git", "-C", str(runtime_root), "rev-parse", "HEAD"], text=True
        ).strip()
        if actual != RUNTIME_REVISION:
            raise ValueError("Kev runtime revision differs from the frozen protocol")
        if subprocess.check_output(
            ["git", "-C", str(runtime_root), "status", "--porcelain"], text=True).strip():
            raise ValueError("Kev runtime contains local changes")
        torch.manual_seed(0)
        torch.set_num_threads(4)
        for name, expected in CHECKPOINT_HASHES.items():
            if sha(checkpoint_path / name) != expected:
                raise ValueError(f"Kev checkpoint bytes changed: {name}")
        base_file = try_to_load_from_cache(
            BASE, "model.safetensors-00001-of-00001.safetensors", revision=BASE_REVISION)
        expected_base_sha = "c2b1e5a17d9c1e27685d92ed9b382911ebb99955ecd89052d1721241adfbab6c"
        if not isinstance(base_file, str) or sha(Path(base_file)) != expected_base_sha:
            raise ValueError("base weights missing or changed; download pinned snapshot first")
        self.checkpoint = Checkpoint(str(checkpoint_path))
        meta = self.checkpoint.meta
        if meta.base != BASE or meta.base_revision != BASE_REVISION:
            raise ValueError("Kev checkpoint base/revision differs from the protocol")
        self.tokenizer, self.model = self.checkpoint.load("mps", LoadOptions(backend="mlx"))
        self.criteria = criteria
        self.torch = torch
        self.metadata = {"backend": self.model.backend, "dtype": self.model.dtype,
                         "head_dtype": "float32", "temperature": self.model.head.temperature,
                         "base_weights_sha256": expected_base_sha,
                         "adapter_sha256": sha(checkpoint_path / "adapter_model.safetensors"),
                         "head_sha256": sha(checkpoint_path / "head.pt")}

    def score(self, text: str) -> tuple[list[float], dict]:
        from kev.api import SystemOneRequest, to_record
        from kev.model import admit

        request = SystemOneRequest(state=text, questions={"icd": {
            "type": "choice", "instructions": INSTRUCTIONS, "criteria": self.criteria}})
        record, _ = to_record(request)
        encoded = admit(self.model, self.tokenizer, record, truncate=False)
        with self.torch.inference_mode():
            probs = self.model.probs(encoded)[0].tolist()
        # All options must survive encoding, with no state/branch truncation.
        if len(probs) != len(self.criteria):
            raise ValueError("Kev failed to return every option")
        return probs, {"input_tokens": len(encoded["ids"])}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint-path", type=Path, required=True)
    ap.add_argument("--split", choices=["dev", "test"], required=True)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--output-dir", type=Path, default=ROOT / "results")
    args = ap.parse_args()
    if args.limit is not None and args.limit < 1:
        ap.error("--limit must be positive")
    train, target = load_split("train"), load_split(args.split)[:args.limit]
    codes = sorted({r.code for r in train})
    dictionary_path = ROOT / "data/icd_labels_ru.json"
    labels = json.loads(dictionary_path.read_text(encoding="utf-8"))["labels"]
    criteria = criteria_for(labels, codes)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_first{args.limit}" if args.limit else ""
    name = f"kev_{args.split}{suffix}"
    output = args.output_dir / f"{name}.json"
    predictions_path = args.output_dir / f"{name}_predictions.jsonl"
    if output.exists() or predictions_path.exists():
        raise FileExistsError("refusing to overwrite evidence; use another output directory")
    t0 = time.perf_counter()
    print("Loading pinned Kev checkpoint...", flush=True)
    scorer = KevScorer(args.checkpoint_path, criteria)
    scorer.score(target[0].text)  # Warmup, gold is not supplied to the scorer.
    load_seconds = time.perf_counter() - t0
    print(f"Loaded and warmed up in {load_seconds:.1f}s; evaluating {len(target)} cases",
          flush=True)
    metadata = scorer.metadata
    versions = {p: importlib.metadata.version(p) for p in
                ("mlx", "mlx-lm", "torch", "transformers", "peft", "numpy", "scikit-learn")}
    protocol = {"model": MODEL, "revision": REVISION, "runtime_revision": RUNTIME_REVISION,
                "base": BASE, "base_revision": BASE_REVISION, "instructions": INSTRUCTIONS,
                "codes": codes, "label_dictionary_sha256": sha(dictionary_path),
                "train_sha256": sha(ROOT / "data/raw/train_v1.jsonl"),
                "seed": 0, "bootstrap_resamples": 2000, "versions": versions,
                "question_type": "choice", "batch_size": 1, "truncation": False,
                "scoring": "all 105 options together; unrounded pointer-head probabilities",
                "selection": "one pretrained checkpoint, no RuMed tuning", **metadata}
    protocol_path = args.output_dir / "kev_protocol.json"
    if protocol_path.exists():
        if json.loads(protocol_path.read_text(encoding="utf-8")) != protocol:
            raise ValueError("protocol changed; refuse mixing experiments")
    else:
        if args.split == "test" and args.limit is None:
            raise ValueError("freeze protocol on dev before full test")
        protocol_path.write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n",
                                 encoding="utf-8", newline="\n")
    if args.split == "test" and args.limit is None:
        dev = json.loads((args.output_dir / "kev_dev.json").read_text(encoding="utf-8"))
        if not dev["full_split"] or dev["protocol_sha256"] != canonical_sha(protocol):
            raise ValueError("complete dev with the same protocol before full test")
    probabilities, preds, timings, tokens = [], [], [], []
    for i, record in enumerate(target):
        start = time.perf_counter()
        p, usage = scorer.score(record.text)
        timings.append(time.perf_counter() - start)
        probabilities.append(p)
        preds.append(rank_probabilities(p, codes))
        tokens.append(usage["input_tokens"])
        if (i + 1) % 25 == 0:
            print(f"{args.split}: {i + 1}/{len(target)}, {sum(timings):.1f}s", flush=True)
    import mlx.core as mx

    mlx_peak_bytes = mx.get_peak_memory()
    del scorer
    from rumed_icd.baselines.tfidf import TfidfBaseline

    print("Kev inference complete; fitting the paired TF-IDF control on train", flush=True)
    tfidf_start = time.perf_counter()
    control = TfidfBaseline().fit(train).predict_topk(target)
    tfidf_seconds = time.perf_counter() - tfidf_start
    rows = [{"idx": r.idx, "gold": r.code, "kev_top3": a, "tfidf_top3": b,
             "kev_probabilities": p, "kev_seconds": round(t, 6), "input_tokens": nt}
            for r, a, b, p, t, nt in zip(target, preds, control, probabilities, timings, tokens,
                                        strict=True)]
    predictions_path.write_text("".join(json.dumps(row) + "\n" for row in rows),
                                 encoding="utf-8", newline="\n")
    gold = [r.code for r in target]
    result = {"method": "kev", "split": args.split, "full_split": args.limit is None,
              "protocol_sha256": canonical_sha(protocol),
              "raw_dataset_sha256": sha(ROOT / "data/raw" / f"{args.split}_v1.jsonl"),
              "predictions_sha256": sha(predictions_path), "n_train": len(train),
              "n_classes_train": len(codes), "metrics": report(preds, gold),
              "tfidf_control": report(control, gold),
              "paired_vs_tfidf": paired_report(preds, control, gold),
              "calibration": calibration_report(probabilities, gold, codes),
              "latency": {"load_and_warmup_seconds": round(load_seconds, 3),
                          "inference_seconds": round(sum(timings), 3),
                          "median_ms": round(float(np.median(timings)) * 1000, 3),
                          "p95_ms": round(float(np.quantile(timings, .95)) * 1000, 3),
                          "includes": "tokenization + state/branch prefill + head + softmax",
                          "tfidf_fit_and_predict_seconds": round(tfidf_seconds, 3)},
              "input_tokens": {"max": max(tokens), "median": float(np.median(tokens))},
              "memory": {"mlx_peak_allocated_bytes": mlx_peak_bytes,
                         "note": "Metal allocations; not whole-process RSS or total system RAM"},
              "python": platform.python_version(), "platform": platform.platform()}
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8", newline="\n")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
