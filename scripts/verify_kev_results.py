"""Verify public Kev predictions, probabilities, timing summaries and paired CIs on CPU."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from rumed_icd.data import load_split
from rumed_icd.kev_eval import (
    BASE,
    BASE_REVISION,
    CHECKPOINT_HASHES,
    MODEL,
    REVISION,
    RUNTIME_REVISION,
    calibration_report,
    canonical_sha,
    criteria_for,
    paired_report,
    rank_probabilities,
    sha,
)
from rumed_icd.metrics import report

ROOT = Path(__file__).resolve().parents[1]


def verify(root: Path = ROOT) -> None:
    protocol = json.loads((root / "results/kev_protocol.json").read_text(encoding="utf-8"))
    assert (protocol["model"], protocol["revision"], protocol["runtime_revision"]) == (
        MODEL, REVISION, RUNTIME_REVISION)
    assert (protocol["base"], protocol["base_revision"]) == (BASE, BASE_REVISION)
    assert protocol["adapter_sha256"] == CHECKPOINT_HASHES["adapter_model.safetensors"]
    assert protocol["head_sha256"] == CHECKPOINT_HASHES["head.pt"]
    dictionary = root / "data/icd_labels_ru.json"
    assert sha(dictionary) == protocol["label_dictionary_sha256"]
    codes = protocol["codes"]
    labels = json.loads(dictionary.read_text(encoding="utf-8"))["labels"]
    assert len(criteria_for(labels, codes)) == 105
    hashes = {name.lstrip("*").removeprefix("./"): digest for digest, name in
              (line.split() for line in
               (root / "data/raw/SHA256SUMS").read_text(encoding="utf-8").splitlines())}
    assert protocol["train_sha256"] == hashes["train_v1.jsonl"]
    api = json.loads((root / "results/kev_api_smoke.json").read_text(encoding="utf-8"))
    assert api["sdk"] == "typesafe-sdk" and api["sdk_version"] == "0.6.0"
    assert api["wide_choice_options"] == len(codes)
    for key in ("backend", "dtype", "temperature"):
        assert api[key] == protocol[key]
    answers = api["answers"]
    assert {k: v["type"] for k, v in answers.items()} == {
        "team": "choice", "duplicate": "noul", "urgency": "score"}
    assert 0 <= answers["duplicate"]["noul"] <= 1
    assert 0 <= answers["urgency"]["score"] <= 2
    for key in ("team", "urgency"):
        p = np.array(list(answers[key]["probabilities"].values()))
        assert np.isfinite(p).all() and np.all((p >= 0) & (p <= 1))
        assert abs(p.sum() - 1) < .02
    for split, n in (("dev", 848), ("test", 822)):
        result = json.loads((root / f"results/kev_{split}.json").read_text(encoding="utf-8"))
        assert result["split"] == split and result["full_split"] is True
        assert result["n_train"] == 4690 and result["n_classes_train"] == len(codes)
        assert result["protocol_sha256"] == canonical_sha(protocol)
        assert result["raw_dataset_sha256"] == hashes[f"{split}_v1.jsonl"]
        path = root / f"results/kev_{split}_predictions.jsonl"
        assert result["predictions_sha256"] == sha(path)
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        assert len(rows) == len({r["idx"] for r in rows}) == n
        # No complaints or private context belong in public evidence.
        assert all(set(r) == {"idx", "gold", "kev_top3", "tfidf_top3", "kev_probabilities",
                              "kev_seconds", "input_tokens"} for r in rows)
        gold, kev, tfidf = [r["gold"] for r in rows], [], []
        probabilities = [r["kev_probabilities"] for r in rows]
        for row in rows:
            assert row["gold"] in codes
            assert rank_probabilities(row["kev_probabilities"], codes) == row["kev_top3"]
            for field, target in (("kev_top3", kev), ("tfidf_top3", tfidf)):
                p = row[field]
                assert len(p) == len(set(p)) == 3 and set(p) <= set(codes)
                target.append(p)
            assert row["kev_seconds"] > 0 and row["input_tokens"] > 0
        assert result["metrics"] == report(kev, gold)
        assert result["tfidf_control"] == report(tfidf, gold)
        assert result["paired_vs_tfidf"] == paired_report(kev, tfidf, gold)
        assert result["calibration"] == calibration_report(probabilities, gold, codes)
        timings = np.array([r["kev_seconds"] for r in rows])
        latency = result["latency"]
        assert abs(latency["inference_seconds"] - timings.sum()) < .002
        assert abs(latency["median_ms"] - np.median(timings) * 1000) < .002
        assert abs(latency["p95_ms"] - np.quantile(timings, .95) * 1000) < .002
        if (root / f"data/raw/{split}_v1.jsonl").exists():
            records = load_split(split, raw_dir=root / "data/raw")
            assert [(r.idx, r.code) for r in records] == [(r["idx"], r["gold"]) for r in rows]
        print(f"{split}: {n} predictions, probabilities, metrics and paired CIs verified")


if __name__ == "__main__":
    verify()
