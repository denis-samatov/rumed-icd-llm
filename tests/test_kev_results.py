"""Independently reconstruct the published typed-decision evidence without Kev or MLX."""

import importlib.util
import json
import shutil
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("verify_kev", ROOT / "scripts/verify_kev_results.py")
VERIFIER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFIER)


def copy_evidence(target):
    (target / "results").mkdir()
    for path in (ROOT / "results").glob("kev*"):
        shutil.copyfile(path, target / "results" / path.name)
    (target / "data/raw").mkdir(parents=True)
    shutil.copyfile(ROOT / "data/raw/SHA256SUMS", target / "data/raw/SHA256SUMS")
    shutil.copyfile(ROOT / "data/icd_labels_ru.json", target / "data/icd_labels_ru.json")


def test_published_kev_metrics_and_probabilities():
    VERIFIER.verify()


def test_modified_prediction_bytes_fail(tmp_path):
    copy_evidence(tmp_path)
    path = tmp_path / "results/kev_test_predictions.jsonl"
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(AssertionError):
        VERIFIER.verify(tmp_path)


def test_modified_metric_fails(tmp_path):
    copy_evidence(tmp_path)
    path = tmp_path / "results/kev_test.json"
    result = json.loads(path.read_text(encoding="utf-8"))
    result["metrics"]["hit@1"] += 1
    path.write_text(json.dumps(result), encoding="utf-8")
    with pytest.raises(AssertionError):
        VERIFIER.verify(tmp_path)
