"""Check the committed prediction evidence independently of the model runtime."""

import importlib.util
import json
import shutil
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "verify_qwen3", ROOT / "scripts/verify_qwen3_results.py"
)
VERIFIER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFIER)


def test_published_scores_match_predictions():
    VERIFIER.verify()


def test_changed_prediction_bytes_are_rejected(tmp_path):
    copy_evidence(tmp_path)
    path = tmp_path / "results/qwen3_predictions_dev.jsonl"
    path.write_text(path.read_text() + "\n")
    with pytest.raises(AssertionError):
        VERIFIER.verify(tmp_path)


def test_changed_metric_is_rejected(tmp_path):
    copy_evidence(tmp_path)
    path = tmp_path / "results/local_lora_dev.json"
    metric = json.loads(path.read_text())
    metric["hit@1"] += 1
    path.write_text(json.dumps(metric))
    with pytest.raises(AssertionError):
        VERIFIER.verify(tmp_path)


def copy_evidence(target):
    shutil.copytree(ROOT / "results", target / "results", ignore=shutil.ignore_patterns("cache"))
    (target / "data/raw").mkdir(parents=True)
    shutil.copyfile(ROOT / "data/raw/SHA256SUMS", target / "data/raw/SHA256SUMS")
