"""Independent scalar statistics are checked on known successes and invalid alignment."""

import importlib.util
from pathlib import Path

import pytest

from rumed_icd.metrics import report

spec = importlib.util.spec_from_file_location(
    "qualification", Path(__file__).resolve().parents[1] / "scripts/qualify_baseline.py"
)
qualification = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qualification)


def test_independent_scalar_bootstrap_matches_vectorized_metrics():
    predictions = [["A01"], ["A01", "B02"], ["A01", "B02", "C03"], ["D04"]]
    gold = ["A01", "B02", "C03", "E05"]
    canonical = report(predictions, gold)
    for k, successes in ((1, 1), (3, 3)):
        actual = qualification.independent_statistics(predictions, gold, k)
        assert actual["successes"] == successes
        assert actual["rate_percent"] == canonical[f"hit@{k}"]
        assert actual["bootstrap_ci95_percent"] == canonical[f"hit@{k}_ci95"]
        lo, hi = actual["wilson_ci95_percent"]
        assert 0 <= lo < actual["rate_percent"] < hi <= 100


@pytest.mark.parametrize(("predictions", "gold"), [([], []), ([["A01"]], [])])
def test_independent_statistics_rejects_empty_or_misaligned_records(predictions, gold):
    with pytest.raises(ValueError, match="aligned"):
        qualification.independent_statistics(predictions, gold, 1)
