"""Portable checks for the frozen typed-choice protocol and its metric semantics."""

import json
from pathlib import Path

import pytest

from rumed_icd.kev_eval import (
    calibration_report,
    criteria_for,
    paired_report,
    rank_probabilities,
)


def test_all_105_dictionary_labels_survive_as_criteria():
    root = Path(__file__).resolve().parents[1]
    labels = json.loads((root / "data/icd_labels_ru.json").read_text(encoding="utf-8"))["labels"]
    codes = [r["code"] for r in labels]
    assert len(codes) == len(set(codes)) == 105
    assert list(criteria_for(labels, codes)) == sorted(codes)


def test_missing_or_reordered_labels_fail():
    codes = ["A00", "A01", "A02"]
    labels = [{"code": c, "description": "description"} for c in reversed(codes)]
    with pytest.raises(ValueError, match="exactly"):
        criteria_for(labels, codes)


def test_top3_uses_unrounded_probabilities_and_stable_ties():
    assert rank_probabilities([.00001, .24999, .25, .5],
                              ["A00", "A01", "A02", "A03"]) == ["A03", "A02", "A01"]
    assert rank_probabilities([.25, .25, .25, .25],
                              ["A00", "A01", "A02", "A03"]) == ["A00", "A01", "A02"]


@pytest.mark.parametrize("p", [[0, 1], [0, .5, float("nan")], [0, .5, .4], [-.1, .1, 1]])
def test_invalid_probability_distribution_fails(p):
    with pytest.raises(ValueError, match="complete normalized"):
        rank_probabilities(p, ["A00", "A01", "A02"])


def test_identical_paired_predictions_have_zero_difference():
    a = [["A00", "A01", "A02"], ["A00", "A02", "A01"]]
    assert paired_report(a, a, ["A00", "A01"]) == {
        f"hit@{k}": {"difference_pp": 0.0, "ci95_pp": [0.0, 0.0]} for k in (1, 3)}


def test_perfect_calibration_and_brier_including_confidence_one():
    result = calibration_report([[1, 0, 0], [0, 1, 0]], ["A00", "A01"],
                                ["A00", "A01", "A02"])
    assert result["multiclass_brier"] == result["ece_10_equal_width_bins"] == 0
    assert result["bins"][-1]["n"] == 2
    assert sum(row["n"] for row in result["bins"]) == 2


def test_brier_is_sum_over_classes_and_not_divided_by_class_count():
    result = calibration_report([[0, 1, 0]], ["A00"], ["A00", "A01", "A02"])
    assert result["multiclass_brier"] == 2
    assert result["ece_10_equal_width_bins"] == 1
