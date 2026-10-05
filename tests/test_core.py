import json

import numpy as np
import pytest

from rumed_icd.baselines.tfidf import TfidfBaseline
from rumed_icd.data import Record, load_jsonl
from rumed_icd.metrics import bootstrap_ci, hit_at_k, is_valid_icd10, report


def write_jsonl(path, rows):
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", "utf-8")


def test_load_jsonl_reads_records(tmp_path):
    p = tmp_path / "dev_v1.jsonl"
    write_jsonl(p, [{"idx": "a", "symptoms": " кашель ", "code": "J06"}])
    assert load_jsonl(p) == [Record("a", "кашель", "J06")]


def test_load_jsonl_rejects_missing_field_and_duplicates(tmp_path):
    p = tmp_path / "x.jsonl"
    write_jsonl(p, [{"idx": "a", "symptoms": "x"}])
    with pytest.raises(ValueError, match="missing"):
        load_jsonl(p)
    write_jsonl(p, [{"idx": "a", "symptoms": "x", "code": "J06"}] * 2)
    with pytest.raises(ValueError, match="duplicate"):
        load_jsonl(p)


def test_hit_at_k():
    preds = [["A00", "B00", "C00"], ["B00", "A00", "C00"], ["C00", "D00", "E00"]]
    gold = ["A00", "A00", "A00"]
    assert hit_at_k(preds, gold, 1) == pytest.approx(1 / 3)
    assert hit_at_k(preds, gold, 3) == pytest.approx(2 / 3)
    with pytest.raises(ValueError):
        hit_at_k(preds, gold[:2], 1)


def test_bootstrap_ci_brackets_mean_and_is_deterministic():
    v = np.array([1.0] * 70 + [0.0] * 30)
    lo, hi = bootstrap_ci(v, seed=1)
    assert lo < 0.7 < hi
    assert (lo, hi) == bootstrap_ci(v, seed=1)


@pytest.mark.parametrize(
    ("code", "ok"),
    [("E11", True), ("I25.1", True), ("M54.5", True), ("e11", True), ("11E", False), ("", False)],
)
def test_icd10_format(code, ok):
    assert is_valid_icd10(code) is ok


def test_tfidf_baseline_learns_separable_toy_data():
    train = [Record(f"t{i}", t, c) for i, (t, c) in enumerate(
        [("кашель насморк температура", "J06"), ("насморк кашель горло", "J06"),
         ("жажда сухость во рту слабость", "E11"), ("сухость во рту жажда", "E11"),
         ("боль в пояснице при наклоне", "M54"), ("поясница болит наклон", "M54")] * 3)]
    test = [Record("q1", "сильный кашель и насморк", "J06"), Record("q2", "жажда, сухость", "E11")]
    preds = TfidfBaseline().fit(train).predict_topk(test, k=3)
    assert [p[0] for p in preds] == ["J06", "E11"]
    assert report(preds, [r.code for r in test])["hit@1"] == 100.0
