"""Split fingerprints describe repeated records, never patient independence."""

from rumed_icd.data import Record
from rumed_icd.split_audit import audit_splits, normalize_text


def test_normalization_is_explicit_and_preserves_punctuation():
    assert normalize_text(" Ａ\t Pain\n") == "a pain"
    assert normalize_text("Pain!") != normalize_text("Pain")


def test_overlap_counts_records_keys_conflicts_and_unseen_labels_without_text():
    records = {
        "train": [Record("shared", "Pain", "A01"), Record("t2", "Pain", "B02")],
        "dev": [Record("shared", " PAIN\t", "A01"), Record("d2", "Pain", "C03")],
        "test": [Record("q", "Fever", "D04")],
    }
    audit = audit_splits(records)
    within = audit["within"]["train"]["exact_text"]
    assert within == {
        "duplicate_keys": 1,
        "records_in_duplicate_groups": 2,
        "excess_records": 1,
        "conflicting_label_keys": 1,
    }
    pair = audit["pairs"]["train:dev"]
    assert pair["idx"]["shared_keys"] == 1
    assert pair["exact_text"] == {
        "shared_keys": 1,
        "left_records": 2,
        "right_records": 1,
        "conflicting_label_keys": 1,
    }
    assert pair["normalized_text"] == {
        "shared_keys": 1,
        "left_records": 2,
        "right_records": 2,
        "conflicting_label_keys": 1,
    }
    assert audit["label_coverage"]["dev"]["unseen_in_train"] == ["C03"]
    assert audit["label_coverage"]["test"]["unseen_in_train"] == ["D04"]
    assert audit["pairs"]["train:test"]["normalized_text"]["shared_keys"] == 0
    assert "Pain" not in str(audit) and "shared" not in str(audit).replace("shared_keys", "")
