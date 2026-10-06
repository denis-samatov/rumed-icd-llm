import pytest

from rumed_icd.data import Record
from rumed_icd.local_llm import (
    build_messages,
    build_trie,
    code_token_ids,
    prediction_key,
    train_lora,
)


class CharTokenizer:
    """One token per character, like Qwen's split of a letter and single digits."""

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]


def test_code_token_ids_requires_equal_lengths():
    ids = code_token_ids(CharTokenizer(), ["M54", "I11"])
    assert ids == {"M54": [77, 53, 52], "I11": [73, 49, 49]}
    with pytest.raises(ValueError, match="different lengths"):
        code_token_ids(CharTokenizer(), ["M54", "I1"])


def test_trie_lists_next_tokens_for_every_proper_prefix():
    ids = code_token_ids(CharTokenizer(), ["M54", "M53", "I11"])
    trie = build_trie(ids)
    assert trie[()] == {ord("M"), ord("I")}
    assert trie[(ord("M"),)] == {ord("5")}
    assert trie[(ord("M"), ord("5"))] == {ord("4"), ord("3")}
    # full codes are leaves, not prefixes
    assert (ord("M"), ord("5"), ord("4")) not in trie


def test_local_prompt_asks_for_one_code_and_keeps_examples():
    msgs = build_messages("кашель", [Record("a", "насморк", "J06")], ["E11", "J06"])
    assert "одним кодом" in msgs[0]["content"]
    assert "Код: J06" in msgs[1]["content"]
    assert msgs[1]["content"].endswith("Жалобы пациента: кашель")


def test_prediction_cache_is_invalidated_by_prompt_adapter_and_vocabulary():
    original = prediction_key("a", [1, 2], ["J06"], None)
    assert original == prediction_key("a", [1, 2], ["J06"], None)
    assert original != prediction_key("a", [1, 3], ["J06"], None)
    assert original != prediction_key("a", [1, 2], ["J06"], "changed-weights")
    assert original != prediction_key("a", [1, 2], ["J06", "E11"], None)


def test_code_vocabulary_cannot_be_empty():
    with pytest.raises(ValueError, match="nonempty"):
        code_token_ids(CharTokenizer(), [])


def test_training_refuses_overwrite_before_loading_mlx(tmp_path):
    adapter = tmp_path / "existing.safetensors"
    adapter.write_bytes(b"user weights")
    records = [Record("a", "кашель", "J06")]
    with pytest.raises(FileExistsError, match="overwrite"):
        train_lora(records, records, adapter)
    assert adapter.read_bytes() == b"user weights"


@pytest.mark.parametrize("max_steps", [0, -1])
def test_invalid_training_budget_fails_before_loading_mlx(tmp_path, max_steps):
    records = [Record("a", "кашель", "J06")]
    with pytest.raises(ValueError, match="max_steps"):
        train_lora(records, records, tmp_path / "new.safetensors", max_steps=max_steps)
