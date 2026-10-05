import pytest

from rumed_icd.data import Record
from rumed_icd.local_llm import build_messages, build_trie, code_token_ids


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
