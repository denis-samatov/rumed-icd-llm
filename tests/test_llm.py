from rumed_icd.data import Record
from rumed_icd.llm import Retriever, build_messages, examples_for, parse_codes

TRAIN = [
    Record("a", "кашель насморк температура", "J06"),
    Record("b", "жажда сухость во рту", "E11"),
    Record("c", "боль в пояснице при наклоне", "M54"),
    Record("d", "насморк заложенность носа кашель", "J06"),
]


def test_parse_codes_json_subcodes_and_dedup():
    assert parse_codes('{"codes": ["M54.5", "m54", "I11", "G90"]}') == ["M54", "I11", "G90"]


def test_parse_codes_falls_back_to_text():
    assert parse_codes("Ответ: E11, I10") == ["E11", "I10"]
    assert parse_codes('{"codes": "J06"}') == ["J06"]
    assert parse_codes("не знаю") == []


def test_build_messages_lists_codes_examples_and_query():
    msgs = build_messages("кашель", TRAIN[:1], ["E11", "J06"])
    user = msgs[1]["content"]
    assert msgs[0]["role"] == "system"
    assert "Разрешённые коды: E11, J06" in user
    assert "Код: J06" in user
    assert user.endswith("Жалобы пациента: кашель")


def test_rag_retrieves_from_train_with_most_similar_last():
    r = Retriever(TRAIN)
    ex = examples_for("rag", Record("q", "сильный кашель и насморк", "J06"), TRAIN, r, 2, 0)
    assert {e.code for e in ex} == {"J06"}
    assert ex[-1] in r.top_k("сильный кашель и насморк", 1)


def test_few_shot_examples_are_fixed_across_queries():
    q1, q2 = Record("q1", "x", "J06"), Record("q2", "y", "E11")
    assert examples_for("few_shot", q1, TRAIN, None, 2, 0) == examples_for(
        "few_shot", q2, TRAIN, None, 2, 0)
    assert examples_for("zero_shot", q1, TRAIN, None, 2, 0) == []
