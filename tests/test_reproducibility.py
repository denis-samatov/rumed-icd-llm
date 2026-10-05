import hashlib
import json
import threading

import pytest

from rumed_icd import evaluate, llm
from rumed_icd.data import Record, load_jsonl, load_split
from rumed_icd.metrics import bootstrap_ci, report

TRAIN = [Record("a", "кашель", "J06"), Record("b", "жажда", "E11")]
TARGET = [Record("q1", "кашель", "J06"), Record("q2", "жажда", "E11")]


def cached_row(rec):
    messages = llm.build_messages(rec.text, [], ["E11", "J06"])
    digest = hashlib.sha256(json.dumps([llm.MODEL, messages], ensure_ascii=False).encode())
    return {"key": f"{rec.idx}:{digest.hexdigest()[:16]}", "idx": rec.idx,
            "content": json.dumps({"codes": [rec.code]}),
            "usage": {"prompt_tokens": 100, "completion_tokens": 10}}


def test_complete_cache_replays_without_credentials_or_network(tmp_path, monkeypatch):
    monkeypatch.setattr(llm, "ROOT", tmp_path)
    path = tmp_path / "results/cache/zero_shot_test.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text("\n".join(json.dumps(cached_row(r)) for r in TARGET) + "\n")

    def forbidden(*args, **kwargs):
        pytest.fail("cached replay must not read credentials or call an API")

    monkeypatch.setattr(llm, "load_api_key", forbidden)
    monkeypatch.setattr(llm, "call_api", forbidden)
    predictions, usage = llm.predict("zero_shot", "test", TRAIN, TARGET)
    assert predictions == [["J06"], ["E11"]]
    assert usage.calls == 2
    assert usage.new_requests == 0


def test_successful_paid_response_survives_another_request_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(llm, "ROOT", tmp_path)
    monkeypatch.setattr(llm, "load_api_key", lambda: "test-key")
    second_completed = threading.Event()

    def api(messages, key):
        if messages[-1]["content"].endswith("кашель"):
            assert second_completed.wait(timeout=5)
            raise RuntimeError("simulated request failure")
        second_completed.set()
        return {"choices": [{"message": {"content": '{"codes": ["E11"]}'}}], "usage": {}}

    monkeypatch.setattr(llm, "call_api", api)
    with pytest.raises(RuntimeError, match="simulated request failure"):
        llm.predict("zero_shot", "test", TRAIN, TARGET, workers=2)
    path = tmp_path / "results/cache/zero_shot_test.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [r["idx"] for r in rows] == ["q2"]

    calls = []

    def resumed_api(messages, key):
        calls.append(messages)
        return {"choices": [{"message": {"content": '{"codes": ["J06"]}'}}], "usage": {}}

    monkeypatch.setattr(llm, "call_api", resumed_api)
    predictions, usage = llm.predict("zero_shot", "test", TRAIN, TARGET, workers=2)
    assert predictions == [["J06"], ["E11"]]  # Target order, rather than cache arrival order.
    assert len(calls) == 1
    assert usage.new_requests == 1
    assert usage.calls == 2


def test_usage_fallback_does_not_count_cached_input_twice():
    usage = llm.Usage()
    usage.add({"prompt_tokens": 100, "prompt_cache_hit_tokens": 80, "completion_tokens": 10})
    assert usage.cache_hit == 80
    assert usage.cache_miss == 20


@pytest.mark.parametrize("limit", [0, -1])
def test_invalid_limit_fails_before_loading_or_calling_api(limit, monkeypatch):
    monkeypatch.setattr(evaluate, "load_split", lambda *_: pytest.fail("must validate first"))
    with pytest.raises(ValueError, match="limit"):
        evaluate.run("zero_shot", "dev", limit=limit)


def test_evaluation_rejects_training_split(monkeypatch):
    monkeypatch.setattr(evaluate, "load_split", lambda *_: pytest.fail("must validate first"))
    with pytest.raises(ValueError, match="split"):
        evaluate.run("tfidf", "train")


@pytest.mark.parametrize("row", [[], {"idx": "a", "symptoms": " ", "code": "J06"},
                                 {"idx": "a", "symptoms": "кашель", "code": 123}])
def test_data_loader_rejects_malformed_records(tmp_path, row):
    path = tmp_path / "test.jsonl"
    path.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError):
        load_jsonl(path)


def test_load_split_rejects_changed_bytes(tmp_path):
    path = tmp_path / "dev_v1.jsonl"
    path.write_text('{"idx":"a","symptoms":"кашель","code":"J06"}\n')
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    (tmp_path / "SHA256SUMS").write_text(f"{digest}  dev_v1.jsonl\n")
    assert load_split("dev", tmp_path)[0].code == "J06"
    path.write_text(path.read_text().replace("J06", "E11"))
    with pytest.raises(ValueError, match="checksum"):
        load_split("dev", tmp_path)


def test_empty_metrics_fail_with_clear_error():
    with pytest.raises(ValueError, match="empty"):
        report([], [])
    with pytest.raises(ValueError, match="empty"):
        bootstrap_ci([])
