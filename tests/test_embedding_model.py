import pytest

import embedding_model


def test_embedding_model_is_initialized_once(monkeypatch):
    model = object()
    load_calls = []
    monkeypatch.setattr(embedding_model, "_model", None)
    monkeypatch.setattr(embedding_model, "_model_name", None)
    monkeypatch.setattr(
        embedding_model,
        "_load_sentence_transformer",
        lambda name: load_calls.append(name) or model,
    )

    assert embedding_model.get_embedding_model("test-model") is model
    assert embedding_model.get_embedding_model("test-model") is model
    assert load_calls == ["test-model"]


def test_embedding_model_rejects_a_second_model_name(monkeypatch):
    monkeypatch.setattr(embedding_model, "_model", object())
    monkeypatch.setattr(embedding_model, "_model_name", "first-model")

    with pytest.raises(RuntimeError, match="already loaded"):
        embedding_model.get_embedding_model("second-model")