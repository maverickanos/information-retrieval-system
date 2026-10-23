import json

import faiss
import joblib
import numpy as np
import pytest

import search_engine
from search_engine import SearchEngine
from search_engine import reciprocal_rank_fusion


def test_rrf_uses_one_based_actual_ranks():
    scores = reciprocal_rank_fusion({"bm25": ["a", "b"], "tfidf": ["b", "a"], "semantic": ["a"]})
    assert scores["a"] == 1 / 61 + 1 / 62 + 1 / 61
    assert scores["b"] == 1 / 62 + 1 / 61


def test_rrf_missing_rankings_contribute_nothing():
    scores = reciprocal_rank_fusion({"bm25": ["a"], "tfidf": [], "semantic": []})
    assert scores == {"a": 1 / 61}


def test_load_artifacts_does_not_initialize_embedding_model(tmp_path, monkeypatch):
    joblib.dump([], tmp_path / "bm25.pkl")
    joblib.dump({"vectorizer": object(), "matrix": object()}, tmp_path / "tfidf.pkl")
    index = faiss.IndexFlatIP(384)
    index.add(np.zeros((1, 384), dtype="float32"))
    faiss.write_index(index, str(tmp_path / "faiss.index"))
    (tmp_path / "metadata.json").write_text(
        json.dumps({"documents": [{"id": "1"}], "embedding_model": "test-model"}),
        encoding="utf-8",
    )

    def fail_model_load(_model_name):
        pytest.fail("Loading retrieval artifacts must not initialize the transformer")

    monkeypatch.setattr(search_engine, "get_embedding_model", fail_model_load)
    engine = SearchEngine(artifacts_dir=tmp_path)

    engine.load()

    assert engine._loaded
    assert engine.embedding_model_name == "test-model"
    assert not hasattr(engine, "model")


def test_search_with_missing_artifacts_fails_gracefully(tmp_path):
    engine = SearchEngine(artifacts_dir=tmp_path)

    with pytest.raises(FileNotFoundError, match="call /index first"):
        engine.search("retrieval")