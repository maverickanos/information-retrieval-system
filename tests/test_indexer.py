import json

import numpy as np
import pytest

import indexer
from indexer import Corpus, load_documents


def test_dataset_has_valid_unique_documents():
    corpus = load_documents()
    ids = [document["id"] for document in corpus.documents]
    assert len(ids) == len(set(ids))
    assert all(corpus.processed_texts)


def test_invalid_document_is_rejected(tmp_path):
    path = tmp_path / "documents.json"
    path.write_text(json.dumps([{"id": "1", "title": "Missing content"}]), encoding="utf-8")
    with pytest.raises(ValueError):
        load_documents(path)


def test_build_index_encodes_documents_in_batches(tmp_path, monkeypatch):
    documents = [
        {"id": str(position), "title": f"Document {position}", "content": f"term{position} shared"}
        for position in range(9)
    ]
    corpus = Corpus(
        documents=documents,
        tokens=[[f"term{position}", "shared"] for position in range(9)],
        processed_texts=[f"term{position} shared" for position in range(9)],
    )
    monkeypatch.setattr(indexer, "load_documents", lambda _path: corpus)

    batch_sizes = []

    class FakeModel:
        def encode(self, texts, **_kwargs):
            batch_sizes.append(len(texts))
            return np.ones((len(texts), 384), dtype="float32")

    monkeypatch.setattr(indexer, "get_embedding_model", lambda _name: FakeModel())
    artifacts_dir = tmp_path / "artifacts"
    result = indexer.build_index(artifacts_dir=artifacts_dir)

    assert batch_sizes == [4, 4, 1]
    assert result["document_count"] == 9
    assert result["embedding_shape"] == [9, 384]
    assert (artifacts_dir / "bm25.pkl").exists()
    assert (artifacts_dir / "tfidf.pkl").exists()
    assert (artifacts_dir / "faiss.index").exists()
    assert (artifacts_dir / "metadata.json").exists()