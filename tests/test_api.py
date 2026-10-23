from fastapi.testclient import TestClient

import api
from api import app


def test_health_endpoint():
    response = TestClient(app).get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_search_rejects_empty_query():
    response = TestClient(app).post("/search", json={"query": "   "})
    assert response.status_code in {400, 422}


def test_evaluate_rejects_invalid_k():
    response = TestClient(app).post("/evaluate", json={"k": 0})
    assert response.status_code == 422


def test_index_success_returns_json(monkeypatch):
    unload_calls = []
    monkeypatch.setattr(api.engine, "unload", lambda: unload_calls.append(True))
    monkeypatch.setattr(
        api,
        "build_index",
        lambda: {"document_count": 14, "embedding_shape": [14, 384]},
    )

    response = TestClient(app).post("/index")

    assert response.status_code == 200
    assert response.json() == {
        "status": "indexed",
        "document_count": 14,
        "embedding_shape": [14, 384],
    }
    assert unload_calls == [True]


def test_index_failure_returns_json_error(monkeypatch):
    def fail_build():
        raise RuntimeError("embedding model unavailable")

    monkeypatch.setattr(api, "build_index", fail_build)
    response = TestClient(app).post("/index")

    assert response.status_code == 500
    assert response.json()["detail"] == "Index build failed: embedding model unavailable"


def test_search_with_missing_artifacts_returns_json_error(monkeypatch):
    def fail_load():
        raise FileNotFoundError("Search artifacts are missing; call /index first")

    monkeypatch.setattr(api.engine, "_ensure_loaded", fail_load)
    response = TestClient(app).post("/search", json={"query": "retrieval"})

    assert response.status_code == 503
    assert response.json()["detail"] == "Search artifacts are missing; call /index first"