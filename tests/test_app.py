"""HTTP API tests. Vision encoder is stubbed so these stay independent of SigLIP."""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

import app as app_module

VALID_RESULT = {
    "model_id": 123,
    "model": "40724_001",
    "image_type": "main",
    "embedding_model": "google/siglip-base-patch16-224",
    "embedding_dimension": 768,
    "embedding": [0.0124, -0.0831, 0.0417],
    "image_hash": "a" * 64,
}


class DummyEncoder:
    embedding_model = "dummy"
    embedding_dimension = 768

    def encode(self, _image: Any) -> Any:
        import numpy as np

        return np.zeros(768, dtype=np.float32)

    def classify_relevance(self, _vector: Any) -> tuple[str, dict[str, float]]:
        return "footwear", {"footwear": 0.42, "irrelevant": 0.11}


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("RECO_API_KEY", "test-key")
    monkeypatch.setattr(app_module, "VisionEncoder", DummyEncoder)

    class _StubRecommender:
        model_version = "the_pretty_model_v1"
        dimension = 64

        @classmethod
        def load(cls, **_kwargs: Any) -> _StubRecommender:
            return cls()

        def recommend(self, model_code: str, *, limit: int = 100) -> list[dict[str, Any]]:
            return []

    monkeypatch.setattr(app_module, "RecommenderService", _StubRecommender)
    with TestClient(app_module.app) as test_client:
        yield test_client


def test_health_unauthenticated(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"


def test_embeddings_rejects_missing_key(client: TestClient) -> None:
    response = client.post("/embeddings/models", json={"models": []})
    assert response.status_code == 401


def test_embeddings_rejects_wrong_key(client: TestClient) -> None:
    response = client.post(
        "/embeddings/models",
        json={"models": []},
        headers={"X-API-Key": "nope"},
    )
    assert response.status_code == 401


def test_embeddings_rejects_bad_payload(client: TestClient) -> None:
    response = client.post(
        "/embeddings/models",
        json={"items": []},
        headers={"X-API-Key": "test-key"},
    )
    assert response.status_code == 400
    assert response.json() == {"detail": "invalid request/payload"}


def test_embeddings_returns_contract_payload(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    def fake_run(payload: dict[str, Any], encoder: Any, batch_size: int) -> dict[str, Any]:
        seen["payload"] = payload
        seen["encoder"] = encoder
        seen["batch_size"] = batch_size
        return {
            "results": [VALID_RESULT],
            "errors": [
                {
                    "model_id": 123,
                    "model": "40724_001",
                    "image_type": "side",
                    "error": "IMAGE_NOT_FOUND",
                }
            ],
        }

    monkeypatch.setattr(app_module, "run", fake_run)
    response = client.post(
        "/embeddings/models",
        json={
            "models": [
                {
                    "model_id": 123,
                    "model": "40724_001",
                    "images": {
                        "main": "https://media.adler.co.il/app/products/40724_001.jpg",
                        "pers": "https://media.adler.co.il/app/products/40724_001_pers.jpg",
                    },
                }
            ]
        },
        headers={"X-API-Key": "test-key"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["results"][0]["model_id"] == 123
    assert body["results"][0]["model"] == "40724_001"
    assert body["results"][0]["image_type"] == "main"
    assert body["results"][0]["embedding_model"] == "google/siglip-base-patch16-224"
    assert body["results"][0]["embedding_dimension"] == 768
    assert body["results"][0]["embedding"] == [0.0124, -0.0831, 0.0417]
    assert len(body["results"][0]["image_hash"]) == 64
    assert body["errors"] == [
        {
            "model_id": 123,
            "model": "40724_001",
            "image_type": "side",
            "error": "IMAGE_NOT_FOUND",
        }
    ]
    assert seen["payload"]["models"][0]["model_id"] == 123
    assert "side" not in seen["payload"]["models"][0]["images"]
    assert isinstance(seen["encoder"], DummyEncoder)


def test_embeddings_encoder_failure_is_500(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("encoder exploded")

    monkeypatch.setattr(app_module, "run", boom)
    response = client.post(
        "/embeddings/models",
        json={"models": [{"model_id": 1, "model": "x", "images": {"main": "https://example.com/a.jpg"}}]},
        headers={"X-API-Key": "test-key"},
    )
    assert response.status_code == 500
    assert response.json() == {"detail": "encoder/service-level failure"}


def test_query_rejects_missing_key(client: TestClient) -> None:
    response = client.post("/embeddings/query", json={"image_base64": "a" * 16})
    assert response.status_code == 401


def test_query_rejects_bad_payload(client: TestClient) -> None:
    response = client.post(
        "/embeddings/query",
        json={"image": "nope"},
        headers={"X-API-Key": "test-key"},
    )
    assert response.status_code == 400
    assert response.json() == {"detail": "invalid request/payload"}


def test_query_returns_embedding_and_relevance(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    import numpy as np

    seen: dict[str, Any] = {}

    def fake_run_query(image_bytes: bytes, encoder: Any) -> dict[str, Any]:
        seen["bytes"] = image_bytes
        seen["encoder"] = encoder
        return {
            "embedding": np.zeros(3, dtype=float).tolist(),
            "embedding_model": "google/siglip-base-patch16-224",
            "embedding_dimension": 3,
            "relevance": "irrelevant",
            "scores": {"footwear": 0.1, "irrelevant": 0.4},
        }

    monkeypatch.setattr(app_module, "run_query", fake_run_query)
    response = client.post(
        "/embeddings/query",
        json={"image_base64": "aGVsbG8="},
        headers={"X-API-Key": "test-key"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["relevance"] == "irrelevant"
    assert body["embedding_model"] == "google/siglip-base-patch16-224"
    assert body["embedding_dimension"] == 3
    assert body["scores"]["irrelevant"] == 0.4
    assert seen["bytes"] == b"hello"
    assert isinstance(seen["encoder"], DummyEncoder)
