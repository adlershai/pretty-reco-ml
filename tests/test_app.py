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


class DummyTextEncoder:
    model_name = "dummy-text"
    embedding_dimension = 3

    def encode(self, texts: list[str]) -> Any:
        import numpy as np

        rows = []
        for text in texts:
            if "order" in text.lower() or "הזמנה" in text:
                rows.append([1.0, 0.0, 0.0])
            else:
                rows.append([0.0, 1.0, 0.0])
        return np.asarray(rows, dtype=np.float32)


class DummyEncoder:
    embedding_model = "dummy"
    embedding_dimension = 768

    def encode(self, _image: Any) -> Any:
        import numpy as np

        return np.zeros(768, dtype=np.float32)

    def encode_batch(self, images: list[Any]) -> Any:
        import numpy as np

        return np.zeros((len(images), 768), dtype=np.float32)

    def classify_relevance(self, _vector: Any) -> tuple[str, dict[str, float]]:
        return "footwear", {"footwear": 0.42, "irrelevant": 0.11}


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("RECO_API_KEY", "test-key")
    monkeypatch.setattr(app_module, "VisionEncoder", DummyEncoder)
    monkeypatch.setattr(app_module, "TextEncoder", DummyTextEncoder)

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


def test_match_image_rejects_missing_key(client: TestClient) -> None:
    response = client.post("/match/image", json={"image_base64": "a" * 16})
    assert response.status_code == 401


def test_match_image_returns_candidates_without_embedding(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    client.app.state.catalog = object()

    def fake_run_match(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        return {
            "status": "needs_verification",
            "match": None,
            "candidates": [
                {
                    "model": "52792_006",
                    "score": 0.91,
                    "best_image_type": "side",
                    "model_id": 9,
                }
            ],
            "preprocessing": {
                "shoe_isolated": True,
                "crop": [0, 90, 200, 280],
                "reason": "studio_panel",
                "image_size": [200, 500],
            },
            "verifier": {"requestType": "watiImageIdentity", "verify_top": 10},
            "relevance": "footwear",
            "scores": {"footwear": 0.4, "irrelevant": 0.1},
            "embedding_model": "google/siglip-base-patch16-224",
        }

    monkeypatch.setattr(app_module, "run_match", fake_run_match)
    response = client.post(
        "/match/image",
        json={"image_base64": "aGVsbG8=", "top": 10},
        headers={"X-API-Key": "test-key"},
    )
    assert response.status_code == 200
    body = response.json()
    assert "embedding" not in body
    assert body["status"] == "needs_verification"
    assert body["match"] is None
    assert body["preprocessing"]["shoe_isolated"] is True
    assert body["candidates"][0]["best_image_type"] == "side"


def test_match_image_unavailable_catalog_is_503(client: TestClient) -> None:
    client.app.state.catalog = None

    def boom() -> None:
        raise RuntimeError("db down")

    client.app.state.load_catalog = boom
    response = client.post(
        "/match/image",
        json={"image_base64": "aGVsbG8="},
        headers={"X-API-Key": "test-key"},
    )
    assert response.status_code == 503
    assert response.json() == {"detail": "catalog is not loaded"}


def test_match_image_decide_unique_same_is_match(client: TestClient) -> None:
    response = client.post(
        "/match/image/decide",
        json={
            "candidates": [
                {"model": "51604_A", "score": 0.78, "best_image_type": "pers", "model_id": 1},
                {"model": "50583_C", "score": 0.79, "best_image_type": "pers", "model_id": 2},
            ],
            "openai_output": {
                "candidates": [
                    {
                        "model": "51604_A",
                        "verdict": "same",
                        "shape": 0.96,
                        "pattern": 0.95,
                        "material": 0.9,
                        "details": 0.94,
                        "color_placement": 0.8,
                        "contradictions": [],
                        "reason": "same last",
                    },
                    {
                        "model": "50583_C",
                        "verdict": "different",
                        "shape": 0.4,
                        "pattern": 0.2,
                        "material": 0.5,
                        "details": 0.4,
                        "color_placement": 0.8,
                        "contradictions": ["round vs pointed"],
                        "reason": "toe",
                    },
                ]
            },
        },
        headers={"X-API-Key": "test-key"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "match"
    assert body["match"]["model"] == "51604_A"
    assert body["match"]["score"] == 0.78


def test_match_image_decide_rejects_missing_key(client: TestClient) -> None:
    response = client.post("/match/image/decide", json={"candidates": [], "openai_output": {}})
    assert response.status_code == 401


def test_text_embeddings_returns_vectors(client: TestClient) -> None:
    response = client.post(
        "/embeddings/text",
        json={"texts": ["order stuck", "store hours"]},
        headers={"X-API-Key": "test-key"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["embedding_model"] == "dummy-text"
    assert body["embedding_dimension"] == 3
    assert body["results"][0]["embedding"] == [1.0, 0.0, 0.0]
    assert body["results"][1]["embedding"] == [0.0, 1.0, 0.0]


def test_text_similarity_returns_ranked_ids(client: TestClient) -> None:
    response = client.post(
        "/similarity/text",
        json={
            "query_text": "order stuck",
            "top": 2,
            "candidates": [
                {"id": "store", "embedding": [0.0, 1.0, 0.0]},
                {"id": "order", "embedding": [1.0, 0.0, 0.0]},
                {"id": "mixed", "embedding": [0.7, 0.7, 0.0]},
            ],
        },
        headers={"X-API-Key": "test-key"},
    )
    assert response.status_code == 200
    body = response.json()
    assert [row["id"] for row in body["results"]] == ["order", "mixed"]
    assert body["results"][0]["score"] > body["results"][1]["score"]


def test_text_similarity_rejects_dimension_mismatch(client: TestClient) -> None:
    response = client.post(
        "/similarity/text",
        json={
            "query_text": "order stuck",
            "candidates": [{"id": "bad", "embedding": [1.0, 0.0]}],
        },
        headers={"X-API-Key": "test-key"},
    )
    assert response.status_code == 400
    assert response.json()["detail"] == "VECTOR_DIMENSION_MISMATCH"
