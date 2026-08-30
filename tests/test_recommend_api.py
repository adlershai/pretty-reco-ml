"""Recommendation API tests."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient

import app as app_module
from inference.recommender import (
    DEFAULT_LIMIT,
    MAX_LIMIT,
    MODEL_VERSION,
    ModelNotEncodableError,
    ModelNotFoundError,
    normalize_limit,
)
from inference.ranking import rank_customers_for_model
from inference.sanity import check_vectors


class DummyEncoder:
    embedding_model = "dummy"
    embedding_dimension = 768


class FakeRecommender:
    model_version = MODEL_VERSION
    dimension = 64

    @classmethod
    def load(cls, **_kwargs: Any) -> FakeRecommender:
        return cls()

    def recommend(self, model_code: str, *, limit: int = DEFAULT_LIMIT) -> list[dict[str, Any]]:
        if model_code == "missing":
            raise ModelNotFoundError(model_code)
        if model_code == "broken":
            raise ModelNotEncodableError(model_code)
        rows = []
        for rank in range(1, limit + 1):
            rows.append(
                {
                    "customer_id": 40_000_000 + rank,
                    "similarity_score": round(1.0 - rank * 0.001, 4),
                    "rank": rank,
                }
            )
        return rows


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("RECO_API_KEY", "test-key")
    monkeypatch.setattr(app_module, "VisionEncoder", DummyEncoder)
    monkeypatch.setattr(app_module, "RecommenderService", FakeRecommender)
    with TestClient(app_module.app) as test_client:
        yield test_client


def test_health_includes_model_metadata(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model": MODEL_VERSION, "dimension": 64}


def test_recommend_default_limit_is_100(client: TestClient) -> None:
    response = client.get("/recommend/customers/53165_003")
    assert response.status_code == 200
    body = response.json()
    assert len(body) == 100
    assert body[0]["rank"] == 1
    assert body[-1]["rank"] == 100


def test_recommend_limit_25(client: TestClient) -> None:
    response = client.get("/recommend/customers/53165_003?limit=25")
    assert response.status_code == 200
    assert len(response.json()) == 25


def test_recommend_limit_200(client: TestClient) -> None:
    response = client.get("/recommend/customers/53165_003?limit=200")
    assert response.status_code == 200
    assert len(response.json()) == 200


def test_recommend_limit_above_200_is_capped(client: TestClient) -> None:
    response = client.get("/recommend/customers/53165_003?limit=500")
    assert response.status_code == 200
    assert len(response.json()) == 200


def test_recommend_scores_descend(client: TestClient) -> None:
    body = client.get("/recommend/customers/53165_003?limit=10").json()
    scores = [row["similarity_score"] for row in body]
    assert scores == sorted(scores, reverse=True)


def test_recommend_ranks_are_sequential_from_one(client: TestClient) -> None:
    body = client.get("/recommend/customers/53165_003?limit=10").json()
    assert [row["rank"] for row in body] == list(range(1, 11))


def test_recommend_customer_ids_are_unique(client: TestClient) -> None:
    body = client.get("/recommend/customers/53165_003?limit=50").json()
    ids = [row["customer_id"] for row in body]
    assert len(ids) == len(set(ids))


def test_recommend_response_fields_only(client: TestClient) -> None:
    row = client.get("/recommend/customers/53165_003?limit=1").json()[0]
    assert set(row.keys()) == {"customer_id", "similarity_score", "rank"}
    assert isinstance(row["customer_id"], int)
    assert isinstance(row["similarity_score"], float)
    assert isinstance(row["rank"], int)


def test_recommend_unknown_model_returns_404(client: TestClient) -> None:
    response = client.get("/recommend/customers/missing")
    assert response.status_code == 404
    assert response.json() == {"error": "model_not_found", "model": "missing"}


def test_recommend_unencodable_model_returns_422(client: TestClient) -> None:
    response = client.get("/recommend/customers/broken")
    assert response.status_code == 422
    assert response.json() == {"error": "model_not_encodable", "model": "broken"}


def test_recommend_invalid_limit_returns_400(client: TestClient) -> None:
    response = client.get("/recommend/customers/53165_003?limit=0")
    assert response.status_code == 400


def test_normalize_limit_defaults_and_caps() -> None:
    assert normalize_limit(None) == DEFAULT_LIMIT
    assert normalize_limit(25) == 25
    assert normalize_limit(200) == 200
    assert normalize_limit(999) == MAX_LIMIT
    with pytest.raises(ValueError):
        normalize_limit(0)


def test_rank_customers_for_model_unit_norm_scores() -> None:
    customers = np.array([[1.0, 0.0], [0.6, 0.8], [0.0, 1.0]], dtype=np.float32)
    model = np.array([[1.0, 0.0]], dtype=np.float32)
    scores = customers @ model.T
    scores = scores[:, 0]
    ranked = rank_customers_for_model(scores, ["a", "b", "c"], top_k=3)
    assert [row["customer_id"] for row in ranked] == ["a", "b", "c"]
    assert [row["rank"] for row in ranked] == [1, 2, 3]
    assert ranked[0]["similarity_score"] >= ranked[1]["similarity_score"] >= ranked[2]["similarity_score"]


def test_vectors_remain_64d_normalized() -> None:
    rng = np.random.default_rng(0)
    vectors = rng.normal(size=(4, 64)).astype(np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / norms
    check_vectors(vectors, dim=64, name="test_vectors")
