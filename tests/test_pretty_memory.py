from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient

import app as app_module
from embeddings.pretty_memory import PrettyMemoryStore, build_index, review_weight


class MemoryEncoder:
    model_name = "dummy-memory"

    def encode(self, texts: list[str], *, prefix: str = "passage") -> Any:
        rows = []
        for text in texts:
            lowered = text.lower()
            if "return" in lowered or "החזר" in lowered:
                rows.append([1.0, 0.0, 0.0])
            elif "hours" in lowered or "שעות" in lowered:
                rows.append([0.0, 1.0, 0.0])
            else:
                rows.append([0.0, 0.0, 1.0])
        return np.asarray(rows, dtype=np.float32)


def write_snapshot(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "published_at": "2026-10-03T00:00:00Z",
        "cases": [
            {
                "case_key": "wati:return-1",
                "ticket_id": "return-1",
                "conversation_id": "c-other",
                "wa_id": "972500000099",
                "role_id": "customer_success",
                "review_status": "unreviewed",
                "case_type": "order",
                "recognition_text": "customer wants a return pickup",
                "case_text": "online order return courier pickup",
                "case_summary": "Return pickup for an online order.",
                "handling_summary": "Emailed operations.",
            },
            {
                "case_key": "wati:hours-1",
                "ticket_id": "hours-1",
                "conversation_id": "c-hours",
                "wa_id": "972500000088",
                "role_id": "customer_success",
                "review_status": "approved",
                "case_type": "static_info",
                "recognition_text": "store opening hours question",
                "case_text": "mamilla opening hours from approved maps",
                "case_summary": "Asked for Mamilla hours.",
                "handling_summary": "Answered from approved source.",
            },
            {
                "case_key": "wati:same-customer",
                "ticket_id": "same-1",
                "conversation_id": "c-live",
                "wa_id": "972500000001",
                "role_id": "customer_success",
                "review_status": "draft",
                "case_type": "order",
                "recognition_text": "order status follow-up from this customer",
                "case_text": "same customer previous return",
                "case_summary": "Previous return for this customer.",
                "handling_summary": "Already in prior context.",
            },
        ],
    }
    (directory / "current.json").write_text(json.dumps(payload), encoding="utf-8")


@pytest.fixture
def memory_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    write_snapshot(tmp_path)
    monkeypatch.setenv("RECO_API_KEY", "test-key")
    monkeypatch.setenv("PRETTY_MEMORY_SNAPSHOT_DIR", str(tmp_path))
    monkeypatch.setattr(app_module, "TextEncoder", MemoryEncoder)

    class _StubRecommender:
        model_version = "the_pretty_model_v1"
        dimension = 64

        @classmethod
        def load(cls, **_kwargs: Any) -> _StubRecommender:
            return cls()

        def recommend(self, model_code: str, *, limit: int = 100) -> list[dict[str, Any]]:
            return []

    class DummyVision:
        embedding_model = "dummy"
        embedding_dimension = 768

        def encode(self, _image: Any) -> Any:
            return np.zeros(768, dtype=np.float32)

        def encode_batch(self, images: list[Any]) -> Any:
            return np.zeros((len(images), 768), dtype=np.float32)

        def classify_relevance(self, _vector: Any) -> tuple[str, dict[str, float]]:
            return "footwear", {"footwear": 0.4, "irrelevant": 0.1}

    monkeypatch.setattr(app_module, "RecommenderService", _StubRecommender)
    monkeypatch.setattr(app_module, "VisionEncoder", DummyVision)
    app_module.app.state.pretty_memory = None
    app_module.app.state.text_encoder = None
    with TestClient(app_module.app) as test_client:
        yield test_client
    app_module.app.state.pretty_memory = None
    app_module.app.state.text_encoder = None


def test_approved_crm_status_ranks_above_unreviewed() -> None:
    encoder = MemoryEncoder()
    index = build_index(
        {
            "published_at": "v1",
            "cases": [
                {
                    "case_key": "wati:plain",
                    "role_id": "customer_success",
                    "review_status": "unreviewed",
                    "recognition_text": "store opening hours question",
                    "case_text": "mamilla opening hours from approved maps",
                    "case_summary": "hours",
                },
                {
                    "case_key": "wati:approved",
                    "role_id": "customer_success",
                    "review_status": "approved",
                    "recognition_text": "store opening hours question",
                    "case_text": "mamilla opening hours from approved maps",
                    "case_summary": "hours",
                },
            ],
        },
        encoder,
    )
    query = encoder.encode(["mamilla opening hours from approved maps"], prefix="query")[0]
    results = index.search("case", query, top_k=2)
    assert results[0]["case_key"] == "wati:approved"
    assert results[0]["review_status"] == "approved"
    assert review_weight("approved") > review_weight("draft")
    assert review_weight("draft") > review_weight("unreviewed")


def test_recognize_returns_matching_case(memory_client: TestClient) -> None:
    response = memory_client.post(
        "/memory/recognize",
        headers={"X-API-Key": "test-key"},
        json={"query_text": "I want a return pickup", "role_id": "customer_success", "top_k": 3},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is True
    assert body["results"][0]["case_key"] == "wati:return-1"
    assert body["results"][0]["representation"] == "recognition"


def test_similar_excludes_same_customer(memory_client: TestClient) -> None:
    response = memory_client.post(
        "/memory/similar",
        headers={"X-API-Key": "test-key"},
        json={
            "query_text": "online order return courier pickup",
            "role_id": "customer_success",
            "top_k": 5,
            "exclude_wa_ids": ["972500000001"],
            "exclude_conversation_ids": ["c-live"],
        },
    )
    assert response.status_code == 200
    keys = [row["case_key"] for row in response.json()["results"]]
    assert "wati:same-customer" not in keys
    assert "wati:return-1" in keys


def test_reload_replaces_stale_index(tmp_path: Path) -> None:
    write_snapshot(tmp_path)
    store = PrettyMemoryStore(str(tmp_path), MemoryEncoder())
    first = store.index()
    assert first is not None
    assert len(first.records) == 3
    payload = json.loads((tmp_path / "current.json").read_text(encoding="utf-8"))
    payload["cases"] = [payload["cases"][1]]
    payload["published_at"] = "2026-10-03T01:00:00Z"
    (tmp_path / "current.json").write_text(json.dumps(payload), encoding="utf-8")
    second = store.index()
    assert second is not None
    assert len(second.records) == 1
    assert second.snapshot_version == "2026-10-03T01:00:00Z"


def test_missing_snapshot_is_503(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RECO_API_KEY", "test-key")
    monkeypatch.setenv("PRETTY_MEMORY_SNAPSHOT_DIR", str(tmp_path / "missing"))
    monkeypatch.setattr(app_module, "TextEncoder", MemoryEncoder)

    class _StubRecommender:
        model_version = "the_pretty_model_v1"
        dimension = 64

        @classmethod
        def load(cls, **_kwargs: Any) -> _StubRecommender:
            return cls()

    class DummyVision:
        def encode(self, _image: Any) -> Any:
            return np.zeros(768, dtype=np.float32)

        def encode_batch(self, images: list[Any]) -> Any:
            return np.zeros((len(images), 768), dtype=np.float32)

        def classify_relevance(self, _vector: Any) -> tuple[str, dict[str, float]]:
            return "footwear", {"footwear": 0.4, "irrelevant": 0.1}

    monkeypatch.setattr(app_module, "RecommenderService", _StubRecommender)
    monkeypatch.setattr(app_module, "VisionEncoder", DummyVision)
    app_module.app.state.pretty_memory = None
    app_module.app.state.text_encoder = None
    with TestClient(app_module.app) as client:
        response = client.post(
            "/memory/recognize",
            headers={"X-API-Key": "test-key"},
            json={"query_text": "hello"},
        )
    assert response.status_code == 503
    assert response.json()["detail"] == "snapshot_not_loaded"
