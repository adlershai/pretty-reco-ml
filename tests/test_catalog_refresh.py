"""Authenticated catalog hot-refresh. Loader is injected; no DB or SigLIP."""

from __future__ import annotations

import threading
from typing import Any

import numpy as np
from fastapi.testclient import TestClient

import app as app_module
from embeddings.catalog_index import CatalogIndex
from embeddings.catalog_refresh import CatalogRefreshService, validate_catalog
from embeddings.vision_encoder import EMBEDDING_DIMENSION
from evaluation.image_match import CatalogImage


def _vector(seed: float) -> np.ndarray:
    vector = np.zeros(EMBEDDING_DIMENSION, dtype=np.float32)
    vector[0] = float(seed)
    vector[1] = 1.0 - float(seed)
    return vector


def _row(model: str, seed: float) -> CatalogImage:
    return CatalogImage(
        model=model,
        image_type="main",
        embedding=_vector(seed),
        model_id=1,
        row_id=1,
        embedding_model="dummy",
        embedding_dimension=EMBEDDING_DIMENSION,
    )


def _index(*models: str) -> CatalogIndex:
    rows = [_row(model, (index + 1) / (len(models) + 1)) for index, model in enumerate(models)]
    return CatalogIndex.from_rows(rows, embedding_model="dummy")


def test_validate_catalog_rejects_empty() -> None:
    with pytest.raises(RuntimeError, match="empty"):
        validate_catalog(CatalogIndex.from_rows([], embedding_model="dummy"))


def test_refresh_swaps_live_catalog_and_reports_counts() -> None:
    current: dict[str, CatalogIndex | None] = {"value": _index("OLD")}
    service = CatalogRefreshService(
        loader=lambda: _index("NEW_A", "NEW_B"),
        get_current=lambda: current["value"],
        activate=lambda index: current.__setitem__("value", index),
    )
    result = service.refresh()
    assert result["ok"] is True
    assert result["status"] == "ready"
    assert result["coalesced"] is False
    assert result["kept_previous"] is False
    assert result["job_id"]
    assert result["image_count"] == 2
    assert result["model_count"] == 2
    assert result["built_at"]
    assert current["value"] is not None
    assert current["value"].model_count == 2
    assert {row.model for row in current["value"].rows} == {"NEW_A", "NEW_B"}


def test_failed_refresh_keeps_previous_catalog() -> None:
    previous = _index("KEEP")
    current: dict[str, CatalogIndex | None] = {"value": previous}

    def boom() -> CatalogIndex:
        raise RuntimeError("db down")

    service = CatalogRefreshService(
        loader=boom,
        get_current=lambda: current["value"],
        activate=lambda index: current.__setitem__("value", index),
    )
    result = service.refresh()
    assert result["ok"] is False
    assert result["status"] == "failed"
    assert result["kept_previous"] is True
    assert result["image_count"] == 1
    assert result["model_count"] == 1
    assert current["value"] is previous


def test_overlapping_refresh_coalesces_to_one_rebuild() -> None:
    started = threading.Event()
    release = threading.Event()
    calls = {"n": 0}
    current: dict[str, CatalogIndex | None] = {"value": _index("OLD")}

    def loader() -> CatalogIndex:
        calls["n"] += 1
        started.set()
        assert release.wait(timeout=2)
        return _index("LIVE")

    service = CatalogRefreshService(
        loader=loader,
        get_current=lambda: current["value"],
        activate=lambda index: current.__setitem__("value", index),
    )
    results: list[dict[str, Any]] = []

    def run() -> None:
        results.append(service.refresh())

    first = threading.Thread(target=run)
    second = threading.Thread(target=run)
    first.start()
    assert started.wait(timeout=2)
    second.start()
    release.set()
    first.join(timeout=2)
    second.join(timeout=2)
    assert calls["n"] == 1
    assert len(results) == 2
    assert results[0]["job_id"] == results[1]["job_id"]
    assert results[0]["status"] == "ready"
    assert results[1]["status"] == "ready"
    assert True in {row["coalesced"] for row in results}
    assert False in {row["coalesced"] for row in results}


def test_catalog_refresh_endpoint_requires_key(client: TestClient) -> None:
    response = client.post("/catalog/refresh")
    assert response.status_code == 401
    status = client.get("/catalog/refresh")
    assert status.status_code == 401


def test_catalog_refresh_endpoint_swaps_app_catalog(client: TestClient) -> None:
    previous = _index("OLD")
    client.app.state.catalog = previous
    client.app.state.catalog_refresh = CatalogRefreshService(
        loader=lambda: _index("FRESH"),
        get_current=lambda: getattr(client.app.state, "catalog", None),
        activate=lambda index: setattr(client.app.state, "catalog", index),
    )
    response = client.post("/catalog/refresh", headers={"X-API-Key": "test-key"})
    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is True
    assert body["status"] == "ready"
    assert body["model_count"] == 1
    assert {row.model for row in client.app.state.catalog.rows} == {"FRESH"}
    status = client.get("/catalog/refresh", headers={"X-API-Key": "test-key"})
    assert status.status_code == 200
    assert status.json()["job_id"] == body["job_id"]
    assert status.json()["status"] == "ready"


def test_catalog_refresh_endpoint_failure_keeps_searchable_catalog(client: TestClient) -> None:
    previous = _index("KEEP")
    client.app.state.catalog = previous

    def boom() -> CatalogIndex:
        raise RuntimeError("catalog is empty")

    client.app.state.catalog_refresh = CatalogRefreshService(
        loader=boom,
        get_current=lambda: getattr(client.app.state, "catalog", None),
        activate=lambda index: setattr(client.app.state, "catalog", index),
    )
    response = client.post("/catalog/refresh", headers={"X-API-Key": "test-key"})
    assert response.status_code == 503
    body = response.json()
    assert body["ok"] is False
    assert body["kept_previous"] is True
    assert {row.model for row in client.app.state.catalog.rows} == {"KEEP"}
