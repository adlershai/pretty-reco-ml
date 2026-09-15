"""Catalog hash audit. Downloads are stubbed."""

from __future__ import annotations

import numpy as np

from embeddings.catalog_integrity import audit_row
from embeddings.vision_encoder import EMBEDDING_DIMENSION
from evaluation.image_match import CatalogImage


def _row(image_hash: str) -> CatalogImage:
    vector = np.zeros(EMBEDDING_DIMENSION, dtype=np.float32)
    vector[0] = 1.0
    return CatalogImage(
        model="52160_001",
        image_type="pers",
        embedding=vector,
        model_id=1,
        row_id=9,
        embedding_model="google/siglip-base-patch16-224",
        embedding_dimension=EMBEDDING_DIMENSION,
        image_hash=image_hash,
        source_url="https://example.com/pers.jpg",
    )


def test_audit_marks_changed_cdn_hash(monkeypatch) -> None:
    monkeypatch.setattr(
        "embeddings.catalog_integrity.download_image_bytes",
        lambda _url, _session: b"fresh",
    )
    monkeypatch.setattr("embeddings.catalog_integrity.sha256_hex", lambda _data: "b" * 64)
    item = audit_row(_row("a" * 64), object())
    assert item.stale is True
    assert item.current_hash == "b" * 64
    assert item.error is None


def test_audit_ok_when_hash_matches(monkeypatch) -> None:
    monkeypatch.setattr(
        "embeddings.catalog_integrity.download_image_bytes",
        lambda _url, _session: b"same",
    )
    monkeypatch.setattr("embeddings.catalog_integrity.sha256_hex", lambda _data: "a" * 64)
    item = audit_row(_row("a" * 64), object())
    assert item.stale is False
