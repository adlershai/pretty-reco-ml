"""Event 736 is a clothing ad, not a shoe. Permanent garbage regression (issue #4)."""

from __future__ import annotations

import pytest
import requests

from embeddings.catalog_index import CatalogIndex
from embeddings.match_image import match_image
from embeddings.vision_encoder import VisionEncoder
from embeddings.worker import decode_image, download_image_bytes

QUERY_URL = "https://media.adler.co.il/ops/wati/6a9ff10bfa14ab98ff00a5f9/736.jpg"


@pytest.mark.live
def test_736_clothing_ad_returns_garbage_no_sku() -> None:
    catalog = CatalogIndex.from_rows([])
    with requests.Session() as session:
        image = decode_image(download_image_bytes(QUERY_URL, session))
    encoder = VisionEncoder()
    result = match_image(image, encoder, catalog, top=10)
    assert result.match is None
    assert result.candidates == []
    assert result.relevance == "irrelevant"
    assert result.image_kind == "garbage"
