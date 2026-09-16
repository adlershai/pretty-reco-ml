"""Event 736 is a clothing ad, not a shoe. Needs catalog cache + SigLIP."""

from __future__ import annotations

import pytest
import requests

from embeddings.catalog_index import CACHE_DIR, load_catalog_index
from embeddings.match_image import match_image
from embeddings.vision_encoder import VisionEncoder
from embeddings.worker import decode_image, download_image_bytes

QUERY_URL = "https://media.adler.co.il/ops/wati/6a9ff10bfa14ab98ff00a5f9/736.jpg"
CACHE_VECTORS = CACHE_DIR / "catalog_image_index.npz"


@pytest.mark.live
def test_736_clothing_ad_returns_no_sku() -> None:
    if not CACHE_VECTORS.is_file():
        pytest.skip("catalog image cache is not present")
    catalog = load_catalog_index()
    with requests.Session() as session:
        image = decode_image(download_image_bytes(QUERY_URL, session))
    encoder = VisionEncoder()
    result = match_image(image, encoder, catalog, top=10)
    assert result.match is None
    assert result.candidates == []
    assert result.relevance == "irrelevant"
