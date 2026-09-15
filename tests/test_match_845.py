"""Known WATI screenshot from GitHub issue #1. Needs catalog cache + SigLIP."""

from __future__ import annotations

import pytest
import requests

from embeddings.catalog_index import CACHE_DIR, load_catalog_index
from embeddings.match_image import match_image
from embeddings.vision_encoder import VisionEncoder
from embeddings.worker import decode_image, download_image_bytes

QUERY_URL = "https://pb-il-app-images-1.s3.eu-central-1.amazonaws.com/ops/wati/6aa3e530791a84d02c58c38e/845.jpg"
EXPECTED = "52792_006"
CACHE_VECTORS = CACHE_DIR / "catalog_image_index.npz"


@pytest.mark.live
def test_845_screenshot_ranks_52792_006_after_isolation() -> None:
    if not CACHE_VECTORS.is_file():
        pytest.skip("catalog image cache is not present")
    catalog = load_catalog_index()
    with requests.Session() as session:
        image = decode_image(download_image_bytes(QUERY_URL, session))
    encoder = VisionEncoder()
    result = match_image(image, encoder, catalog, top=10)
    assert result.match is not None
    assert result.match.model == EXPECTED
    assert result.shoe_isolated is True
    assert result.candidates[0].model == EXPECTED
    assert result.match.best_image_type == "side"
