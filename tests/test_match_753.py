"""WATI event 753 is the #87610 order screenshot (issue #4). Real image, not generated."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from embeddings.catalog_index import CatalogIndex
from embeddings.match_image import match_image
from embeddings.order_extract import ocr_image_text, parse_order_fields
from embeddings.vision_encoder import VisionEncoder

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "order_87610.jpg"


@pytest.mark.live
def test_753_order_screenshot_is_order_not_shoe() -> None:
    if not FIXTURE.is_file():
        pytest.skip("order_87610.jpg fixture is missing")
    catalog = CatalogIndex.from_rows([])
    image = Image.open(FIXTURE).convert("RGB")
    encoder = VisionEncoder()
    result = match_image(image, encoder, catalog, top=10)
    assert result.match is None
    assert result.candidates == []
    assert result.image_kind == "order"
    assert result.relevance == "irrelevant"
    text = ocr_image_text(image)
    if text.strip():
        fields = parse_order_fields(text)
        assert fields["order_number"] == "87610"
        assert fields["model"] == "53698_003"
        assert fields["size"] == "38.5"
        assert result.order is not None
        assert result.order["order_number"] == "87610"
        assert result.order["model"] == "53698_003"
        assert result.order["size"] == "38.5"
