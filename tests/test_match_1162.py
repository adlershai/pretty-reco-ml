"""WATI 1162 is an order confirmation with a recognizable product thumbnail (issue #7)."""

from __future__ import annotations

from pathlib import Path
import re

import pytest
from PIL import Image

from embeddings.catalog_index import CatalogIndex
from embeddings.match_image import match_image
from embeddings.order_extract import ocr_image_text, parse_order_fields
from embeddings.vision_encoder import VisionEncoder

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "wati_1162.jpg"


@pytest.mark.live
def test_1162_order_confirmation_is_order_not_shoe() -> None:
    if not FIXTURE.is_file():
        pytest.skip("wati_1162.jpg fixture is missing")
    catalog = CatalogIndex.from_rows([])
    image = Image.open(FIXTURE).convert("RGB")
    encoder = VisionEncoder()
    result = match_image(image, encoder, catalog, top=10)
    assert result.match is None
    assert result.candidates == []
    assert result.image_kind == "order"
    assert result.order is not None
    number = str(result.order["order_number"] or "")
    assert re.fullmatch(r"CS\d{8,12}", number, re.I), number
    text = ocr_image_text(image)
    if text.strip():
        fields = parse_order_fields(text)
        parsed = str(fields["order_number"] or "")
        assert re.fullmatch(r"CS\d{8,12}", parsed, re.I), parsed
