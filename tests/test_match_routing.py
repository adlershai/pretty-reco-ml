"""Issue #4 sequential router on real WATI images. Needs SigLIP + catalog cache."""

from __future__ import annotations

from pathlib import Path

import pytest
import requests
from PIL import Image

from embeddings.catalog_index import CACHE_DIR, load_catalog_index
from embeddings.match_image import match_image
from embeddings.vision_encoder import VisionEncoder
from embeddings.worker import decode_image, download_image_bytes

FIXTURES = Path(__file__).resolve().parent / "fixtures"
CACHE_VECTORS = CACHE_DIR / "catalog_image_index.npz"
EVENT_736_URL = "https://media.adler.co.il/ops/wati/6a9ff10bfa14ab98ff00a5f9/736.jpg"

SHOE_FIXTURES = (
    ("1070", FIXTURES / "wati_1070.jpg", "53235_007", True),
    ("1072", FIXTURES / "wati_1072.jpg", "51604_A", False),
    ("1068", FIXTURES / "wati_1068.jpg", "51604_A", False),
    ("1071", FIXTURES / "wati_1071.jpg", "52207_A", True),
)


@pytest.fixture(scope="module")
def encoder() -> VisionEncoder:
    return VisionEncoder()


@pytest.fixture(scope="module")
def catalog():
    if not CACHE_VECTORS.is_file():
        pytest.skip("catalog image cache is not present")
    return load_catalog_index()


@pytest.mark.live
@pytest.mark.parametrize("event_id,path,expected_model,require_model", SHOE_FIXTURES)
def test_pdp_screenshot_routes_shoe_not_order(
    event_id, path, expected_model, require_model, encoder, catalog
) -> None:
    if not path.is_file():
        pytest.skip(f"{path.name} fixture is missing")
    image = Image.open(path).convert("RGB")
    result = match_image(image, encoder, catalog, top=10)
    assert result.image_kind == "shoe", (
        f"{event_id} routed {result.image_kind} scores={result.scores}"
    )
    assert result.order is None
    models = [hit.model for hit in result.candidates]
    if require_model:
        assert expected_model in models, (
            f"{event_id} expected {expected_model} in top-10 {models} scores={result.scores}"
        )


@pytest.mark.live
def test_753_order_screenshot_still_order(encoder, catalog) -> None:
    path = FIXTURES / "order_87610.jpg"
    if not path.is_file():
        pytest.skip("order_87610.jpg fixture is missing")
    image = Image.open(path).convert("RGB")
    result = match_image(image, encoder, catalog, top=10)
    assert result.image_kind == "order", f"753 routed {result.image_kind} scores={result.scores}"
    assert result.match is None
    assert result.candidates == []


@pytest.mark.live
def test_1057_on_the_way_is_delivery_notice(encoder, catalog) -> None:
    path = FIXTURES / "wati_1057.jpg"
    if not path.is_file():
        pytest.skip("wati_1057.jpg fixture is missing")
    image = Image.open(path).convert("RGB")
    result = match_image(image, encoder, catalog, top=10)
    assert result.image_kind == "delivery_notice", (
        f"1057 routed {result.image_kind} scores={result.scores}"
    )
    assert result.match is None
    assert result.candidates == []
    assert result.order is None


@pytest.mark.live
def test_1162_order_confirmation_is_order_not_shoe(encoder, catalog) -> None:
    path = FIXTURES / "wati_1162.jpg"
    if not path.is_file():
        pytest.skip("wati_1162.jpg fixture is missing")
    image = Image.open(path).convert("RGB")
    result = match_image(image, encoder, catalog, top=10)
    assert result.image_kind == "order", (
        f"1162 routed {result.image_kind} scores={result.scores} order={result.order}"
    )
    assert result.match is None
    assert result.candidates == []
    assert result.order is not None
    number = str(result.order["order_number"] or "")
    assert number.upper().startswith("CS"), number


@pytest.mark.live
def test_736_clothing_ad_still_garbage(encoder, catalog) -> None:
    with requests.Session() as session:
        image = decode_image(download_image_bytes(EVENT_736_URL, session))
    result = match_image(image, encoder, catalog, top=10)
    assert result.image_kind == "garbage", f"736 routed {result.image_kind} scores={result.scores}"
    assert result.match is None
    assert result.candidates == []


@pytest.mark.live
def test_routing_confusion_matrix(encoder, catalog) -> None:
    cases: list[tuple[str, str, object]] = [
        ("1070", "shoe", SHOE_FIXTURES[0][1]),
        ("1072", "shoe", SHOE_FIXTURES[1][1]),
        ("1068", "shoe", SHOE_FIXTURES[2][1]),
        ("1071", "shoe", SHOE_FIXTURES[3][1]),
        ("753", "order", FIXTURES / "order_87610.jpg"),
        ("1162", "order", FIXTURES / "wati_1162.jpg"),
        ("1057", "delivery_notice", FIXTURES / "wati_1057.jpg"),
    ]
    with requests.Session() as session:
        image_736 = decode_image(download_image_bytes(EVENT_736_URL, session))
    predicted: dict[str, str] = {}
    details: list[str] = []
    for event_id, expected, path in cases:
        image = Image.open(path).convert("RGB")
        result = match_image(image, encoder, catalog, top=10)
        predicted[event_id] = result.image_kind
        details.append(
            f"{event_id} expected={expected} got={result.image_kind} scores={result.scores}"
        )
        assert result.image_kind == expected, details[-1]
    result_736 = match_image(image_736, encoder, catalog, top=10)
    predicted["736"] = result_736.image_kind
    details.append(f"736 expected=garbage got={result_736.image_kind} scores={result_736.scores}")
    assert result_736.image_kind == "garbage", details[-1]
    kinds = ("shoe", "order", "delivery_notice", "garbage")
    expected_map = {
        "1070": "shoe",
        "1072": "shoe",
        "1068": "shoe",
        "1071": "shoe",
        "753": "order",
        "1162": "order",
        "1057": "delivery_notice",
        "736": "garbage",
    }
    matrix = {row: {col: 0 for col in kinds} for row in kinds}
    for event_id, expected in expected_map.items():
        matrix[expected][predicted[event_id]] += 1
    print("routing confusion matrix (rows=expected, cols=predicted)")
    print("         " + " ".join(f"{kind:16}" for kind in kinds))
    for row in kinds:
        cells = " ".join(f"{matrix[row][col]:16}" for col in kinds)
        print(f"{row:16} {cells}")
    for line in details:
        print(line)
    assert matrix["shoe"]["order"] == 0
    assert matrix["shoe"]["delivery_notice"] == 0
    assert matrix["order"]["shoe"] == 0
    assert matrix["delivery_notice"]["order"] == 0
    assert matrix["garbage"]["shoe"] == 0
