"""Visual match helpers. Encoder and catalog are fakes; no SigLIP download."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from embeddings.catalog_index import CatalogIndex
from embeddings.match_image import match_image, match_result_to_dict
from embeddings.vision_encoder import EMBEDDING_DIMENSION
from evaluation.image_match import CatalogImage


def _unit(values: list[float], dim: int = EMBEDDING_DIMENSION) -> np.ndarray:
    array = np.zeros(dim, dtype=np.float32)
    take = min(len(values), dim)
    array[:take] = np.asarray(values[:take], dtype=np.float32)
    return array / np.linalg.norm(array)


class FakeEncoder:
    embedding_model = "google/siglip-base-patch16-224"
    embedding_dimension = EMBEDDING_DIMENSION

    def encode_batch(self, images):
        vectors = []
        for image in images:
            array = np.asarray(image.convert("RGB"), dtype=np.float32)
            mean = float(array.mean())
            if mean > 180:
                vectors.append(_unit([0.0, 1.0, 0.0]))
            else:
                vectors.append(_unit([1.0, 0.0, 0.0]))
        return np.stack(vectors, axis=0)

    def classify_relevance(self, _vector: np.ndarray) -> tuple[str, dict[str, float]]:
        return "footwear", {"footwear": 0.4, "irrelevant": 0.1}


def test_match_image_picks_isolated_panel() -> None:
    image = Image.new("RGB", (200, 500), (20, 20, 22))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 90, 199, 279), fill=(236, 228, 230))
    catalog = CatalogIndex.from_rows(
        [
            CatalogImage(
                model="WRONG",
                image_type="pers",
                embedding=_unit([1.0, 0.0, 0.0]),
                model_id=1,
                embedding_model="google/siglip-base-patch16-224",
                embedding_dimension=EMBEDDING_DIMENSION,
            ),
            CatalogImage(
                model="52792_006",
                image_type="side",
                embedding=_unit([0.0, 1.0, 0.0]),
                model_id=2,
                embedding_model="google/siglip-base-patch16-224",
                embedding_dimension=EMBEDDING_DIMENSION,
            ),
        ]
    )
    result = match_image(image, FakeEncoder(), catalog, top=10)
    assert result.match is not None
    assert result.match.model == "52792_006"
    assert result.match.best_image_type == "side"
    assert result.shoe_isolated is True
    payload = match_result_to_dict(result, image_size=image.size, image=image, catalog=catalog)
    assert payload["status"] == "needs_verification"
    assert payload["match"] is None
    assert "embedding" not in payload
    assert payload["preprocessing"]["crop"][1] >= 80
    assert payload["candidates"][0]["model"] == "52792_006"
    assert payload["verifier"]["verify_top"] == 10
    assert payload["preprocessing"]["crop_jpeg_base64"]


def test_packshot_keeps_full_frame_when_it_is_the_best_crop() -> None:
    image = Image.new("RGB", (200, 200), (236, 228, 230))
    catalog = CatalogIndex.from_rows(
        [
            CatalogImage(
                model="40724_001",
                image_type="main",
                embedding=_unit([0.0, 1.0, 0.0]),
                model_id=3,
                embedding_model="google/siglip-base-patch16-224",
                embedding_dimension=EMBEDDING_DIMENSION,
            )
        ]
    )
    result = match_image(image, FakeEncoder(), catalog, top=10)
    assert result.match is not None
    assert result.match.model == "40724_001"
    assert result.shoe_isolated is False
    assert result.crop.reason == "full"


def test_match_image_skips_catalog_when_no_crop_is_footwear() -> None:
    class JunkEncoder(FakeEncoder):
        def classify_relevance(self, _vector: np.ndarray) -> tuple[str, dict[str, float]]:
            return "irrelevant", {"footwear": 0.1, "irrelevant": 0.4}

    image = Image.new("RGB", (200, 200), (236, 228, 230))
    catalog = CatalogIndex.from_rows(
        [
            CatalogImage(
                model="51604_004",
                image_type="main",
                embedding=_unit([0.0, 1.0, 0.0]),
                model_id=4,
                embedding_model="google/siglip-base-patch16-224",
                embedding_dimension=EMBEDDING_DIMENSION,
            )
        ]
    )
    result = match_image(image, JunkEncoder(), catalog, top=10)
    assert result.match is None
    assert result.candidates == []
    assert result.relevance == "irrelevant"
    assert result.image_kind == "garbage"
    payload = match_result_to_dict(result, image_size=image.size)
    assert payload["status"] == "garbage"
    assert payload["image_kind"] == "garbage"
    assert payload["candidates"] == []


def test_match_image_routes_order_without_catalog_search() -> None:
    class OrderEncoder(FakeEncoder):
        def classify_relevance(self, _vector: np.ndarray) -> tuple[str, dict[str, float]]:
            return "irrelevant", {"footwear": 0.05, "irrelevant": 0.4}

        def classify_scene(self, _vector: np.ndarray) -> tuple[str, dict[str, float]]:
            return "order", {"footwear": 0.05, "irrelevant": 0.2, "order": 0.5, "garbage": 0.2}

    image = Image.new("RGB", (200, 200), (236, 228, 230))
    catalog = CatalogIndex.from_rows(
        [
            CatalogImage(
                model="51604_004",
                image_type="main",
                embedding=_unit([0.0, 1.0, 0.0]),
                model_id=4,
                embedding_model="google/siglip-base-patch16-224",
                embedding_dimension=EMBEDDING_DIMENSION,
            )
        ]
    )
    result = match_image(
        image,
        OrderEncoder(),
        catalog,
        top=10,
        extract_order_fn=lambda _img: {
            "order_number": "87610",
            "model": "53698_003",
            "size": "38.5",
        },
    )
    assert result.match is None
    assert result.candidates == []
    assert result.image_kind == "order"
    assert result.order == {
        "order_number": "87610",
        "model": "53698_003",
        "size": "38.5",
    }
    payload = match_result_to_dict(result, image_size=image.size)
    assert payload["status"] == "order"
    assert payload["verifier"] is None
    assert payload["order"]["order_number"] == "87610"
