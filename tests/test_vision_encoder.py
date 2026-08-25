"""Live checks against the known Pretty Ballerinas model 40724_001."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import requests
from PIL import Image

from embeddings.vision_encoder import EMBEDDING_DIMENSION, SIGLIP_MODEL_ID, VisionEncoder
from embeddings.worker import (
    IMAGE_TYPES,
    decode_image,
    download_image_bytes,
    run,
    sha256_hex,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_PAYLOAD = REPO_ROOT / "examples" / "40724_001.json"
MAIN_URL = "https://media.adler.co.il/app/products/40724_001.jpg"


@pytest.fixture(scope="module")
def encoder() -> VisionEncoder:
    return VisionEncoder()


@pytest.fixture(scope="module")
def main_image_bytes() -> bytes:
    with requests.Session() as session:
        return download_image_bytes(MAIN_URL, session)


def test_image_downloads_successfully(main_image_bytes: bytes) -> None:
    assert len(main_image_bytes) > 0


def test_image_hash_is_sha256(main_image_bytes: bytes) -> None:
    digest = sha256_hex(main_image_bytes)
    assert len(digest) == 64
    int(digest, 16)


def test_embedding_properties(encoder: VisionEncoder, main_image_bytes: bytes) -> None:
    image = decode_image(main_image_bytes)
    vector = encoder.encode(image)

    assert encoder.embedding_model == SIGLIP_MODEL_ID
    assert encoder.embedding_dimension == EMBEDDING_DIMENSION
    assert vector.shape == (EMBEDDING_DIMENSION,)
    assert vector.dtype == np.float32
    assert np.isfinite(vector).all()
    assert vector.tolist()
    np.testing.assert_allclose(np.linalg.norm(vector), 1.0, rtol=0, atol=1e-5)


def test_repeated_encoding_is_equivalent(encoder: VisionEncoder, main_image_bytes: bytes) -> None:
    image = decode_image(main_image_bytes)
    first = encoder.encode(image)
    second = encoder.encode(image)
    np.testing.assert_allclose(first, second, rtol=0, atol=1e-5)


def test_batch_keeps_images_independent(encoder: VisionEncoder, main_image_bytes: bytes) -> None:
    packshot = decode_image(main_image_bytes)
    other = Image.new("RGB", packshot.size, (220, 20, 60))
    batch = encoder.encode_batch([packshot, other])
    assert batch.shape == (2, EMBEDDING_DIMENSION)
    assert not np.allclose(batch[0], batch[1], atol=1e-3)


def test_worker_encodes_available_views(encoder: VisionEncoder) -> None:
    payload = json.loads(EXAMPLE_PAYLOAD.read_text(encoding="utf-8"))
    output = run(payload, encoder, batch_size=8)

    assert set(output) == {"results", "errors"}
    types = {item["image_type"] for item in output["results"]}
    assert types == set(IMAGE_TYPES)
    assert output["errors"] == []

    for item in output["results"]:
        assert item["model_id"] == 123
        assert item["model"] == "40724_001"
        assert item["embedding_model"] == SIGLIP_MODEL_ID
        assert item["embedding_dimension"] == EMBEDDING_DIMENSION
        assert len(item["embedding"]) == EMBEDDING_DIMENSION
        assert all(isinstance(value, float) for value in item["embedding"])
        assert len(item["image_hash"]) == 64
