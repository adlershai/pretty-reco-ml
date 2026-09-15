"""Query-image helpers. Does not load SigLIP."""

from __future__ import annotations

import base64

import pytest

from embeddings.query_image import QueryImageError, decode_image_base64


def test_decode_image_base64_plain() -> None:
    raw = b"hello-image"
    encoded = base64.b64encode(raw).decode("ascii")
    assert decode_image_base64(encoded) == raw


def test_decode_image_base64_data_url() -> None:
    raw = b"jpeg-bytes"
    encoded = base64.b64encode(raw).decode("ascii")
    assert decode_image_base64(f"data:image/jpeg;base64,{encoded}") == raw


def test_decode_image_base64_empty() -> None:
    with pytest.raises(QueryImageError):
        decode_image_base64("   ")
