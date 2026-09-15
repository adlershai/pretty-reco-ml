"""Decode a caller-supplied image and return embedding + relevance.

Callers send bytes (base64 in JSON). This module does not download URLs.
"""

from __future__ import annotations

import base64
import binascii
from typing import Any

from embeddings.vision_encoder import VisionEncoder
from embeddings.worker import decode_image


class QueryImageError(ValueError):
    """Invalid query image payload."""


def decode_image_base64(raw: str) -> bytes:
    text = str(raw or "").strip()
    if not text:
        raise QueryImageError("INVALID_IMAGE")
    if "," in text and text.lower().startswith("data:"):
        text = text.split(",", 1)[1]
    try:
        data = base64.b64decode(text, validate=False)
    except (binascii.Error, ValueError) as exc:
        raise QueryImageError("INVALID_IMAGE") from exc
    if not data:
        raise QueryImageError("INVALID_IMAGE")
    return data


def run_query(image_bytes: bytes, encoder: VisionEncoder) -> dict[str, Any]:
    image = decode_image(image_bytes)
    vector = encoder.encode(image)
    relevance, scores = encoder.classify_relevance(vector)
    return {
        "embedding": vector.astype(float).tolist(),
        "embedding_model": encoder.embedding_model,
        "embedding_dimension": int(vector.shape[0]),
        "relevance": relevance,
        "scores": scores,
    }
