"""Zero-shot footwear vs junk scoring on SigLIP image/text vectors.

Prompts stay generic (footwear vs not a product photo). Catalog matching
decides the model; this gate only skips obvious junk before nearest-neighbor.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

FOOTWEAR_PROMPTS: tuple[str, ...] = (
    "a photograph of footwear",
    "a photograph of shoes",
    "a product photo of a shoe",
)

JUNK_PROMPTS: tuple[str, ...] = (
    "a phone screenshot with text and icons",
    "a selfie of a person's face",
    "a paper document or receipt",
    "a meme or cartoon drawing",
    "an indoor photo with no product",
)

RELEVANCE_FOOTWEAR = "footwear"
RELEVANCE_IRRELEVANT = "irrelevant"


def classify_relevance(
    image_vector: np.ndarray,
    footwear_vectors: np.ndarray,
    junk_vectors: np.ndarray,
) -> tuple[str, dict[str, float]]:
    """Return (relevance, scores) from L2-normalized vectors (cosine = dot)."""
    image = np.asarray(image_vector, dtype=np.float32).reshape(-1)
    footwear = np.asarray(footwear_vectors, dtype=np.float32)
    junk = np.asarray(junk_vectors, dtype=np.float32)
    if footwear.ndim != 2 or junk.ndim != 2:
        raise ValueError("prompt matrices must be 2-D")
    if image.shape[0] != footwear.shape[1] or image.shape[0] != junk.shape[1]:
        raise ValueError("image and prompt embedding dimensions must match")

    footwear_score = float(np.max(footwear @ image))
    junk_score = float(np.max(junk @ image))
    relevance = (
        RELEVANCE_FOOTWEAR if footwear_score >= junk_score else RELEVANCE_IRRELEVANT
    )
    return relevance, {
        RELEVANCE_FOOTWEAR: footwear_score,
        RELEVANCE_IRRELEVANT: junk_score,
    }


def all_prompts() -> Sequence[str]:
    return (*FOOTWEAR_PROMPTS, *JUNK_PROMPTS)
