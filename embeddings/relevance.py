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
    "a clothing or fashion advertisement with no shoes",
    "a photo of a dress, shirt, coat, or pants",
    "a website page of apparel, not a close-up of a shoe",
)

# Full-image scene prompts. Receipt/screenshot junk stays in JUNK_PROMPTS so the
# shoe crop gate still skips documents; order vs garbage must not use those or
# a Thank-you-for-your-order page loses to "phone screenshot".
ORDER_PROMPTS: tuple[str, ...] = (
    "a screenshot of an online shoe store order confirmation with an order number",
    "a thank you for your order email showing a product table and price",
    "a Pretty Ballerinas order summary with billing and shipping address",
)

GARBAGE_PROMPTS: tuple[str, ...] = (
    "a selfie of a person's face",
    "a meme or cartoon drawing",
    "an indoor photo with no product",
    "a clothing or fashion advertisement with no shoes",
    "a photo of a dress, shirt, coat, or pants",
    "a bridal or evening gown product page",
    "a website page of apparel, not a close-up of a shoe",
)

RELEVANCE_FOOTWEAR = "footwear"
RELEVANCE_IRRELEVANT = "irrelevant"

IMAGE_KIND_SHOE = "shoe"
IMAGE_KIND_ORDER = "order"
IMAGE_KIND_GARBAGE = "garbage"


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


def classify_scene(
    image_vector: np.ndarray,
    footwear_vectors: np.ndarray,
    garbage_vectors: np.ndarray,
    order_vectors: np.ndarray,
) -> tuple[str, dict[str, float]]:
    """Primary content of the full frame: shoe, order screenshot, or garbage."""
    image = np.asarray(image_vector, dtype=np.float32).reshape(-1)
    footwear = np.asarray(footwear_vectors, dtype=np.float32)
    garbage = np.asarray(garbage_vectors, dtype=np.float32)
    order = np.asarray(order_vectors, dtype=np.float32)
    if footwear.ndim != 2 or garbage.ndim != 2 or order.ndim != 2:
        raise ValueError("prompt matrices must be 2-D")
    dim = image.shape[0]
    if footwear.shape[1] != dim or garbage.shape[1] != dim or order.shape[1] != dim:
        raise ValueError("image and prompt embedding dimensions must match")

    footwear_score = float(np.max(footwear @ image))
    garbage_score = float(np.max(garbage @ image))
    order_score = float(np.max(order @ image))
    scores = {
        RELEVANCE_FOOTWEAR: footwear_score,
        RELEVANCE_IRRELEVANT: garbage_score,
        IMAGE_KIND_ORDER: order_score,
        IMAGE_KIND_GARBAGE: garbage_score,
    }
    if order_score >= footwear_score and order_score >= garbage_score:
        return IMAGE_KIND_ORDER, scores
    if footwear_score >= garbage_score:
        return IMAGE_KIND_SHOE, scores
    return IMAGE_KIND_GARBAGE, scores


def classify_non_shoe(
    image_vector: np.ndarray,
    garbage_vectors: np.ndarray,
    order_vectors: np.ndarray,
) -> tuple[str, dict[str, float]]:
    """Order vs garbage after Step 1 already rejected an identifiable shoe."""
    image = np.asarray(image_vector, dtype=np.float32).reshape(-1)
    garbage = np.asarray(garbage_vectors, dtype=np.float32)
    order = np.asarray(order_vectors, dtype=np.float32)
    if garbage.ndim != 2 or order.ndim != 2:
        raise ValueError("prompt matrices must be 2-D")
    if image.shape[0] != garbage.shape[1] or image.shape[0] != order.shape[1]:
        raise ValueError("image and prompt embedding dimensions must match")

    garbage_score = float(np.max(garbage @ image))
    order_score = float(np.max(order @ image))
    scores = {
        IMAGE_KIND_ORDER: order_score,
        IMAGE_KIND_GARBAGE: garbage_score,
        RELEVANCE_IRRELEVANT: garbage_score,
    }
    if order_score >= garbage_score:
        return IMAGE_KIND_ORDER, scores
    return IMAGE_KIND_GARBAGE, scores


def all_prompts() -> Sequence[str]:
    return (*FOOTWEAR_PROMPTS, *JUNK_PROMPTS, *ORDER_PROMPTS, *GARBAGE_PROMPTS)
