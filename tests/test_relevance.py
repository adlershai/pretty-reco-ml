"""Relevance gate scoring does not load SigLIP."""

from __future__ import annotations

import numpy as np

from embeddings.relevance import (
    RELEVANCE_FOOTWEAR,
    RELEVANCE_IRRELEVANT,
    classify_relevance,
)


def test_footwear_wins_when_closer_to_footwear_prompts() -> None:
    image = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    footwear = np.array([[0.9, 0.1, 0.0], [0.8, 0.2, 0.0]], dtype=np.float32)
    footwear = footwear / np.linalg.norm(footwear, axis=1, keepdims=True)
    junk = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    relevance, scores = classify_relevance(image, footwear, junk)
    assert relevance == RELEVANCE_FOOTWEAR
    assert scores[RELEVANCE_FOOTWEAR] > scores[RELEVANCE_IRRELEVANT]


def test_junk_wins_when_closer_to_junk_prompts() -> None:
    image = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    footwear = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    junk = np.array([[0.0, 1.0, 0.0]], dtype=np.float32)
    relevance, scores = classify_relevance(image, footwear, junk)
    assert relevance == RELEVANCE_IRRELEVANT
    assert scores[RELEVANCE_IRRELEVANT] > scores[RELEVANCE_FOOTWEAR]
