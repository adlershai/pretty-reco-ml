"""Full-image scene routing does not load SigLIP."""

from __future__ import annotations

import numpy as np

from embeddings.relevance import (
    IMAGE_KIND_GARBAGE,
    IMAGE_KIND_ORDER,
    IMAGE_KIND_SHOE,
    classify_scene,
)


def test_order_wins_on_full_image_even_if_footwear_is_close() -> None:
    image = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    footwear = np.array([[0.2, 0.0, 0.8]], dtype=np.float32)
    footwear = footwear / np.linalg.norm(footwear, axis=1, keepdims=True)
    garbage = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    order = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)
    kind, scores = classify_scene(image, footwear, garbage, order)
    assert kind == IMAGE_KIND_ORDER
    assert scores["order"] > scores["footwear"]


def test_garbage_wins_for_clothing() -> None:
    image = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    footwear = np.array([[0.0, 1.0, 0.0]], dtype=np.float32)
    garbage = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    order = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)
    kind, _scores = classify_scene(image, footwear, garbage, order)
    assert kind == IMAGE_KIND_GARBAGE


def test_shoe_wins_when_closer_to_footwear() -> None:
    image = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    footwear = np.array([[0.0, 1.0, 0.0]], dtype=np.float32)
    garbage = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    order = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)
    kind, _scores = classify_scene(image, footwear, garbage, order)
    assert kind == IMAGE_KIND_SHOE
