from __future__ import annotations

import numpy as np
import pytest

from embeddings.text_similarity import cosine_similarity, rank_candidates


def test_cosine_similarity_same_vector_is_one() -> None:
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)


def test_cosine_similarity_orthogonal_is_zero() -> None:
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_cosine_similarity_rejects_dimension_mismatch() -> None:
    with pytest.raises(ValueError, match="VECTOR_DIMENSION_MISMATCH"):
        cosine_similarity([1.0, 0.0], [1.0])


def test_rank_candidates_returns_top_k() -> None:
    ranked = rank_candidates(
        [1.0, 0.0],
        [
            {"id": "a", "embedding": [0.2, 0.8]},
            {"id": "b", "embedding": [1.0, 0.0]},
            {"id": "c", "embedding": [0.7, 0.3]},
        ],
        top=2,
    )
    assert [row["id"] for row in ranked] == ["b", "c"]
    assert np.isfinite(ranked[0]["score"])
