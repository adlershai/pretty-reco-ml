"""Image-level catalog matching diagnostics. No SigLIP, no network."""

from __future__ import annotations

import numpy as np

from evaluation.image_match import (
    CatalogImage,
    ImageHit,
    ModelHit,
    aggregate_models_max,
    catalog_from_model_view,
    classify_failure,
    find_image_hit,
    inspect_catalog_integrity,
    inspect_vector,
    rank_catalog_images,
)


def _unit(values: list[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    return array / np.linalg.norm(array)


def _row(model: str, image_type: str, values: list[float], **overrides: object) -> CatalogImage:
    embedding = _unit(values)
    return CatalogImage(
        model=model,
        image_type=image_type,
        embedding=embedding,
        embedding_model=str(overrides.get("embedding_model", "google/siglip-base-patch16-224")),
        embedding_dimension=int(overrides.get("embedding_dimension", embedding.size)),
        model_id=overrides.get("model_id"),  # type: ignore[arg-type]
    )


def test_ranks_individual_images_not_model_average() -> None:
    catalog = [
        _row("52792_006", "main", [0.2, 0.8, 0.0]),
        _row("52792_006", "side", [1.0, 0.0, 0.0]),
        _row("52160_001", "main", [0.7, 0.7, 0.0]),
        _row("52160_001", "pers", [0.8, 0.2, 0.0]),
    ]
    query = _unit([1.0, 0.0, 0.0])
    hits = rank_catalog_images(query, catalog)
    assert [(hit.model, hit.image_type) for hit in hits[:2]] == [
        ("52792_006", "side"),
        ("52160_001", "pers"),
    ]
    side = find_image_hit(hits, "52792_006", "side")
    assert side is not None and side.rank == 1
    models = aggregate_models_max(hits)
    assert models[0].model == "52792_006"
    assert models[0].best_image_type == "side"
    assert models[0].model_score == side.similarity


def test_max_aggregation_does_not_average_weak_views() -> None:
    catalog = [
        _row("pattern", "side", [1.0, 0.0, 0.0]),
        _row("pattern", "main", [0.0, 1.0, 0.0]),
        _row("plain", "main", [0.6, 0.4, 0.0]),
        _row("plain", "pers", [0.6, 0.4, 0.0]),
        _row("plain", "side", [0.6, 0.4, 0.0]),
    ]
    query = _unit([1.0, 0.0, 0.0])
    models = aggregate_models_max(rank_catalog_images(query, catalog))
    assert models[0].model == "pattern"
    mean_pattern = float(np.mean([models[0].model_score, float(np.dot(query, catalog[1].embedding))]))
    assert mean_pattern < models[1].model_score


def test_self_match_is_rank_one() -> None:
    target = _row("52792_006", "side", [0.3, 0.4, 0.5])
    catalog = [
        _row("52160_001", "pers", [0.9, 0.1, 0.0]),
        target,
        _row("40724_001", "main", [0.1, 0.9, 0.0]),
    ]
    hits = rank_catalog_images(target.embedding, catalog)
    assert hits[0].model == "52792_006"
    assert hits[0].image_type == "side"
    np.testing.assert_allclose(hits[0].similarity, 1.0, atol=1e-6)


def test_integrity_flags_wrong_dim_and_norm() -> None:
    rows = [
        CatalogImage(
            model="A",
            image_type="main",
            embedding=np.array([1.0, 0.0], dtype=np.float32),
            embedding_model="other-model",
            embedding_dimension=2,
        ),
        CatalogImage(
            model="B",
            image_type="side",
            embedding=np.array([3.0, 0.0, 4.0], dtype=np.float32),
            embedding_model="google/siglip-base-patch16-224",
            embedding_dimension=768,
        ),
    ]
    report = inspect_catalog_integrity(rows, expected_dimension=3)
    codes = {issue.code for issue in report.issues}
    assert "embedding_model" in codes
    assert "dimension" in codes
    assert "norm" in codes
    flags = inspect_vector(np.array([1.0, 0.0], dtype=np.float32), embedding_model="x", expected_dimension=3)
    assert any("dimension" in flag for flag in flags)


def test_catalog_from_model_view_keeps_views_separate() -> None:
    rows = catalog_from_model_view(
        [
            {
                "model": "52792_006",
                "model_id": 9,
                "embedding_model": "google/siglip-base-patch16-224",
                "embedding_dimension": 3,
                "main_embedding": [1.0, 0.0, 0.0],
                "pers_embedding": None,
                "side_embedding": [0.0, 1.0, 0.0],
            }
        ]
    )
    assert {(row.model, row.image_type) for row in rows} == {("52792_006", "main"), ("52792_006", "side")}
    assert all(row.source_url.endswith(suffix) for row, suffix in zip(rows, (".jpg", "_side.jpg")))


def test_classify_search_bug_vs_retrieval_limit() -> None:
    expected_side = ImageHit(rank=1, model="52792_006", image_type="side", similarity=0.90)
    compare = [ImageHit(rank=2, model="52160_001", image_type="pers", similarity=0.70)]
    wrong = ModelHit(
        rank=1,
        model="52160_001",
        model_score=0.70,
        best_image_type="pers",
        best_image_similarity=0.70,
    )
    assert (
        classify_failure(
            expected="52792_006",
            compare="52160_001",
            expected_side=expected_side,
            compare_hits=compare,
            production=wrong,
            expected_self_test_ok=True,
        )
        == "A"
    )
    genuine = ImageHit(rank=4, model="52792_006", image_type="side", similarity=0.66)
    compare_win = [ImageHit(rank=1, model="52160_001", image_type="pers", similarity=0.69)]
    assert (
        classify_failure(
            expected="52792_006",
            compare="52160_001",
            expected_side=genuine,
            compare_hits=compare_win,
            production=wrong,
            expected_self_test_ok=True,
        )
        == "C"
    )
    assert (
        classify_failure(
            expected="52792_006",
            compare="52160_001",
            expected_side=genuine,
            compare_hits=compare_win,
            production=wrong,
            expected_self_test_ok=False,
        )
        == "B"
    )
