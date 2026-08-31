"""Like-score calibration and API schema tests."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pytest

from inference.csv_models import ANONYMOUS_CUSTOMER_ID
from inference.like_score import LikeCalibrator, load_calibrator
from inference.recency import rank_customers_with_recency
from inference.recommender import DEFAULT_ARTIFACT


TODAY = date(2026, 8, 31)


def _iso(days_ago: int) -> str:
    return (TODAY - timedelta(days=days_ago)).isoformat()


def _calibrator() -> LikeCalibrator:
    positives = np.concatenate(
        [
            np.full(40, 0.70),
            np.full(30, 0.55),
            np.full(10, 0.40),
        ]
    )
    negatives = np.concatenate(
        [
            np.full(200, 0.10),
            np.full(80, 0.25),
            np.full(20, 0.40),
        ]
    )
    return LikeCalibrator.fit_pairs(
        positives,
        negatives,
        metadata={"candidate_sampling_method": "unit_test"},
    )


def test_like_score_is_always_in_unit_interval() -> None:
    calibrator = _calibrator()
    mapped = calibrator.transform(np.linspace(-1.0, 1.0, 50))
    assert float(mapped.min()) >= 0.0
    assert float(mapped.max()) <= 1.0


def test_stronger_similarity_never_maps_to_lower_like_score() -> None:
    calibrator = _calibrator()
    raw = np.array([-0.2, 0.0, 0.2, 0.4, 0.55, 0.7, 0.9], dtype=np.float32)
    mapped = calibrator.transform(raw)
    assert np.all(np.diff(mapped) >= -1e-6)


def test_mapping_is_not_a_linear_shift_of_cosine() -> None:
    calibrator = _calibrator()
    raw = np.array([-0.2, 0.1, 0.4, 0.7], dtype=np.float32)
    mapped = calibrator.transform(raw)
    linear = (raw + 1.0) / 2.0
    assert not np.allclose(mapped, linear, atol=0.02)


def test_raw_similarity_stays_on_ranked_rows() -> None:
    calibrator = _calibrator()
    similarities = np.array([0.70, 0.20], dtype=np.float32)
    likes = calibrator.transform(similarities)
    rows, _diagnostics = rank_customers_with_recency(
        similarities,
        ["1", "2"],
        [_iso(400), _iso(400)],
        like_scores=likes,
        top_k=2,
        now=TODAY,
    )
    assert rows[0]["similarity_score"] == pytest.approx(0.70, abs=1e-6)
    assert rows[0]["like_score"] == pytest.approx(float(likes[0]), abs=1e-6)
    assert rows[0]["like_score"] != pytest.approx(rows[0]["similarity_score"], abs=1e-3)


def test_recency_does_not_alter_like_score() -> None:
    calibrator = _calibrator()
    similarity = np.array([0.62, 0.62], dtype=np.float32)
    likes = calibrator.transform(similarity)
    rows, _diagnostics = rank_customers_with_recency(
        similarity,
        ["recent", "dormant"],
        [_iso(80), _iso(1500)],
        like_scores=likes,
        top_k=2,
        now=TODAY,
    )
    assert rows[0]["like_score"] == pytest.approx(rows[1]["like_score"], abs=1e-6)
    assert rows[0]["similarity_score"] == pytest.approx(rows[1]["similarity_score"], abs=1e-6)


def test_ranking_after_eligibility_is_like_score_desc() -> None:
    likes = np.array([0.40, 0.91, 0.70], dtype=np.float32)
    rows, _diagnostics = rank_customers_with_recency(
        np.array([0.2, 0.8, 0.5], dtype=np.float32),
        ["a", "b", "c"],
        [_iso(200), _iso(400), _iso(800)],
        like_scores=likes,
        top_k=3,
        now=TODAY,
    )
    assert [row["customer_id"] for row in rows] == ["b", "c", "a"]
    assert [row["like_score"] for row in rows] == pytest.approx([0.91, 0.70, 0.40], abs=1e-5)


def test_limit_100_returns_100_when_enough_eligible() -> None:
    n_recent = 50
    n_eligible = 120
    similarities = np.concatenate(
        [
            np.full(n_recent, 0.95, dtype=np.float32),
            np.linspace(0.1, 0.4, n_eligible, dtype=np.float32),
        ]
    )
    ids = [str(i) for i in range(n_recent + n_eligible)]
    dates = [_iso(5)] * n_recent + [_iso(400)] * n_eligible
    ranked, diagnostics = rank_customers_with_recency(
        similarities,
        ids,
        dates,
        like_scores=similarities,
        top_k=100,
        now=TODAY,
    )
    assert diagnostics["eligible"] >= 100
    assert len(ranked) == 100
    assert [row["rank"] for row in ranked] == list(range(1, 101))


def test_anonymous_customer_never_appears() -> None:
    ranked, _diagnostics = rank_customers_with_recency(
        np.array([0.99, 0.1], dtype=np.float32),
        [ANONYMOUS_CUSTOMER_ID, "2"],
        [_iso(800), _iso(800)],
        top_k=10,
        now=TODAY,
    )
    assert [row["customer_id"] for row in ranked] == ["2"]


def test_customer_ids_are_unique() -> None:
    ranked, _diagnostics = rank_customers_with_recency(
        np.array([0.3, 0.3, 0.2], dtype=np.float32),
        ["1", "2", "3"],
        [_iso(200), _iso(400), _iso(800)],
        top_k=10,
        now=TODAY,
    )
    ids = [row["customer_id"] for row in ranked]
    assert len(ids) == len(set(ids))


def test_production_calibrator_loads_with_v1_artifact() -> None:
    calibrator = load_calibrator(DEFAULT_ARTIFACT)
    mapped = calibrator.transform([0.0, 0.5, 0.9])
    assert mapped.shape == (3,)
    assert float(mapped.min()) >= 0.0
    assert float(mapped.max()) <= 1.0
    assert calibrator.metadata.get("calibration_method") == "isotonic_pav"


def test_unseen_similarity_is_clipped_not_rejected() -> None:
    calibrator = _calibrator()
    mapped = calibrator.transform([-5.0, 5.0])
    assert float(mapped[0]) >= 0.0
    assert float(mapped[1]) <= 1.0


def test_calibrator_roundtrip(tmp_path: Path) -> None:
    calibrator = _calibrator()
    path = tmp_path / "like_calibrator.json"
    calibrator.dump(path)
    loaded = LikeCalibrator.load(path)
    raw = np.array([0.1, 0.4, 0.7], dtype=np.float32)
    assert np.allclose(calibrator.transform(raw), loaded.transform(raw), atol=1e-6)
