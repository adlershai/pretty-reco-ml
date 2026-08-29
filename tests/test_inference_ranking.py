"""Shoe→customer ranking and vector sanity tests."""

from __future__ import annotations

import numpy as np
import pytest

from inference.csv_models import ANONYMOUS_CUSTOMER_ID
from inference.customers import encode_customer_vectors
from inference.models import encode_catalog_models, missing_attributes, model_status_rows
from inference.ranking import (
    CUSTOMER_TOP_MODELS,
    enrich_customer_ranking,
    overlap_stats,
    rank_customers_for_models,
    rank_models_for_customers,
    score_matrix,
)
from inference.sanity import check_customer_ranking, check_vectors
from tests.two_tower_fixtures import make_model, unit_visual
from training.config import UNKNOWN_TOKEN
from training.dataset import CatalogModel
from training.feature_encoders import categorical_string


def _l2(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return (matrix / norms).astype(np.float32)


def test_score_matrix_is_cosine_for_unit_vectors() -> None:
    customers = _l2(np.array([[1.0, 0.0], [0.6, 0.8], [0.0, 1.0]], dtype=np.float32))
    models = _l2(np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))
    scores = score_matrix(customers, models)
    ranked = rank_customers_for_models(scores, ["a", "b", "c"], ["M1", "M2"], top_k=3)
    m1 = [row for row in ranked if row["model"] == "M1"]
    assert [row["customer_id"] for row in m1] == ["a", "b", "c"]
    assert [row["customer_rank"] for row in m1] == [1, 2, 3]
    assert m1[0]["similarity_score"] >= m1[1]["similarity_score"] >= m1[2]["similarity_score"]
    check_customer_ranking(ranked, new_model_codes={"M1", "M2"})


def test_ranking_rejects_anonymous_and_duplicates() -> None:
    rows = [
        {"model": "M1", "customer_rank": 1, "customer_id": ANONYMOUS_CUSTOMER_ID, "similarity_score": 0.9},
        {"model": "M1", "customer_rank": 2, "customer_id": "1", "similarity_score": 0.1},
    ]
    with pytest.raises(ValueError, match="anonymous"):
        check_customer_ranking(rows, new_model_codes={"M1"})


def test_customer_top_models_keeps_six() -> None:
    rng = np.random.default_rng(0)
    customers = _l2(rng.normal(size=(5, 4)).astype(np.float32))
    models = _l2(rng.normal(size=(8, 4)).astype(np.float32))
    scores = score_matrix(customers, models)
    codes = [f"N{i}" for i in range(8)]
    ids = [str(i) for i in range(5)]
    matches = rank_models_for_customers(scores, ids, codes, top_k=CUSTOMER_TOP_MODELS)
    for customer_id in ids:
        group = [row for row in matches if row["customer_id"] == customer_id]
        assert len(group) == 6
        assert [row["rank_for_customer"] for row in group] == [1, 2, 3, 4, 5, 6]
        scores_desc = [row["similarity_score"] for row in group]
        assert scores_desc == sorted(scores_desc, reverse=True)


def test_overlap_stats_count_repeated_customers() -> None:
    ranked = []
    for model in ("A", "B", "C"):
        ranked.append({"model": model, "customer_rank": 1, "customer_id": "1", "similarity_score": 0.9})
        ranked.append({"model": model, "customer_rank": 2, "customer_id": model, "similarity_score": 0.1})
    stats = overlap_stats(ranked, top_n=25)
    assert stats["unique_customers"] == 4
    assert stats["highest_repeated_customer_id"] == "1"
    assert stats["highest_repeated_customer_count"] == 3
    assert stats["max_models_assigned_to_one_customer"] == 3


def test_encode_new_models_ignores_sales_history_and_uses_unk() -> None:
    model, encoders, _config = make_model(8)
    visual = unit_visual(9)
    catalog = {
        "NEW": CatalogModel(
            code="NEW",
            features={
                "main_category": "WOMEN",
                "sub_category": "Shoes",
                "material": "Leather",
                "color": "brand-new-never-seen",
                "last_type": "angelis",
            },
            visual=visual,
        )
    }
    vectors, codes = encode_catalog_models(model, catalog, ["NEW"])
    assert codes == ["NEW"]
    check_vectors(vectors, dim=8, name="models")
    encoded = encoders.encode_model_categoricals(catalog["NEW"].features)
    assert encoded["color"] == encoders.categoricals["color"].token_to_index[UNKNOWN_TOKEN]
    assert categorical_string(None) == UNKNOWN_TOKEN


def test_model_status_reports_missing_visual() -> None:
    rows = [{"model": "A", "main_embedding": None, "pers_embedding": None, "side_embedding": None}]
    status = model_status_rows(["A", "B"], rows)
    assert status[0]["visual_available"] is False
    assert status[1]["missing_attributes"] == ["not_in_catalog"]
    assert "color" in missing_attributes({"main_category": "WOMEN"})


def test_customer_encode_unit_norm_and_enrichment() -> None:
    tower, _encoders, _config = make_model(8)
    visual = unit_visual(1)
    catalog = {
        "OLD": CatalogModel(
            code="OLD",
            features={
                "main_category": "WOMEN",
                "sub_category": "Shoes",
                "material": "Leather",
                "color": "black",
                "last_type": "angelis",
            },
            visual=visual,
        )
    }
    records = [
        {
            "customer_id": "10",
            "customer_name": "Dana",
            "catalog_history_ids": ["OLD"],
            "static": {"age_group": "18-29", "tenure_days": 40},
            "behavior": {
                "purchase_count_before": 1,
                "history_length": 1,
                "recency_days": 3,
                "full_price_ratio_before": 1.0,
                "discount_ratio_before": 0.0,
                "avg_discount_before": 0.0,
            },
            "history_length": 1,
            "last_purchase_date": "2026-01-01",
            "preferred_size": "37",
            "last_purchased_size": "37",
        }
    ]
    vectors = encode_customer_vectors(tower, records, catalog)
    check_vectors(vectors, dim=8, name="customers")
    ranked = enrich_customer_ranking(
        [{"model": "NEW", "customer_rank": 1, "customer_id": "10", "similarity_score": 0.5, "customer_index": 0}],
        records,
    )
    assert ranked[0]["customer_name"] == "Dana"
    assert ranked[0]["preferred_size"] == "37"
