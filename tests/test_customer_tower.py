from __future__ import annotations

import numpy as np
import torch

from tests.two_tower_fixtures import example_model, make_model


def test_customer_tower_output_shape_and_norm() -> None:
    model, encoders, config = make_model(8)
    a = example_model("A", 1)
    b = example_model("B", 2, color="red")
    vector = model.encode_customer(
        {
            "age_group": "18-29",
            "static": {"age_group": "18-29", "tenure_days": 40},
            "behavior": {
                "purchase_count_before": 2,
                "history_length": 2,
                "recency_days": 5,
                "full_price_ratio_before": 1.0,
                "discount_ratio_before": 0.0,
                "avg_discount_before": 0.0,
            },
            "history_models": [a, b],
        }
    )
    assert vector.shape == (config.embedding_dim,)
    np.testing.assert_allclose(np.linalg.norm(vector), 1.0, rtol=1e-5)


def test_updated_history_changes_vector_without_retraining() -> None:
    model, _encoders, _config = make_model(8)
    model.eval()
    a = example_model("A", 1)
    b = example_model("B", 2, color="red")
    c = example_model("C", 3, last_type="rosario")
    payload = {
        "static": {"age_group": "30-39", "tenure_days": 80},
        "behavior": {
            "purchase_count_before": 2,
            "history_length": 2,
            "recency_days": 3,
            "full_price_ratio_before": 0.5,
            "discount_ratio_before": 0.5,
            "avg_discount_before": 0.1,
        },
    }
    first = model.encode_customer({**payload, "history_models": [a, b]})
    payload["behavior"] = {
        **payload["behavior"],
        "purchase_count_before": 3,
        "history_length": 3,
    }
    second = model.encode_customer({**payload, "history_models": [a, b, c]})
    assert first.shape == second.shape == (8,)
    assert not np.allclose(first, second, atol=1e-6)
    for param in model.parameters():
        assert param.grad is None
