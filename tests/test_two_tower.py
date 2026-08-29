from __future__ import annotations

import numpy as np
import torch

from tests.two_tower_fixtures import example_model, make_model, tiny_batch
from training.artifact import load_artifact, save_artifact
from training.two_tower import in_batch_softmax_loss


def test_towers_share_configured_dimension() -> None:
    model, _encoders, config = make_model(8)
    batch = tiny_batch(model.encoders, config)
    customers, models = model(batch)
    assert customers.shape == models.shape == (4, 8)
    np.testing.assert_allclose(
        torch.linalg.vector_norm(customers, dim=-1).detach().numpy(),
        np.ones(4),
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        torch.linalg.vector_norm(models, dim=-1).detach().numpy(),
        np.ones(4),
        rtol=1e-5,
        atol=1e-5,
    )


def test_similarity_is_numeric_dot_product() -> None:
    model, _encoders, _config = make_model(8)
    a = example_model("A", 1)
    customer = model.encode_customer(
        {
            "static": {"age_group": "18-29", "tenure_days": 12},
            "behavior": {
                "purchase_count_before": 1,
                "history_length": 1,
                "recency_days": 4,
                "full_price_ratio_before": 1.0,
                "discount_ratio_before": 0.0,
                "avg_discount_before": 0.0,
            },
            "history_models": [a],
        }
    )
    shoe = model.encode_model(example_model("B", 2, color="red"))
    score = float(customer @ shoe)
    assert np.isfinite(score)
    assert -1.01 <= score <= 1.01


def test_save_load_reproduces_vectors(tmp_path) -> None:
    model, encoders, config = make_model(8)
    shoe = example_model("B", 2, color="red")
    history = {
        "static": {"age_group": "18-29", "tenure_days": 12},
        "behavior": {
            "purchase_count_before": 1,
            "history_length": 1,
            "recency_days": 4,
            "full_price_ratio_before": 1.0,
            "discount_ratio_before": 0.0,
            "avg_discount_before": 0.0,
        },
        "history_models": [example_model("A", 1)],
    }
    before_c = model.encode_customer(history)
    before_m = model.encode_model(shoe)
    save_artifact(
        artifact_dir=tmp_path,
        model=model,
        config=config,
        encoders=encoders,
        metrics={"validation": {"recall@10": 0.1}},
        metadata={"model_version": "test"},
    )
    loaded = load_artifact(tmp_path)
    after_c = loaded.encode_customer(history)
    after_m = loaded.encode_model(shoe)
    np.testing.assert_allclose(before_c, after_c, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(before_m, after_m, rtol=1e-5, atol=1e-5)
